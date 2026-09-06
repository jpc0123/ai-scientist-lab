"""Instrumented DetSolver.fit that samples mechanism diagnostics at fixed epochs."""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from mechanism_diagnostics import (
    append_csv,
    run_activation_probe,
    run_gradient_probe,
    write_json,
)


def install_instrumented_fit(
    solver: Any,
    *,
    arm: str,
    diag_dir: Path,
    sample_epochs_1based: list[int],
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
) -> None:
    """Replace solver.fit with a copy of vendor fit that records probes."""
    from src.misc import dist_utils, stats  # type: ignore
    from src.solver.det_engine import evaluate, train_one_epoch  # type: ignore

    diag_dir = Path(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    sample_set = {int(x) for x in sample_epochs_1based}
    gate_csv = diag_dir / "gate_statistics.csv"
    feat_csv = diag_dir / "feature_statistics.csv"
    sim_csv = diag_dir / "feature_similarity.csv"
    grad_csv = diag_dir / "gradient_norms.csv"
    cos_csv = diag_dir / "gradient_cosines.csv"
    epoch_trace: list[dict[str, Any]] = []

    def _should_sample(epoch_0based: int, is_last: bool, is_best_now: bool) -> list[str]:
        ep1 = epoch_0based + 1
        tags: list[str] = []
        if ep1 in sample_set:
            tags.append(f"epoch_{ep1}")
        if is_last:
            tags.append("last")
        if is_best_now:
            tags.append("best")
        # Deduplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _probe(epoch_0based: int, tags: list[str], test_stats: dict[str, Any]) -> None:
        if not tags:
            return
        loader = solver.train_dataloader
        samples, targets = next(iter(loader))
        device = solver.device
        samples = samples.to(device)
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]
        module = solver.ema.module if solver.ema else solver.model
        for tag in tags:
            act = run_activation_probe(
                module,
                samples,
                arm=arm,
                epoch_1based=epoch_0based + 1,
                tag=tag,
            )
            append_csv(
                gate_csv,
                act["gates"],
                [
                    "arm",
                    "epoch",
                    "tag",
                    "level",
                    "gate_mean",
                    "gate_std",
                    "gate_min",
                    "gate_max",
                    "gate_lt_0_1_ratio",
                    "gate_gt_0_9_ratio",
                    "gate_entropy",
                    "rgb_weight_mean",
                    "thermal_weight_mean",
                ],
            )
            append_csv(
                feat_csv,
                act["features"],
                [
                    "arm",
                    "epoch",
                    "tag",
                    "level",
                    "role",
                    "feature_mean",
                    "feature_std",
                    "l2_norm",
                    "channel_variance",
                    "zero_activation_ratio",
                ],
            )
            append_csv(
                sim_csv,
                act["similarities"],
                ["arm", "epoch", "tag", "level", "pair", "cosine"],
            )
            grads = run_gradient_probe(
                solver.model,
                solver.criterion,
                samples,
                targets,
                arm=arm,
                epoch_1based=epoch_0based + 1,
                tag=tag,
                device=device,
            )
            append_csv(
                grad_csv,
                grads["grad_rows"],
                [
                    "arm",
                    "epoch",
                    "tag",
                    "module",
                    "grad_l2",
                    "grad_max",
                    "zero_grad_param_ratio",
                    "n_params",
                ],
            )
            append_csv(
                cos_csv,
                [grads["cosine_row"]],
                [
                    "arm",
                    "epoch",
                    "tag",
                    "fusion_fdpn_grad_cosine",
                    "rgb_thermal_grad_ratio",
                ],
            )
            epoch_trace.append(
                {
                    "arm": arm,
                    "epoch": epoch_0based + 1,
                    "tag": tag,
                    "test_AP": (test_stats.get("coco_eval_bbox") or [None])[0],
                    "probe_loss": grads.get("loss"),
                }
            )

    def fit_with_diag(self=solver) -> None:  # noqa: ANN001
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training (mechanism diagnosis)" + "-" * 20)
        top1 = 0.0
        best_stat: dict[str, Any] = {"epoch": -1}
        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            backbone_actual_lr = float(self.optimizer.param_groups[0]["lr"])
            non_backbone_actual_lr = float(
                max(float(pg["lr"]) for pg in self.optimizer.param_groups)
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            scheduler_last_epoch = int(getattr(self.lr_scheduler, "last_epoch", -1))

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            is_best_now = False
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}", v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    is_best_now = True
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {"epoch": -1}
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                "backbone_actual_lr": backbone_actual_lr,
                "non_backbone_actual_lr": non_backbone_actual_lr,
                "scheduler_last_epoch": scheduler_last_epoch,
            }
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            self.output_dir / "eval" / "latest.pth",
                        )

            is_last = epoch == args.epochs - 1
            tags = _should_sample(epoch, is_last=is_last, is_best_now=is_best_now)
            try:
                _probe(epoch, tags, test_stats)
            except Exception as exc:  # noqa: BLE001
                print(f"[mechanism_diag] probe failed at epoch {epoch}: {exc}")

            if on_epoch_end is not None:
                on_epoch_end(epoch, test_stats)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        write_json(
            diag_dir / f"{arm}_TRACE.json",
            {
                "arm": arm,
                "sample_epochs_1based": sorted(sample_set),
                "epoch_trace": epoch_trace,
                "best_stat": best_stat_print,
                "training_time_sec": total_time,
                "metric_names": metric_names,
            },
        )

    solver.fit = fit_with_diag.__get__(solver, type(solver))
