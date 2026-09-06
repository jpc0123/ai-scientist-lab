# B0x matched baseline freeze (exploratory Fast Eval)

frozen_at: 2026-07-28T12:50:16.5346441+08:00
code_sha: 284fb6d57880424d366ddb49d1e9d965bddc136f
image_tag: scientist-rgbt-detection:v2-cuda
image_id: sha256:6b4dfd4c96d42f22d874d296545ce1205595b2fdc3abfbf585be574143142656
immutable_s00_tag: scientist-rgbt-detection:v2.3-cu128-torch2.7.1-s00
protocol_id: protocol_rgbt_cuda_001
dataset: dataset:rgbt_fast_eval_v1
split: split:fast_eval_fixed_v1
seed: 42
epochs: 1
batch_size: 2
image_size: 160x160
lr: 0.0002
num_workers: 0
mixed_precision: false
scale_queries_to_tokens: true (fast_eval only)
allowed_variable: input_mode + fusion_method only

claim_level: exploratory_comparison
formal_superiority: forbidden
