"""v2.6-P1 LiteratureRetriever. No GPU. Fake corpus needs no API key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scientist_lab.cli import main
from scientist_lab.core.claim_gate import evaluate_claim
from scientist_lab.core.schema_registry import validate_named
from scientist_lab.literature.factory import resolve_literature_provider
from scientist_lab.literature.fake_provider import FakeLiteratureProvider
from scientist_lab.literature.gate import gate_paper
from scientist_lab.literature.models import PaperRecord
from scientist_lab.literature.retriever import LiteratureRetriever
from scientist_lab.literature.semantic_scholar import SemanticScholarProvider
from scientist_lab.llm.errors import MissingAPIKeyError, RealProviderNotEnabledError, UnsupportedProviderError
from scientist_lab.llm.http_transport import HttpResponse, MockTransport


def _claim(**overrides):
    doc = {
        "schema_version": "1.0.0",
        "claim_id": "claim_lit_001",
        "claim_type": "component_effectiveness",
        "claim_text": "FDPN is effective because papers say so",
        "claim_strength": "C2",
        "metric": "APS",
        "asserts": {"component_effective": True},
    }
    doc.update(overrides)
    return doc


def test_fake_search_and_gate_admit_planner(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(
        live=False,
        provenance_dir=tmp_path / "prov",
        cache_dir=tmp_path / "cache",
    )
    packet = retriever.search(
        "low-light RGB-T small object detection",
        year_from=2022,
        limit=10,
        round_id="round_01",
        used_by_plan=["S2:lowlight-rgbt-001"],
    )
    assert packet["ok"] is True
    assert packet["provider"] == "fake"
    assert packet["live"] is False
    assert packet["provenance"]["used_by_plan"] == ["S2:lowlight-rgbt-001"]
    assert packet["planner_admissible"]
    for row in packet["planner_admissible"]:
        validate_named("literature_evidence", row)
        assert row["can_enter_claim_gate"] is False
        assert "single_paper_is_not_field_consensus" in row["blocked_statements"]
        assert "literature_cannot_substitute_experiment_evidence" in row["blocked_statements"]
    validate_named("literature_query", packet["provenance"])


def test_gate_metadata_only_is_not_planner_admissible() -> None:
    paper = PaperRecord(
        paper_id="S2:incomplete",
        title="A title",
        source="fake",
        retrieval_query="q",
        retrieved_at="2026-08-18T00:00:00+00:00",
        year=None,
        abstract=None,
    )
    packet = gate_paper(paper)
    assert packet["planner_admissible"] is False
    assert packet["completeness"] == "metadata"
    assert "abstract" in str(packet["reject_reason"])


def test_claim_gate_blocks_literature_evidence() -> None:
    lit = gate_paper(FakeLiteratureProvider().search("thermal fusion", limit=1)[0])
    verdict = evaluate_claim(_claim(), evidence=lit)
    assert verdict["status"] == "BLOCKED"
    assert "LiteratureEvidence" in verdict["reason"]


def test_claim_gate_does_not_treat_gpu_evidence_with_lit_provenance_as_literature() -> None:
    """Round provenance on ExperimentEvidence is not LiteratureEvidence."""
    evidence = {
        "evidence_status": "VALID",
        "run_state": "COMPLETED",
        "run_level": "probe",
        "budget_class": "probe",
        "run_id": "run_probe_with_lit_ref",
        "metrics": {"APS": 0.0, "mAP50_95": 0.0, "mAP50": 0.0},
        "result": {
            "run_id": "run_probe_with_lit_ref",
            "metrics": {"APS": 0.0, "mAP50_95": 0.0, "mAP50": 0.0},
            "execution": {"status": "success"},
        },
        "fingerprint_comparable": True,
        "review_decision": "KEEP",
        "baseline": {"present": False},
        "literature_query_id": "litq_round_01",
        "paper_refs": ["S2:lowlight-rgbt-001"],
    }
    verdict = evaluate_claim(
        _claim(claim_type="observational", claim_strength="C0", claim_text="probe APS under protocol"),
        evidence=evidence,
    )
    assert "LiteratureEvidence" not in verdict["reason"]
    assert "cannot substitute ExperimentEvidence" not in verdict["reason"]


def test_claim_gate_blocks_retriever_search_packet() -> None:
    packet = FakeLiteratureProvider().search("thermal", limit=1)
    gated = gate_paper(packet[0])
    search_blob = {
        "ok": True,
        "papers": [gated],
        "planner_admissible": [gated],
        "claim_gate_note": "LiteratureEvidence cannot enter ClaimGate",
        "provenance": {"literature_query_id": "litq_accidental"},
    }
    verdict = evaluate_claim(_claim(), evidence=search_blob)
    assert verdict["status"] == "BLOCKED"
    assert "LiteratureEvidence" in verdict["reason"]


def test_live_without_key_fail_closed() -> None:
    with pytest.raises(MissingAPIKeyError):
        resolve_literature_provider(live=True, provider="semantic_scholar", environ={})


def test_live_fake_forbidden() -> None:
    with pytest.raises(RealProviderNotEnabledError):
        resolve_literature_provider(live=True, provider="fake")


def test_future_provider_reserved() -> None:
    with pytest.raises(UnsupportedProviderError):
        resolve_literature_provider(live=False, provider="arxiv")


def test_semantic_scholar_search_uses_mock_transport_and_redacts_key(tmp_path: Path) -> None:
    body = json.dumps(
        {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "RGB thermal fusion low illumination",
                    "year": 2024,
                    "authors": [{"name": "Ada"}],
                    "abstract": "Thermal can complement degraded RGB in low light.",
                    "externalIds": {"DOI": "10.1/fake", "ArXiv": "2401.00001"},
                    "venue": "Test",
                    "citationCount": 9,
                    "url": "https://example.invalid/abc123",
                }
            ]
        }
    )
    transport = MockTransport(
        responses=[HttpResponse(status_code=200, body=body, headers={})]
    )
    provider = SemanticScholarProvider(api_key="secret-s2-key", transport=transport)
    papers = provider.search("RGB thermal fusion low illumination", year_from=2022, limit=5)
    assert len(papers) == 1
    assert papers[0].paper_id == "S2:abc123"
    assert papers[0].doi == "10.1/fake"
    assert "secret-s2-key" not in json.dumps(transport.calls)
    assert transport.calls[0]["headers"]["x-api-key"] == "[REDACTED]"
    assert "paper/search" in transport.calls[0]["url"]
    assert "year=2022-" in transport.calls[0]["url"]


def test_semantic_scholar_http_error_fail_closed() -> None:
    from scientist_lab.literature.errors import LiteratureProviderError

    transport = MockTransport(
        responses=[HttpResponse(status_code=401, body='{"error":"unauthorized"}', headers={})]
    )
    provider = SemanticScholarProvider(api_key="secret-s2-key", transport=transport)
    with pytest.raises(LiteratureProviderError):
        provider.search("rgb-t", year_from=2022, limit=5)


def test_semantic_scholar_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "scientist_lab.literature.semantic_scholar.time.sleep",
        lambda s: sleeps.append(float(s)),
    )
    ok_body = json.dumps({"data": [{"paperId": "abc", "title": "T", "year": 2024, "authors": []}]})
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=429, body='{"message":"Too Many Requests","code":"429"}', headers={}),
            HttpResponse(status_code=200, body=ok_body, headers={}),
        ]
    )
    provider = SemanticScholarProvider(api_key="secret-s2-key", transport=transport)
    papers = provider.search("rgb-t", year_from=2022, limit=5)
    assert len(papers) == 1
    assert len(transport.calls) == 2
    assert sleeps == [5.0]


def test_cache_avoids_second_provider_call(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(
        live=False,
        provenance_dir=tmp_path / "prov",
        cache_dir=tmp_path / "cache",
    )
    first = retriever.search("adaptive RGB thermal fusion", round_id="round_02")
    second = retriever.search("adaptive RGB thermal fusion", round_id="round_02")
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["provenance"]["paper_refs"] == first["provenance"]["paper_refs"]


def test_empty_search_is_not_cached(tmp_path: Path) -> None:
    """Empty S2 misses must not stick in cache (Live A polluted-query poison)."""
    transport = MockTransport(
        responses=[
            HttpResponse(status_code=200, body=json.dumps({"data": []})),
            HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "data": [
                            {
                                "paperId": "abc",
                                "title": "RGB-T low-light detection",
                                "year": 2024,
                                "url": "https://example.invalid/p",
                            }
                        ]
                    }
                ),
            ),
        ]
    )
    provider = SemanticScholarProvider(api_key="secret-s2-key", transport=transport)
    retriever = LiteratureRetriever(
        live=True,
        provider="semantic_scholar",
        environ={"SEMANTIC_SCHOLAR_API_KEY": "secret-s2-key"},
        transport=transport,
        provenance_dir=tmp_path / "prov",
        cache_dir=tmp_path / "cache",
    )
    # Force our provider instance (factory would rebuild).
    retriever.provider = provider
    first = retriever.search("polluted query", round_id="r0", year_from=2022, limit=5)
    assert first["papers"] == []
    assert list((tmp_path / "cache").glob("*.json")) == []
    second = retriever.search("polluted query", round_id="r1", year_from=2022, limit=5)
    assert second["cached"] is False
    assert len(second["papers"]) == 1


def test_s2_retries_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from scientist_lab.llm.errors import LLMTimeoutError

    sleeps: list[float] = []
    monkeypatch.setattr("scientist_lab.literature.semantic_scholar.time.sleep", sleeps.append)

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def request(self, *args, **kwargs):
            self.n += 1
            if self.n == 1:
                raise LLMTimeoutError("HTTP timeout: ConnectTimeout")
            return HttpResponse(
                status_code=200,
                body=json.dumps(
                    {
                        "data": [
                            {
                                "paperId": "xyz",
                                "title": "Recovered after timeout",
                                "year": 2023,
                            }
                        ]
                    }
                ),
            )

    provider = SemanticScholarProvider(api_key="secret-s2-key", transport=Flaky())
    papers = provider.search("rgb-t fusion", year_from=2022, limit=3)
    assert len(papers) == 1
    assert sleeps == [5.0]


def test_cli_literature_search_fake(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "literature-search",
            "--query",
            "low illumination thermal",
            "--output-dir",
            str(tmp_path / "lit"),
            "--no-cache",
            "--round-id",
            "round_01",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["provider"] == "fake"
    assert payload["planner_admissible"]


def test_cli_live_without_key_exits_1(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setattr(
        "scientist_lab.literature.runtime_secrets.apply_runtime_literature_env",
        lambda *_a, **_k: None,
    )
    code = main(["literature-search", "--live", "--query", "rgb-t"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "SEMANTIC_SCHOLAR_API_KEY" in payload["error"]


def test_cli_literature_refs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "literature-refs",
            "--paper-id",
            "S2:lowlight-rgbt-001",
            "--kind",
            "references",
            "--output-dir",
            str(tmp_path),
            "--no-cache",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "references"
    assert payload["ok"] is True
