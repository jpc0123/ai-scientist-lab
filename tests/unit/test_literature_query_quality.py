"""Query synthesis + rerank quality. No GPU. No live network."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scientist_lab.core.claim_gate import evaluate_claim
from scientist_lab.core.how_pending import (
    persist_scout,
    planner_literature_context,
    resolve_scout_query,
    scout_literature_for_how,
    set_scout_intent,
)
from scientist_lab.core.schema_registry import SCHEMA_DIR, load_json, validate_named
from scientist_lab.literature.fake_provider import FakeLiteratureProvider
from scientist_lab.literature.gate import gate_paper
from scientist_lab.literature.models import PaperRecord
from scientist_lab.literature.query_synth import (
    _V26_STOCK,
    catalog_query_terms,
    compose_scout_queries,
    key_terms,
    last_ditch_fallback_query,
)
from scientist_lab.literature.display import (
    apply_identity_labels,
    pick_display_text,
    translate_ranked_papers,
)
from scientist_lab.literature.filter import filter_papers
from scientist_lab.literature.rerank import protocol_clues, rerank_papers
from scientist_lab.llm.gateway import ScriptedProvider
from scientist_lab.llm.planner_contract import (
    build_contract_input,
    build_planner_request,
    planner_system_prompt,
)

EXAMPLES = SCHEMA_DIR / "examples"


def _brightness_protocol() -> dict:
    proto = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    proto = dict(proto)
    proto["goal"] = {
        "improve": "brightness_bucket_error_analysis",
        "task_type": "object_detection",
        "notes": "Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    }
    proto["title"] = "Brightness-bucket error analysis on RGBT-Tiny"
    proto["notes"] = (
        "LLM research question: Does brightness-bucket error analysis change APS on RGBT-Tiny? "
        "KEEP ≠ Claim. This document is not a Claim."
    )
    proto["objective"] = {"primary": {"metric": "APS", "direction": "maximize"}}
    proto["condition_slice"] = None
    proto["baseline"] = dict(proto.get("baseline") or {})
    proto["baseline"]["dataset"] = "dataset:rgbt_tiny_v1"
    return proto


def _paper(**overrides) -> PaperRecord:
    row = {
        "paper_id": "S2:x",
        "title": "A title",
        "year": 2024,
        "abstract": "An abstract.",
        "doi": "10.0000/x",
        "url": "https://example.invalid/x",
        "source": "fake",
    }
    row.update(overrides)
    return PaperRecord.from_mapping(
        row,
        retrieval_query=str(row.get("retrieval_query") or "q"),
        source="fake",
        retrieved_at="2026-08-25T00:00:00+00:00",
    )


def test_no_key_fail_closed_no_forged_papers(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=True,
        round_id="round_no_key",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        environ={},
    )
    assert packet["fail_closed"] is True
    assert packet["papers"] == []
    assert packet["planner_admissible"] == []
    assert "没有实际检索" in str(packet.get("error") or "")
    scout = persist_scout(tmp_path / "how_pending.json", packet)["scout"]
    assert scout["papers"] == []
    assert scout["can_enter_claim_gate"] is False


def test_human_query_not_overwritten_by_compose() -> None:
    proto = _brightness_protocol()
    store = {
        "scout_intent": {
            "source": "human",
            "status": "active",
            "query": "search thermal-only F0 full-val APS",
            "why": "human direction",
        }
    }
    resolved = compose_scout_queries(
        store,
        proto,
        evidence={"last_review_decision": "DISCARD", "last_how_id": "F1"},
        live=False,
    )
    assert resolved["source"] == "human"
    assert resolved["fallback"] is False
    assert resolved["query"] == "search thermal-only F0 full-val APS"
    via_store = resolve_scout_query(store, proto)
    assert via_store["query"] == resolved["query"]
    assert via_store["source"] == "human"


def test_non_v26_research_question_enters_generated_query() -> None:
    proto = _brightness_protocol()
    v26 = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    resolved = compose_scout_queries({}, proto, evidence={"last_how_id": "F0"})
    query = resolved["query"].lower()
    assert resolved["source"] == "fallback"
    assert resolved["fallback"] is True
    assert "brightness" in query
    assert "bucket" in query or "brightness-bucket" in query
    assert "aps" in query
    stock = last_ditch_fallback_query(v26)
    assert "brightness-bucket" not in stock.lower()
    assert resolved["query"] != _V26_STOCK
    assert resolved["query"] != last_ditch_fallback_query(v26)
    terms = key_terms(proto["goal"]["notes"])
    assert any(tok.lower().startswith("brightness") for tok in terms)


def test_irrelevant_abstract_dropped_or_ranked_last() -> None:
    question = "Does brightness-bucket error analysis change APS on RGBT-Tiny?"
    relevant = _paper(
        paper_id="S2:keep-me",
        title="Brightness-bucket error analysis for RGB-T detection",
        abstract=(
            "We report APS by illumination / brightness buckets on RGBT-Tiny. "
            "Error analysis shows which bins fail."
        ),
    )
    survey = _paper(
        paper_id="S2:survey",
        title="A Comprehensive Survey of Deep Learning",
        abstract="We review CNNs on ImageNet classification and generic deep learning practice.",
    )
    other = _paper(
        paper_id="S2:nlp",
        title="Transformer language models for machine translation",
        abstract="BLEU scores on WMT. No detection, no APS, no brightness buckets.",
    )
    ranked = rerank_papers(
        [survey, other, relevant],
        research_question=question,
        query="brightness-bucket APS RGBT-Tiny",
        limit=8,
    )
    kept_ids = [str(row["paper"].paper_id) for row in ranked["kept"]]
    dropped_ids = [str(row["paper"].paper_id) for row in ranked["dropped"]]
    assert kept_ids[0] == "S2:keep-me"
    assert "S2:survey" in dropped_ids or kept_ids[-1] == "S2:survey"
    assert "S2:nlp" in dropped_ids or (len(kept_ids) > 1 and kept_ids[-1] == "S2:nlp")
    assert all(row["can_enter_claim_gate"] is False for row in ranked["kept"])


def test_scout_rerank_writes_why_relevant(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_rerank",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    assert packet["fail_closed"] is False
    papers = packet["papers"]
    assert papers
    ids = [str(row.get("paper_id") or "") for row in papers]
    assert "S2:dl-survey-999" not in ids
    assert any(row.get("why_relevant") for row in papers)
    assert any(row.get("score") is not None for row in papers)
    ctx = planner_literature_context(packet)
    assert ctx["can_enter_claim_gate"] is False
    assert ctx["clues"]
    assert "NOT Claim evidence" in ctx["usage"]
    assert ctx["clues"][0].get("title")
    for row in packet["planner_admissible"]:
        validate_named("literature_evidence", row)


def test_claim_gate_still_blocks_literature_packet(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_claim",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS",
    )
    claim = {
        "schema_version": "1.0.0",
        "claim_id": "claim_lit_quality",
        "claim_type": "component_effectiveness",
        "claim_text": "brightness buckets work because papers say so",
        "claim_strength": "C2",
        "metric": "APS",
        "asserts": {"component_effective": True},
    }
    verdict = evaluate_claim(claim, evidence=packet)
    assert verdict["status"] == "BLOCKED"
    assert "LiteratureEvidence" in verdict["reason"]
    if packet.get("planner_admissible"):
        gated = packet["planner_admissible"][0]
        assert gated["can_enter_claim_gate"] is False
        one = evaluate_claim(claim, evidence=gated)
        assert one["status"] == "BLOCKED"


def test_planner_prompt_forbids_literature_as_claim() -> None:
    prompt = planner_system_prompt()
    assert "MUST NOT enter ClaimGate" in prompt
    assert "unregistered HOW" in prompt
    proto = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v1.json")
    plan = {
        "plan_id": "p",
        "round_index": 0,
        "budget_class": "probe",
        "hypothesis": "h",
        "modification_scope": ["fusion"],
        "proposed_changes": [{"target": "fusion", "summary": "F3"}],
    }
    contract = build_contract_input(
        protocol=proto,
        previous_plan=plan,
        memory_refs={"lesson_ids": [], "strategy_ids": []},
        literature={
            "can_enter_claim_gate": False,
            "clues": [
                {
                    "paper_id": "S2:keep-me",
                    "title": "Brightness-bucket error analysis",
                    "year": 2024,
                    "why_relevant": "hits: brightness-bucket, APS",
                    "can_enter_claim_gate": False,
                }
            ],
            "usage": "Literature is a read-only clue. It is NOT Claim evidence.",
            "paper_refs": ["S2:keep-me"],
            "planner_admissible": [{"paper_id": "S2:keep-me"}],
        },
    )
    request = build_planner_request(contract)
    user = request.messages[1]["content"]
    assert "why_relevant" in user
    assert "NOT Claim evidence" in user
    assert "can_enter_claim_gate" in user


def test_fake_corpus_survey_exists_for_rerank() -> None:
    hits = FakeLiteratureProvider().search("deep learning survey", limit=10)
    assert any(p.paper_id == "S2:dl-survey-999" for p in hits)
    gated = gate_paper(hits[0] if hits[0].paper_id == "S2:dl-survey-999" else hits[-1])
    assert gated["can_enter_claim_gate"] is False


def test_set_human_intent_still_wins_over_brightness_protocol(tmp_path: Path) -> None:
    path = tmp_path / "how_pending.json"
    proto = _brightness_protocol()
    set_scout_intent(
        path,
        source="human",
        query="thermal-only F0 full val APS",
        why="human",
        status="active",
    )
    from scientist_lab.core.how_pending import load_store

    resolved = resolve_scout_query(load_store(path), proto, live=False)
    assert resolved["source"] == "human"
    assert resolved["query"] == "thermal-only F0 full val APS"
    assert "brightness-bucket" not in resolved["query"]


def test_discard_f3_query_uses_catalog_terms_not_unregistered_ops() -> None:
    proto = _brightness_protocol()
    resolved = compose_scout_queries(
        {},
        proto,
        evidence={"last_how_id": "F3", "last_review_decision": "DISCARD"},
    )
    query = resolved["query"].lower()
    assert resolved["fallback"] is True
    assert "gated" in query or "multiscale" in query
    assert "alternative" in query
    assert "late fusion" not in query
    assert "mid fusion" not in query
    # Bare HOW letter codes must not pollute the S2 string.
    assert not re.search(r"\bf3\b", query)
    assert "rgbt_tiny_v1" not in query
    assert "aps_lowlight" not in query
    terms = catalog_query_terms("F3")
    assert "F3" in terms
    assert any("gated" in t.lower() or "multiscale" in t.lower() for t in terms)


def test_strip_internal_tokens_and_v26_heuristic_avoids_lab_ids() -> None:
    from scientist_lab.literature.query_synth import (
        fingerprint_academic_terms,
        strip_internal_search_tokens,
    )

    polluted = (
        "RGB-T low-light small object detection early fusion improvement "
        "small-object D-FINE rgbt_tiny_v1 low_light_subset_v1 APS_lowlight F1"
    )
    cleaned = strip_internal_search_tokens(polluted)
    assert "rgbt_tiny_v1" not in cleaned.lower()
    assert "low_light_subset_v1" not in cleaned.lower()
    assert "aps_lowlight" not in cleaned.lower()
    assert "d-fine" not in cleaned.lower()
    assert "improvement" not in cleaned.lower()
    assert not re.search(r"\bf1\b", cleaned, re.IGNORECASE)
    assert "rgb-t" in cleaned.lower() or "rgb" in cleaned.lower()
    assert "low-light" in cleaned.lower() or "low" in cleaned.lower()
    assert len(cleaned.split()) <= 8

    v26 = load_json(EXAMPLES / "research_protocol_rgbt_dfine_v26.json")
    academic = fingerprint_academic_terms(v26)
    joined = " ".join(academic).lower()
    assert "rgbt_tiny_v1" not in joined
    assert "aps_lowlight" not in joined
    assert "d-fine" not in joined
    assert "rgb" in joined or "rgbt" in joined
    resolved = compose_scout_queries(
        {},
        v26,
        evidence={"last_how_id": "F1", "last_review_decision": "KEEP"},
    )
    q = resolved["query"].lower()
    assert "rgbt_tiny_v1" not in q
    assert "low_light_subset_v1" not in q
    assert "aps_lowlight" not in q
    assert "d-fine" not in q
    assert "rgbt-tiny" not in q
    assert "rgb" in q or "low" in q
    assert len(q.split()) <= 8


def test_scout_sanitizes_polluted_query_before_fake_search(tmp_path: Path) -> None:
    """Even if a caller passes lab ids, the fire path strips them."""
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_sanitize",
        provenance_dir=tmp_path / "literature",
        query=(
            "brightness-bucket APS RGBT-Tiny rgbt_tiny_v1 "
            "low_light_subset_v1 APS_lowlight"
        ),
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    assert packet["fail_closed"] is False
    assert packet["papers"]
    fired = str(packet.get("query") or "").lower()
    assert "rgbt_tiny_v1" not in fired
    assert "aps_lowlight" not in fired
    assert "brightness" in fired or "bucket" in fired or "rgbt" in fired



def test_written_lesson_terms_enter_heuristic_query() -> None:
    proto = _brightness_protocol()
    resolved = compose_scout_queries(
        {},
        proto,
        evidence={
            "last_how_id": "F0",
            "last_review_decision": "DISCARD",
            "lessons": [
                {
                    "statement": "RGB-only collapsed under low illumination APS",
                    "type": "negative_evidence",
                }
            ],
        },
    )
    query = resolved["query"].lower()
    assert "rgb" in query
    assert "illumination" in query or "aps" in query
    assert resolved["source"] == "fallback"


def test_metadata_filter_drops_old_year_and_survey_venue() -> None:
    proto = _brightness_protocol()
    old = _paper(
        paper_id="S2:old",
        title="Brightness-bucket APS on RGBT-Tiny",
        year=2019,
        abstract="Brightness-bucket error analysis APS RGBT-Tiny.",
        venue="Old Workshop",
        citation_count=50,
    )
    survey = _paper(
        paper_id="S2:survey-venue",
        title="A Comprehensive Survey of Deep Learning",
        year=2024,
        abstract="We review CNNs on ImageNet.",
        venue="Fake Survey Journal",
        citation_count=400,
    )
    keep = _paper(
        paper_id="S2:keep-meta",
        title="Brightness-bucket error analysis for RGB-T detection",
        year=2024,
        abstract="APS by brightness buckets on RGBT-Tiny.",
        venue="Fake Detection Conf",
        citation_count=3,
    )
    ranked = filter_papers([old, survey, keep], year_from=2022, min_citations=0)
    kept_ids = [str(row["paper"].paper_id) for row in ranked["kept"]]
    dropped_ids = [str(row["paper"].paper_id) for row in ranked["dropped"]]
    assert "S2:keep-meta" in kept_ids
    assert "S2:old" in dropped_ids
    assert "S2:survey-venue" in dropped_ids
    empty_like = filter_papers([old], year_from=2022)
    assert empty_like["used_fallback"] is True
    assert empty_like["kept"]
    cited = filter_papers(
        [
            keep,
            _paper(
                paper_id="S2:well-cited",
                title="Brightness-bucket error analysis",
                year=2024,
                abstract="APS brightness buckets RGBT-Tiny.",
                venue="Fake Detection Conf",
                citation_count=20,
            ),
        ],
        year_from=2022,
        min_citations=10,
    )
    cited_ids = [str(row["paper"].paper_id) for row in cited["kept"]]
    assert cited["used_fallback"] is False
    assert "S2:well-cited" in cited_ids
    assert "S2:keep-meta" not in cited_ids
    clues = protocol_clues(keep, protocol=proto, research_question=proto["goal"]["notes"])
    assert clues["dataset_hit"] is True
    assert clues["metric_hit"] is True
    assert clues["can_enter_claim_gate"] is False
    miss = protocol_clues(
        _paper(
            paper_id="S2:nlp",
            title="Transformer language models for machine translation",
            abstract="BLEU scores on WMT. No detection.",
        ),
        protocol=proto,
        research_question=proto["goal"]["notes"],
    )
    assert miss["dataset_hit"] is False
    assert miss["protocol_aligned"] is False


def test_scout_one_hop_adds_reference_and_drops_old_year(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_hop",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
        hop=True,
    )
    assert packet["fail_closed"] is False
    ids = [str(row.get("paper_id") or "") for row in packet["papers"]]
    assert "S2:hop-only-005" in ids
    assert "S2:old-brightness-010" not in ids
    assert "S2:dl-survey-999" not in ids
    assert packet["hop_added"] >= 1
    hop_rows = [row for row in packet["papers"] if int(row.get("hop") or 0) == 1]
    assert hop_rows
    assert all(row.get("can_enter_claim_gate") is False for row in packet["papers"])
    ctx = planner_literature_context(packet)
    assert ctx["hop_added"] >= 1
    assert "not Claim evidence" in str((packet["papers"][0] or {}).get("why_relevant") or "").lower() or (
        "not Claim" in str((packet["papers"][0] or {}).get("why_relevant") or "")
    )


def test_ranked_table_best_first_with_urls(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_table",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    table = list(packet.get("ranked_table") or packet["papers"])
    assert table
    assert table[0]["rank"] == 1
    scores = [float(row["score"] or 0) for row in table]
    assert scores == sorted(scores, reverse=True)
    assert all(row.get("url") for row in table)
    assert all(row.get("link_ok") is True for row in table)
    assert all(row.get("can_enter_claim_gate") is False for row in table)
    ctx = planner_literature_context(packet)
    assert ctx["clues"][0].get("url")
    assert ctx["clues"][0].get("rank") == 1
    saved = persist_scout(tmp_path / "how_pending.json", packet)
    ledger = saved["literature_ledger"]
    assert ledger
    assert ledger[0]["url"]
    assert ledger[0]["paper_id"] == table[0]["paper_id"] or float(ledger[0]["best_score"] or 0) >= float(
        table[0]["score"] or 0
    )


def test_fail_closed_can_read_campaign_library_not_invent(tmp_path: Path) -> None:
    proto = _brightness_protocol()
    live_packet = scout_literature_for_how(
        protocol=proto,
        live=False,
        round_id="round_lib_seed",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    persist_scout(tmp_path / "how_pending.json", live_packet)
    from scientist_lab.core.how_pending import load_store

    ledger = load_store(tmp_path / "how_pending.json")["literature_ledger"]
    assert ledger
    closed = scout_literature_for_how(
        protocol=proto,
        live=True,
        round_id="round_lib_closed",
        provenance_dir=tmp_path / "literature_closed",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
        ledger=ledger,
        environ={},
    )
    assert closed["fail_closed"] is False
    assert closed["actual_search"] is False
    assert closed["library_only"] is True
    assert closed["papers"]
    assert closed["papers"][0]["retrieval_source"] == "library"
    assert closed["papers"][0]["url"]
    assert "文献库" in str(closed.get("note") or "")


def test_identity_labels_and_pick_display_text() -> None:
    en = apply_identity_labels(
        {"paper_id": "S2:1", "title": "Hello World", "why_relevant": "uses APS"}
    )
    assert en["title_en"] == "Hello World"
    assert not en.get("title_zh")
    assert pick_display_text(en, "title", "zh") == "Hello World"
    assert pick_display_text(en, "title", "en") == "Hello World"
    zh = apply_identity_labels(
        {"paper_id": "S2:2", "title": "低光融合", "why_relevant": "对准 APS"}
    )
    assert zh["title_zh"] == "低光融合"
    assert pick_display_text(zh, "title", "en") == "低光融合"


def test_translate_fills_zh_without_mutating_url_or_id() -> None:
    provider = ScriptedProvider(
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": "S2:1",
                        "title": "低光 RGB-T 小目标检测",
                        "why_relevant": "对准亮度分桶 APS",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    rows = translate_ranked_papers(
        [
            {
                "paper_id": "S2:1",
                "title": "Low-Light RGB-T Detection",
                "why_relevant": "uses APS",
                "url": "https://example.invalid/x",
                "rank": 1,
                "score": 0.9,
            }
        ],
        locale="zh",
        live=True,
        provider=provider,
    )
    assert rows[0]["paper_id"] == "S2:1"
    assert rows[0]["url"] == "https://example.invalid/x"
    assert rows[0]["rank"] == 1
    assert rows[0]["title"] == "Low-Light RGB-T Detection"
    assert rows[0]["title_zh"] == "低光 RGB-T 小目标检测"
    assert rows[0]["why_relevant_zh"] == "对准亮度分桶 APS"
    assert rows[0]["display_translated"] is True


def test_persist_scout_translates_display_fields(tmp_path: Path) -> None:
    packet = scout_literature_for_how(
        protocol=_brightness_protocol(),
        live=False,
        round_id="round_display",
        provenance_dir=tmp_path / "literature",
        query="brightness-bucket APS RGBT-Tiny",
        research_question="Does brightness-bucket error analysis change APS on RGBT-Tiny?",
    )
    assert packet["papers"]
    first = packet["papers"][0]
    provider = ScriptedProvider(
        json.dumps(
            {
                "papers": [
                    {
                        "paper_id": first["paper_id"],
                        "title": "RGBT-Tiny 亮度分桶误差分析",
                        "why_relevant": "对准 APS 与亮度分桶",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    saved = persist_scout(
        tmp_path / "how_pending.json",
        packet,
        locale="zh",
        live=True,
        provider=provider,
    )
    by_id = {str(row["paper_id"]): row for row in saved["scout"]["papers"]}
    row = by_id[str(first["paper_id"])]
    assert row["url"] == first["url"]
    assert row["title"] == first["title"]
    assert row["title_zh"] == "RGBT-Tiny 亮度分桶误差分析"
    assert row["can_enter_claim_gate"] is False
