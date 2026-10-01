"""Translation is an additive, blinded reading aid, not a new review dataset."""

import copy
import json

import pytest
from bs4 import BeautifulSoup

from llm_committee.pivot.models import canonical, digest
from scripts.human_answer_review import TEMPLATE, attach_translation
from scripts.translate_human_review import TRANSLATION_SCHEMA, assemble, english_payload, parse_translation, segments


@pytest.fixture
def review():
    public = {
        "version": "blind-answer-review-v1",
        "items": [
            {
                "id": "opaque-one",
                "question": "Which policy?",
                "options": ["Support", "Oppose"],
                "answers": {"A": "First answer.", "B": "Second answer."},
            },
            {
                "id": "opaque-two",
                "question": "Which other policy?",
                "options": ["More", "Same", "Less"],
                "answers": {"A": "Another answer.", "B": "Final answer."},
            },
        ],
    }
    public["dataset_id"] = digest(public)[:24]
    return public


def translation_for(public):
    translated = {key: f"译文：{value}" for key, value in segments(public).items()}
    items = assemble(public, translated)
    return {
        "version": "blind-review-translation-zh-v2",
        "dataset_id": public["dataset_id"],
        "language": "zh",
        "source_items_sha256": digest(public["items"]),
        "items": items,
        "translation_id": digest(items),
        "model": "translator-model-must-not-be-in-public-payload",
        "note": "private-provenance-not-for-public-payload",
    }


def test_list_schema_avoids_named_property_limit():
    def count(value):
        if isinstance(value, dict):
            return len(value.get("properties", {})) + sum(count(v) for v in value.values())
        if isinstance(value, list):
            return sum(count(v) for v in value)
        return 0

    assert count(TRANSLATION_SCHEMA) == 3


def test_parser_and_assembly_keep_identity_options_and_sides(review):
    source = segments(review)
    entries = [{"id": key, "text": f"中文 {key}"} for key in reversed(source)]
    result = parse_translation(canonical({"translations": entries}), source)
    assert set(result) == set(source)
    items = assemble(review, result)
    for old, new in zip(review["items"], items, strict=True):
        assert new["id"] == old["id"]
        assert new["question"] == f"中文 {old['id']}_question"
        assert new["options"] == [f"中文 {old['id']}_option_{i}" for i in range(len(old["options"]))]
        assert new["answers"] == {side: f"中文 {old['id']}_answer_{side}" for side in ("A", "B")}


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"translations":[],"translations":[]}',
        '{"translations":[{"id":"one","id":"one","text":"中文"}]}',
        '{"translations":{}}',
        '{"translations":[],"extra":true}',
        '{"translations":[]}',
        '{"translations":[{"id":"unknown","text":"中文"}]}',
        '{"translations":[{"id":"one","text":""}]}',
        '{"translations":[{"id":"one","text":"  "}]}',
        '{"translations":[{"id":"one","text":42}]}',
        '{"translations":[{"id":"one","text":"中文","extra":true}]}',
        '{"translations":[{"id":"one","text":"中文"},{"id":"one","text":"重复"}]}',
    ],
)
def test_translation_parser_rejects_incomplete_ambiguous_or_malformed_output(text):
    with pytest.raises(ValueError):
        parse_translation(text, {"one": "English"})


def test_missing_translation_keeps_english_fallback(review, tmp_path):
    assert attach_translation(review, tmp_path / "absent.json") is review


def test_attachment_is_additive_and_never_embeds_provenance(review, tmp_path):
    original = copy.deepcopy(review)
    path = tmp_path / "translation.json"
    artifact = translation_for(review)
    path.write_text(canonical(artifact))
    public = attach_translation(review, path)
    assert review == original
    assert {key: public[key] for key in review} == original
    assert public["translations"]["zh"] == artifact["items"]
    assert public["translation_ids"] == {"zh": artifact["translation_id"]}
    assert "model" not in canonical(public) and artifact["note"] not in canonical(public)
    html = TEMPLATE.read_text().replace("__BLIND_DATA__", canonical(public).replace("<", "\\u003c"))
    page = tmp_path / "review.html"
    page.write_text(html)
    assert english_payload(page) == review
    soup = BeautifulSoup(html, "html.parser")
    assert soup.select_one("#lang-en") and soup.select_one("#lang-zh")
    assert not soup.select("script[src],link[rel=stylesheet],a[href]")


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(dataset_id="wrong"),
        lambda x: x.update(source_items_sha256="wrong"),
        lambda x: x.update(translation_id="wrong"),
        lambda x: x.update(language="en"),
        lambda x: x["items"].pop(),
        lambda x: x["items"].reverse(),
        lambda x: x["items"][0].update(id="wrong"),
        lambda x: x["items"][0]["options"].pop(),
        lambda x: x["items"][0].update(options="中文"),
        lambda x: x["items"][0].update(answers={"A": "中文", "C": "中文"}),
        lambda x: x["items"][0]["answers"].update(A=""),
        lambda x: x["items"][0].update(question=None),
        lambda x: x["items"][0].update(origin="metadata-leak"),
    ],
)
def test_attachment_rejects_mismatched_or_invalid_translation(review, tmp_path, change):
    artifact = translation_for(review)
    change(artifact)
    if artifact["translation_id"] != "wrong":
        artifact["translation_id"] = digest(artifact["items"])
    path = tmp_path / "translation.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        attach_translation(review, path)


def test_assembly_rejects_missing_segments(review):
    with pytest.raises(ValueError, match="exactly match"):
        assemble(review, {})


def test_recorded_fidelity_corrections_are_additive_validated_and_private(review, tmp_path):
    artifact = translation_for(review)
    path = tmp_path / "translation.zh.json"
    path.write_text(canonical(artifact))
    correction_path = tmp_path / "translation.zh-corrections.json"
    corrections = {
        "base_translation_id": artifact["translation_id"],
        "edits": [
            {
                "item_id": "opaque-one",
                "field": ["answers", "A"],
                "before": "译文：First answer.",
                "after": "第一份回答。",
                "source_excerpt": "First answer.",
                "reason": "private-review-note",
            }
        ],
    }
    correction_path.write_text(canonical(corrections))
    result = attach_translation(review, path)
    assert result["items"] == review["items"]
    assert result["dataset_id"] == review["dataset_id"]
    assert result["translations"]["zh"][0]["answers"]["A"] == "第一份回答。"
    assert result["translations"]["zh"][0]["answers"]["B"] == artifact["items"][0]["answers"]["B"]
    assert result["translation_ids"]["zh"] == digest(result["translations"]["zh"])
    assert result["translation_ids"]["zh"] != artifact["translation_id"]
    assert json.loads(path.read_text()) == artifact
    assert "private-review-note" not in canonical(result)
    assert "base_translation_id" not in canonical(result)
    for field, value in [
        ("before", "not present"),
        ("source_excerpt", "not in source"),
        ("field", ["id"]),
        ("field", ["answers", "C"]),
        ("field", ["options", -1]),
        ("item_id", "unknown"),
    ]:
        invalid = copy.deepcopy(corrections)
        invalid["edits"][0][field] = value
        correction_path.write_text(canonical(invalid))
        with pytest.raises(ValueError):
            attach_translation(review, path)
    corrections["base_translation_id"] = "wrong"
    correction_path.write_text(canonical(corrections))
    with pytest.raises(ValueError, match="different translation"):
        attach_translation(review, path)
