"""The dashboard separates settings and never invents unrun results."""

import hashlib
import json
from urllib.parse import unquote

import pytest
from bs4 import BeautifulSoup

from scripts.report_experiment_dashboard import DEFAULT_TAB, REPORTS, ROOT, load_report, publish, settings


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    output = tmp_path_factory.mktemp("dashboard") / "index.html"
    sources = [ROOT / "docs" / name for name, _ in REPORTS.values()]
    sources += [path.with_suffix(".json") for path in sources]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    publish(output=output)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    assert before == after
    return output, BeautifulSoup(output.read_text(), "html.parser")


def test_six_settings_with_completed_sources_and_unrun_placeholders(dashboard):
    _, soup = dashboard
    tabs = soup.select('[role="tab"]')
    assert len(tabs) == 6
    assert len(soup.select('[role="tabpanel"]')) == 6
    assert len(soup.select('.tab[data-status^="completed"]')) == len(REPORTS)
    assert len(soup.select('.tab[data-status="not-run"]')) == 6 - len(REPORTS)
    assert {tab["data-setting"] for tab in tabs} == {s["id"] for s in settings()}
    for tab in tabs:
        if tab["data-status"] == "not-run":
            panel = soup.find(id=tab["aria-controls"])
            assert "Not run yet" in panel.get_text()
            assert not panel.select("table, iframe, script")


def test_accessible_tabs_and_default(dashboard):
    _, soup = dashboard
    selected = soup.select('[role="tab"][aria-selected="true"]')
    assert len(selected) == 1 and selected[0]["data-setting"] == DEFAULT_TAB
    assert len(soup.select('[role="tabpanel"]:not([hidden])')) == 1
    for tab in soup.select('[role="tab"]'):
        panel = soup.find(id=tab["aria-controls"])
        assert panel["aria-labelledby"] == tab["id"]
        assert tab["tabindex"] == ("0" if tab["aria-selected"] == "true" else "-1")
    ids = [node["id"] for node in soup.select("[id]")]
    assert len(ids) == len(set(ids))


def test_layer_scopes_and_rosters(dashboard):
    _, soup = dashboard
    for spec in settings():
        panel = soup.find(id="panel-" + spec["id"])
        assert panel["data-layers"] == spec["layers"]
        if spec["members"] == 3:
            assert panel["data-layers"] == "A · B · E"
        for model in spec["models"]:
            assert model in panel.select_one(".models").get_text()
        if spec["members"] == 2:
            assert "AB, CA and BC" in panel.select_one(".routing").get_text()
    assert "D" not in soup.find(id="panel-same-family-2")["data-layers"]
    assert "D" in soup.find(id="panel-mixed-family-2")["data-layers"]
    assert "isolate committee size" in soup.select_one("footer").get_text()


def test_embedded_reports_are_exact_snapshots_with_resolvable_links(dashboard):
    output, soup = dashboard
    for (roster, members), (name, _) in REPORTS.items():
        sid = f"{roster.replace('_', '-')}-{members}"
        payload = json.loads(soup.find(id="snapshot-" + sid).string)
        raw = (ROOT / "docs" / name).read_bytes()
        assert payload["html"].encode("utf-8") == raw
        assert payload["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (output.parent / unquote(payload["href"])).resolve() == (ROOT / "docs" / name)
        source_soup = BeautifulSoup(payload["html"], "html.parser")
        for link in source_soup.select('a[href^="#"]'):
            assert source_soup.find(id=unquote(link["href"][1:]))
    for link in soup.select("a[href]"):
        assert (output.parent / unquote(link["href"])).is_file()


def test_offline_isolation_and_no_source_scripts(dashboard):
    _, soup = dashboard
    assert not soup.select("script[src], link[rel=stylesheet]")
    assert len(soup.select('script[type="application/json"]')) == len(REPORTS)
    for frame in soup.select("iframe"):
        assert "allow-scripts" not in frame["sandbox"]
        assert frame.get("title") and frame.get("data-snapshot")
    js = soup.find_all("script")[-1].string
    assert "fetch(" not in js
    for key in ("ArrowLeft", "ArrowRight", "Home", "End", "hashchange", "ResizeObserver"):
        assert key in js


def test_source_cannot_be_overwritten():
    for name, _ in REPORTS.values():
        with pytest.raises(ValueError, match="overwrite"):
            publish(output=ROOT / "docs" / name)
        with pytest.raises(ValueError, match="overwrite"):
            publish(output=(ROOT / "docs" / name).with_suffix(".json"))


@pytest.mark.parametrize("failure", ["protocol", "partial", "mock", "missing-E", "roster"])
def test_rejects_incorrect_completed_sources(tmp_path, failure):
    spec = next(s for s in settings() if s["id"] == "mixed-family-2")
    name, protocol = REPORTS[("mixed_family", 2)]
    data = {
        "protocol_version": protocol,
        "run_report": {
            "protocol_version": protocol,
            "status": "completed",
            "kind": "paid_run",
            "completed_questions": 20,
        },
        "length_controlled_E": {"report": {"status": "completed"}},
    }
    if failure == "protocol":
        data["protocol_version"] = "archived-protocol"
    elif failure == "partial":
        data["run_report"]["status"] = "running"
    elif failure == "mock":
        data["run_report"]["kind"] = "offline_mock"
    elif failure == "roster":
        data["roster"] = "same_family"
    else:
        data["length_controlled_E"]["report"]["status"] = "running"
    (tmp_path / name).write_text("<!doctype html><html><body>Placeholder</body></html>")
    (tmp_path / name).with_suffix(".json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_report(spec, tmp_path, tmp_path / "index.html")


def test_html_in_snapshot_cannot_escape_data_script(tmp_path):
    marker = '</script><script>alert("not executable")</script>'
    for name, protocol in REPORTS.values():
        (tmp_path / name).write_text(marker)
        (tmp_path / name).with_suffix(".json").write_text(
            json.dumps(
                {
                    "protocol_version": protocol,
                    "run_report": {
                        "protocol_version": protocol,
                        "status": "completed",
                        "kind": "paid_run",
                        "completed_questions": 20,
                    },
                    "length_controlled_E": {"report": {"status": "completed"}},
                }
            )
        )
    output = publish(tmp_path)
    soup = BeautifulSoup(output.read_text(), "html.parser")
    assert len(soup.select("script:not([type])")) == 1
    for snapshot in soup.select('script[type="application/json"]'):
        assert json.loads(snapshot.string)["html"] == marker
