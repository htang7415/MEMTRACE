import json
from pathlib import Path

import scripts.export_release as release_module
from scripts.export_release import build_manifest, build_static_leaderboard, export_release_bundle, remove_appledouble_files


def test_build_manifest_skips_appledouble_files(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("x", encoding="utf-8")
    (tmp_path / "._skip.txt").write_text("y", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert manifest["files"] == [{"path": "ok.txt", "bytes": 1}]


def test_remove_appledouble_files(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("x", encoding="utf-8")
    (tmp_path / "._skip.txt").write_text("y", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "._nested.txt").write_text("z", encoding="utf-8")

    remove_appledouble_files(tmp_path)

    assert (tmp_path / "ok.txt").exists()
    assert not (tmp_path / "._skip.txt").exists()
    assert not (nested / "._nested.txt").exists()


def test_build_static_leaderboard_renders_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        '{"by_configuration":{"mlx-community/Qwen2.5-3B-Instruct-4bit":{"S0":{"CSR":1.0,"OVR":0.0,"SVR":0.0,"SRG":0.0,"PAR":0.0},"S1":{"CSR":1.0,"OVR":0.5,"SVR":0.8,"SRG":0.3,"PAR":1.0},"S2":{"CSR":1.0,"OVR":0.1,"SVR":0.2,"SRG":0.1,"PAR":0.2}}}}',
        encoding="utf-8",
    )
    leaderboard = build_static_leaderboard(metrics_path)
    assert (
        "| mlx-community/Qwen2.5-3B-Instruct-4bit | S2 | 1.000 | 0.100 | 0.200 | 0.100 | 0.200 |"
        in leaderboard
    )


def test_export_release_bundle_includes_github_harness(tmp_path: Path, monkeypatch) -> None:
    release_dir = tmp_path / "release"
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "by_configuration": {
                    "mlx-community/Qwen2.5-3B-Instruct-4bit": {
                        "S0": {"CSR": 1.0, "OVR": 0.0, "SVR": 0.0, "SRG": 0.0, "PAR": 0.0},
                        "S1": {"CSR": 1.0, "OVR": 0.5, "SVR": 0.8, "SRG": 0.3, "PAR": 1.0},
                        "S2": {"CSR": 1.0, "OVR": 0.1, "SVR": 0.2, "SRG": 0.1, "PAR": 0.2},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    files = {
        "passages.jsonl": "[]\n",
        "allowlist.json": "[]\n",
        "episodes.json": "[]\n",
        "tasks.json": "[]\n",
        "labels.json": "[]\n",
        "verification.json": "[]\n",
        "run_summary.json": "[]\n",
        "episode_scores.json": "[]\n",
        "table1.md": "# Table 1\n",
        "supplementary_tables.md": "# Supplementary\n",
        "dataset_card.md": "# Dataset Card\n",
        "LICENSE": "MIT License\n\nCopyright (c) 2026 Named Author\n",
        "project.md": "# project\n",
        "README.md": "# MEMTRACE\n",
        "pyproject.toml": "[project]\nname='memtrace'\n",
        "reproduce.md": "# Reproduce\n",
        "eval_card.md": "# Eval Card\n",
        "third_party_assets.md": "# Third Party\n",
        "croissant_metadata.json": "{}\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "planner.txt").write_text("planner", encoding="utf-8")
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "figure1.svg").write_text("<svg/>", encoding="utf-8")
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    trace_path = traces_dir / "trace.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(
        json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path), "protocol_version": "project-md-v3"}]),
        encoding="utf-8",
    )
    (tmp_path / "episode_scores.json").write_text(
        json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path)}]),
        encoding="utf-8",
    )
    audit_sample_path = tmp_path / "audit_sample.json"
    audit_sample_path.write_text(json.dumps([{"episode_id": "ep1", "trace_path": str(trace_path)}]), encoding="utf-8")
    audit_template_path = tmp_path / "audit_template.jsonl"
    audit_template_path.write_text(json.dumps({"episode_id": "ep1", "trace_path": str(trace_path)}) + "\n", encoding="utf-8")
    memtrace_dir = tmp_path / "memtrace"
    memtrace_dir.mkdir()
    (memtrace_dir / "__init__.py").write_text("", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('x')\n", encoding="utf-8")
    (scripts_dir / "promote_results.py").write_text("print('internal')\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")

    monkeypatch.setattr(release_module, "RELEASE_DIR", release_dir)
    monkeypatch.setattr(release_module, "RELEASE_MANIFEST_PATH", release_dir / "manifest.json")
    monkeypatch.setattr(release_module, "PASSAGES_PATH", tmp_path / "passages.jsonl")
    monkeypatch.setattr(release_module, "ALLOWLIST_PATH", tmp_path / "allowlist.json")
    monkeypatch.setattr(release_module, "EPISODES_PATH", tmp_path / "episodes.json")
    monkeypatch.setattr(release_module, "RETRIEVAL_VERIFICATION_PATH", tmp_path / "verification.json")
    monkeypatch.setattr(release_module, "GOLD_DIR", tmp_path)
    monkeypatch.setattr(release_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(release_module, "EPISODE_SCORES_PATH", tmp_path / "episode_scores.json")
    monkeypatch.setattr(release_module, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(release_module, "TABLE1_MD_PATH", tmp_path / "table1.md")
    monkeypatch.setattr(release_module, "SUPPLEMENTARY_TABLES_MD_PATH", tmp_path / "supplementary_tables.md")
    monkeypatch.setattr(release_module, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(release_module, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(release_module, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(release_module, "ATTRIBUTION_LABELS_PATH", tmp_path / "missing_attribution_labels.json")
    monkeypatch.setattr(release_module, "ATTRIBUTION_REPORT_PATH", tmp_path / "missing_attribution_report.md")
    monkeypatch.setattr(release_module, "AUDIT_SAMPLE_PATH", audit_sample_path)
    monkeypatch.setattr(release_module, "AUDIT_TEMPLATE_PATH", audit_template_path)
    monkeypatch.setattr(release_module, "AUDIT_REPORT_JSON_PATH", tmp_path / "missing_audit_report.json")
    monkeypatch.setattr(release_module, "AUDIT_REPORT_MD_PATH", tmp_path / "missing_audit_report.md")
    monkeypatch.chdir(tmp_path)

    export_release_bundle()

    assert (release_dir / "github_harness" / "memtrace" / "__init__.py").exists()
    assert (release_dir / "github_harness" / "scripts" / "run.py").exists()
    assert not (release_dir / "github_harness" / "scripts" / "promote_results.py").exists()
    assert (release_dir / "github_harness" / "tests" / "test_x.py").exists()
    assert (release_dir / "github_harness" / "pyproject.toml").exists()
    assert "Anonymous Authors" in (release_dir / "LICENSE").read_text(encoding="utf-8")
    assert "Named Author" not in (release_dir / "LICENSE").read_text(encoding="utf-8")
    assert (release_dir / "REPRODUCE.md").exists()
    assert (release_dir / "VALIDATION.md").exists()
    assert (release_dir / "DATASET_CARD.md").exists()
    assert (release_dir / "EVAL_CARD.md").exists()
    assert (release_dir / "THIRD_PARTY_ASSETS.md").exists()
    assert (release_dir / "croissant_metadata.json").exists()
    assert (release_dir / "pyproject.toml").exists()
    assert (release_dir / "configs" / "scoring.yaml").exists()
    assert (release_dir / "data" / "corpus" / "passages.jsonl").exists()
    assert (release_dir / "tables" / "main_metrics.json").exists()
    assert (release_dir / "tables" / "confidence_intervals.json").exists()
    assert (release_dir / "tools" / "README.md").exists()
    assert (release_dir / "scripts" / "validate_artifact.py").exists()
    assert (release_dir / "traces" / "v1_main_324" / "trace.jsonl").exists()
    release_trace = json.loads((release_dir / "traces" / "v1_main_324" / "trace.jsonl").read_text(encoding="utf-8"))
    assert release_trace["schema_version"] == "project-md-v3"
    release_summary = json.loads((release_dir / "results" / "run_summary.json").read_text(encoding="utf-8"))
    assert release_summary[0]["trace_path"] == "traces/v1_main_324/trace.jsonl"
    release_audit_sample = json.loads((release_dir / "audit" / "audit_sample.json").read_text(encoding="utf-8"))
    assert release_audit_sample[0]["trace_path"] == "traces/v1_main_324/trace.jsonl"
    assert not (release_dir / "audit" / "audit_review.md").exists()
