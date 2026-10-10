"""Tests for the seed, evaluation and load-test scripts."""

import pytest

from scripts.evaluate import (
    build_orchestrator,
    feedback_curve,
    split_dataset,
)
from scripts.load_test import run_in_process
from scripts.seed_data import generate_dataset, main as seed_main


def test_generate_dataset_is_deterministic_and_sized():
    first = generate_dataset(30, 40, seed=5)
    second = generate_dataset(30, 40, seed=5)
    assert first == second
    assert len(first["users"]) == 32  # 30 + 2 cold-start users
    assert len(first["contents"]) == 40
    assert all(1 <= c["difficulty"] <= 5 for c in first["contents"])


def test_split_dataset_holds_out_positives():
    data = generate_dataset(30, 40, seed=5)
    train, relevant = split_dataset(data)
    assert relevant
    assert len(train["interactions"]) < len(data["interactions"])
    seen = {(i["user"], i["content"]) for i in train["interactions"]}
    for user, items in relevant.items():
        assert all((user, item) not in seen for item in items)


def test_feedback_curve_runs():
    data = generate_dataset(60, 60, seed=5)
    train, relevant = split_dataset(data)
    rows, probes = feedback_curve(train, relevant, steps=[0, 2])
    assert probes > 0 and [r["feedback_items"] for r in rows] == [0, 2]
    assert build_orchestrator(train).stats()["users"] == 62


def test_load_test_has_no_errors():
    result = run_in_process(workers=4, requests_per_worker=20)
    assert result["errors"] == 0 and result["total_requests"] == 80
    assert result["latency_ms"]["p95"] < 500


def test_seed_cli(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'seed.db'}"
    assert seed_main(["--db", url, "--reset", "--small"]) == 0
    assert "14 users" in capsys.readouterr().out
    assert seed_main(["--db", url, "--if-empty"]) == 0
    assert "already seeded" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        seed_main(["--bogus"])
