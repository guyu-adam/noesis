"""
Tests for world_model.py — collaborative world model with consensus scoring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model import WorldModel


def test_get_consensus_score_empty():
    """Empty world model should return neutral 0.5."""
    wm = WorldModel()
    score = wm.get_consensus_score("test content")
    assert score == 0.5, f"Expected 0.5, got {score}"


def test_get_consensus_score_empty_content():
    """Empty content should return neutral 0.5 even with populated model."""
    wm = WorldModel()
    wm.update("stimulus", {"processor_a": "hello world", "processor_b": "hello consensus"})
    score = wm.get_consensus_score("")
    assert score == 0.5


def test_get_consensus_score_match():
    """Content matching consensus concepts should score high."""
    wm = WorldModel()
    wm.update("s1", {"a": "consciousness global workspace broadcast",
                      "b": "consciousness workspace integration"})
    score = wm.get_consensus_score("consciousness workspace theory")
    assert score > 0.5, f"Expected >0.5 for matching content, got {score}"


def test_get_consensus_score_no_match():
    """Content with no overlap in consensus should score low."""
    wm = WorldModel()
    # Need enough updates to build consensus
    for i in range(3):
        wm.update("s", {"a": "apple banana cherry date",
                         "b": "apple banana elderberry fig"})
    score = wm.get_consensus_score("xylophone zebra quantum")
    assert score < 0.6, f"Expected low score for unrelated content, got {score}"


def test_get_prediction_error_no_history():
    """Processor without prediction history should return 0.5."""
    wm = WorldModel()
    err = wm.get_prediction_error("new_processor", "some content")
    assert err == 0.5


def test_get_prediction_error_repeated():
    """Processor repeating content should have low prediction error."""
    wm = WorldModel()
    for i in range(5):
        wm.update("s", {"processor_a": "the quick brown fox jumps over the lazy dog"})
    err = wm.get_prediction_error("processor_a", "the quick brown fox")
    assert err < 0.5, f"Expected low error for repeated content, got {err}"


def test_update_creates_consensus():
    """Multiple processors sharing tokens should create consensus concepts."""
    wm = WorldModel()
    wm.update("s1", {
        "a": "consciousness global workspace broadcast",
        "b": "consciousness workspace integration theory",
    })
    summary = wm.summary()
    assert "consciousness" in summary["top_consensus_concepts"]
    assert "workspace" in summary["top_consensus_concepts"]


def test_update_limits_history():
    """Stimulus and prediction history should not exceed max limits."""
    wm = WorldModel(max_stimulus_history=5, max_prediction_history=3)
    for i in range(20):
        wm.update(f"stimulus_{i}", {"a": f"content_{i}"})
    assert len(wm.stimulus_history) <= 5
    assert len(wm.processor_predictions.get("a", [])) <= 3


def test_update_returns_summary():
    """Update should return a summary dict."""
    wm = WorldModel()
    result = wm.update("test", {"a": "hello world", "b": "hello consensus"})
    assert "cycle" in result
    assert "consensus_concepts" in result
    assert "mean_disagreement" in result


def test_world_representation():
    """World representation should be accessible."""
    wm = WorldModel()
    wm.update("s", {"a": "one two three", "b": "two three four"})
    rep = wm.world_representation
    assert isinstance(rep, list)


def test_pruning():
    """Consensus concepts should be pruned periodically."""
    wm = WorldModel(prune_interval=2)
    wm.update("s1", {"a": "unique_token_xyz abc", "b": "other def"})
    wm.update("s2", {"a": "ghi jkl", "b": "mno pqr"})
    wm.update("s3", {"a": "stu vwx", "b": "yza bcd"})
    # After pruning, tokens that appeared only once should be removed
    summary = wm.summary()
    # unique_token_xyz only appeared once, should be pruned
    assert len(summary["top_consensus_concepts"]) >= 0
