"""Unit tests for the frozen fixed-threshold matching protocol."""

from __future__ import annotations

from src.evaluate import KNOWN_CLASSES, match_fixed_protocol


def _prediction(class_id: int, box: tuple[float, float, float, float], confidence: float = 0.9) -> dict[str, object]:
    return {"class_id": class_id, "xyxy": box, "confidence": confidence}


def _target(class_id: int, box: tuple[float, float, float, float]) -> dict[str, object]:
    return {"class_id": class_id, "xyxy": box}


def test_fixed_matching_counts_correct_class_as_tp() -> None:
    counts = match_fixed_protocol([_prediction(0, (0, 0, 10, 10))], [_target(0, (0, 0, 10, 10))])
    assert counts[KNOWN_CLASSES[0]] == {"tp": 1, "fp": 0, "fn": 0}


def test_different_class_overlap_cannot_steal_same_class_match() -> None:
    counts = match_fixed_protocol(
        [_prediction(0, (0, 0, 10, 10))],
        [_target(1, (0, 0, 10, 10)), _target(0, (0, 0, 8, 8))],
    )
    assert counts[KNOWN_CLASSES[0]] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts[KNOWN_CLASSES[1]] == {"tp": 0, "fp": 0, "fn": 1}


def test_multiple_predictions_can_match_one_gt_only_once() -> None:
    counts = match_fixed_protocol(
        [_prediction(0, (0, 0, 10, 10), 0.9), _prediction(0, (0, 0, 10, 10), 0.8)],
        [_target(0, (0, 0, 10, 10))],
    )
    assert counts[KNOWN_CLASSES[0]] == {"tp": 1, "fp": 1, "fn": 0}


def test_no_predictions_produces_false_negative() -> None:
    counts = match_fixed_protocol([], [_target(0, (0, 0, 10, 10))])
    assert counts[KNOWN_CLASSES[0]] == {"tp": 0, "fp": 0, "fn": 1}


def test_no_ground_truth_produces_false_positive() -> None:
    counts = match_fixed_protocol([_prediction(0, (0, 0, 10, 10))], [])
    assert counts[KNOWN_CLASSES[0]] == {"tp": 0, "fp": 1, "fn": 0}


def test_overlapping_ground_truth_of_different_classes_match_independently() -> None:
    counts = match_fixed_protocol(
        [_prediction(0, (0, 0, 10, 10)), _prediction(1, (0, 0, 10, 10), 0.8)],
        [_target(0, (0, 0, 10, 10)), _target(1, (0, 0, 10, 10))],
    )
    assert counts[KNOWN_CLASSES[0]] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts[KNOWN_CLASSES[1]] == {"tp": 1, "fp": 0, "fn": 0}
