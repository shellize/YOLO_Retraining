import pytest

from scripts.validate_redundancy_protocol import validate_protocol


def _summary(retained: list[float]) -> dict:
    return {
        "thresholds": {
            str(threshold): {
                "retained_fraction": fraction,
                "effective_train_images": int(1000 * fraction),
                "original_train_images": 1000,
                "layout": f"tau-{threshold}.yaml",
            }
            for threshold, fraction in zip([0.97, 0.997, 0.999], retained)
        }
    }


def test_protocol_requires_aggressive_and_conservative_retention() -> None:
    result = validate_protocol(
        _summary([0.43, 0.64, 0.78]),
        [0.97, 0.997, 0.999],
        aggressive_max_retained=0.5,
        conservative_min_retained=0.65,
    )
    assert result["status"] == "valid"


def test_protocol_rejects_a_non_aggressive_first_threshold() -> None:
    with pytest.raises(ValueError, match="aggressive threshold"):
        validate_protocol(
            _summary([0.51, 0.64, 0.78]),
            [0.97, 0.997, 0.999],
            aggressive_max_retained=0.5,
            conservative_min_retained=0.65,
        )
