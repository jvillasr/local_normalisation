from pathlib import Path


def test_expected_top_level_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "src",
        "scripts",
        "configs",
        "tests",
        "data",
        "experiments",
        "reports",
        "docs",
    ]
    for name in expected:
        assert (root / name).exists(), f"missing: {name}"
