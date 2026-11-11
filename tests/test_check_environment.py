"""Fast tests for the non-invasive environment diagnostic."""

from src.check_environment import collect_report, library_versions


def test_library_versions_contains_core_packages() -> None:
    versions = library_versions()
    assert {"torch", "ultralytics"}.issubset(versions)


def test_report_has_readiness_information() -> None:
    report = collect_report()
    assert "python" in report
    assert "pytorch" in report
    assert "training_readiness" in report
