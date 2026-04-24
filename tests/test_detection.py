from document_verifier.detection import detect_tampering, draw_highlights
from document_verifier.sample_data import create_sample


def test_detect_tampering_returns_real_regions():
    runtime = __import__("pathlib").Path(__file__).resolve().parent / "runtime" / "detection"
    runtime.mkdir(parents=True, exist_ok=True)
    sample = create_sample(runtime / "sample.png")
    issues = detect_tampering(sample)
    assert len(issues) >= 2
    assert {issue.issue_type for issue in issues}
    assert all(issue.width > 0 and issue.height > 0 for issue in issues)
    output = draw_highlights(sample, issues, runtime / "highlighted.png")
    assert output.exists()
