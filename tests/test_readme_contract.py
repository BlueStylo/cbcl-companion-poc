"""README에 공개한 실행 계약이 코드 정책과 어긋나지 않는지 확인한다."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_describes_validation_and_measured_call_ranges():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "입력은 Pydantic으로 검증하고 출력은 결정론 가드레일로 검증" in readme
    assert "호출 1회, 합계 12B 3~5초, 31B 5~8초" in readme
    assert "입력/출력 스키마 검증" not in readme
    assert "호출 2회 합계 12B 9초" not in readme


def test_readme_limits_detection_rate_claim_to_defined_seeds():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "정의한 시드 47건 검출률 100%" in readme


def test_readme_documents_remote_api_key_guard():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "클라우드 URL에서는 필수" in readme
    assert "미설정 또는 `ollama`이면 시작 전에 실패" in readme
