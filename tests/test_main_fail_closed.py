"""main.py의 LLM 실패 fail-closed 종료 테스트."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as cli
from src.llm_client import LLMError


class FailingClient:
    """API 연결 실패를 흉내 내는 클라이언트."""

    def generate(self, *args, **kwargs):
        raise LLMError("모의 연결 실패")


def test_llm_failure_exits_fail_closed(tmp_path, monkeypatch, capsys):
    """LLM 실패 시: 원인 한 줄 + 미출력 안내 + 종료 코드 1, out/에 산출물 없음."""
    monkeypatch.setattr(cli, "make_client", lambda mode: FailingClient())
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--profile",
        str(ROOT / "data/profiles/p2_partial_borderline.json"),
        "--mock", "--out", str(tmp_path),
    ])

    rc = cli.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "LLM 호출 실패" in captured.err
    assert "생성이 완료되지 않아 리포트를 출력하지 않습니다" in captured.err
    assert list(tmp_path.iterdir()) == []  # 부분 산출물이 남지 않는다
