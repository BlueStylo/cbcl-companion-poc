"""리포트의 문서 내 링크가 srcdoc iframe(탐색 콘솔) 안에서도 부모 앱을 다시 열지 않는지.

원인: srcdoc 문서는 부모의 base URL을 물려받아 href="#prep"가 부모 URL(#prep)로 해석된다.
수정: 템플릿의 스크립트가 문서 내 링크 클릭을 가로채 scrollIntoView로 처리한다.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "templates" / "report.html.j2"


def test_template_has_inpage_jump_handler():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'href="#prep"' in html, "질문 보기 링크는 단독 파일용으로 앵커 href를 유지한다"
    assert "data-inpage-jump" in html and "scrollIntoView" in html and "preventDefault" in html


def test_rendered_report_carries_handler(tmp_path):
    subprocess.run([sys.executable, "main.py", "--mock", "--profile",
                    "data/profiles/p2_partial_borderline.json", "--out", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True, timeout=120)
    out = (tmp_path / "p2_partial_borderline.html").read_text(encoding="utf-8")
    assert out.count("data-inpage-jump") == 1
    assert 'id="prep"' in out
