"""탐색 콘솔의 기본값이 코어 클라이언트와 어긋나지 않는지 (소스 수준 점검).

Streamlit 앱은 import 시 화면을 그리므로 소스 텍스트로 확인한다.
"""

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "app" / "explorer.py").read_text(encoding="utf-8")


def test_api_default_model_matches_client_default():
    assert re.search(r'API_DEFAULT_MODEL = "gpt-5\.6-luna"', SRC)


def test_env_file_is_loaded_at_startup_for_prefill():
    """.env 로드가 위젯 생성 전(모듈 상단)에서 한 번 일어나야 Ollama 입력칸이 미리 채워진다."""
    first_load = SRC.find('load_env_file(ROOT / ".env")')
    first_widget = SRC.find("st.text_input(")
    assert 0 < first_load < first_widget
