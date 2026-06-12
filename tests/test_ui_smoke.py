"""Headless reproduction of the name-gate → problem flow using nicegui.testing.
Run: .venv/bin/python -m pytest tests/_repro_ui.py -v -s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from nicegui.testing import User

pytest_plugins = ["nicegui.testing.user_plugin"]

@pytest.mark.nicegui_main_file(str(Path(__file__).parent.parent / "src" / "ui" / "app.py"))
@pytest.mark.asyncio
async def test_name_gate_to_problem(user: User) -> None:
    await user.open("/")
    await user.should_see("Math Tutor")
    user.find("Your name").type("repro_kid")
    user.find("Start learning").click()
    # problem generation takes a while with Ollama — poll generously
    await user.should_see("Your answer", retries=600)
