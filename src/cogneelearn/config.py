"""Project bootstrap: load .env from the project root before anything else.

Cognee calls ``dotenv.load_dotenv()`` on import, but that searches from the
*current working directory* upward — so running the UI or CLI from another
directory silently drops our config and Cognee falls back to its defaults
(OpenAI). We load the project ``.env`` explicitly so the local-LLM setup is
applied regardless of where the app is launched from.
"""

from __future__ import annotations

from pathlib import Path

import dotenv


def project_root() -> Path:
    """Return the repository root (the folder containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent


def load_env() -> Path | None:
    """Load ``.env`` from the project root. Returns the path or None."""
    env_path = project_root() / ".env"
    if env_path.exists():
        dotenv.load_dotenv(env_path, override=True)
        return env_path
    return None


# Load on import so any subsequent `import cognee` picks up the right config.
ENV_PATH = load_env()
