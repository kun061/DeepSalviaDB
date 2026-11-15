"""Runtime settings shared by the SalviaChat graph."""
from __future__ import annotations

import os

from dotenv import load_dotenv


def configure_settings() -> None:
    """Load local configuration without overriding exported environment variables."""
    load_dotenv(override=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USER_AGENT", "DeepSalviaDB/1.0")
