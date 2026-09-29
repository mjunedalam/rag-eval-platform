"""Run the API with Uvicorn: ``uv run python scripts/serve_api.py``."""

import uvicorn

from rag_eval_platform.api.main import create_app_from_settings
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(create_app_from_settings(settings), host=settings.api_host,
                port=settings.api_port, log_config=None)  # fmt: skip


if __name__ == "__main__":
    main()
