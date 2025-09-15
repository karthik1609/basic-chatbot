import logging
import os


_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")
    logging.basicConfig(level=level, format=log_format, datefmt=datefmt)
    _CONFIGURED = True


