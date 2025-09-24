import logging
import os
import json


_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    structured = os.getenv("LOG_JSON", "0") in ("1", "true", "TRUE", "yes")
    if structured:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # noqa: D401
                payload = {
                    "ts": self.formatTime(record, os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.setLevel(level)
        root.handlers = [handler]
        _CONFIGURED = True
        return
    log_format = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    datefmt = os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")
    logging.basicConfig(level=level, format=log_format, datefmt=datefmt)
    _CONFIGURED = True


