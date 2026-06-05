# Day 10 — GenAI & Agentic AI Foundation
# logger.py — Structured logging setup for production GenAI applications

import logging
import json
import sys
import time
from datetime import datetime, timezone
from functools import wraps

# ------------------------------------------
# JSON formatter for structured logs
# ------------------------------------------
class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "function":  record.funcName,
            "line":      record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        return json.dumps(log_obj)


# ------------------------------------------
# Logger factory
# ------------------------------------------
def get_logger(name: str, level: str = "INFO", json_format: bool = True) -> logging.Logger:
    """
    Get a named logger. Call this once per module.

    Args:
        name:        Logger name (usually __name__)
        level:       Log level string: DEBUG, INFO, WARNING, ERROR, CRITICAL
        json_format: True for JSON (production), False for human-readable (dev)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        if json_format:
            handler.setFormatter(JSONFormatter())
        else:
            fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
            handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ------------------------------------------
# Decorator — log function calls + duration
# ------------------------------------------
def log_call(logger: logging.Logger):
    """Decorator to log when a function is called and how long it takes."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"Calling {func.__name__}", extra={"extra": {"function": func.__name__}})
            try:
                result = func(*args, **kwargs)
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                logger.info(f"{func.__name__} completed in {elapsed}ms")
                return result
            except Exception as e:
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                logger.error(f"{func.__name__} failed after {elapsed}ms: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


# ------------------------------------------
# Demo
# ------------------------------------------
if __name__ == "__main__":
    # Human-readable logger (dev mode)
    dev_log = get_logger("genai.dev", level="DEBUG", json_format=False)

    dev_log.debug("Debug message — detailed internal state")
    dev_log.info("App started successfully")
    dev_log.warning("GROQ_API_KEY not set — using mock mode")
    dev_log.error("Failed to connect to vector store")

    print()

    # JSON logger (production mode)
    prod_log = get_logger("genai.prod", level="INFO", json_format=True)

    prod_log.info("RAG pipeline initialised")
    prod_log.warning("Retriever returned 0 documents for query")

    print()

    # Decorator usage demo
    @log_call(dev_log)
    def fake_llm_call(prompt: str) -> str:
        time.sleep(0.05)  # simulate latency
        return "mocked response"

    fake_llm_call("What is Agentic AI?")
