"""Dual-output logger: writes to both terminal and a log file."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """Log to both terminal (stdout) and a file.

    Usage:
        log = Logger("train.log")
        log.print("Generation 10, best fitness: 0.85")
        log.print(key="best_fitness", value=0.85)           # key=value style
        log.print("=== Epoch 1 ===", key="epoch", value=1)   # mixed
    """

    def __init__(self, path: str | Path, mode: str = "w") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open(mode, encoding="utf-8")

    def print(self, *args, key: str | None = None, value: object = None,
              end: str = "\n", flush: bool = True, timestamp: bool = True) -> None:
        """Print to both terminal and log file.

        Args:
            *args: Regular text to print (same as built-in print).
            key:   If provided, formats as "key=value".
            value: Value paired with key.
            end:   Line terminator (default newline).
            flush: Whether to flush after writing.
            timestamp: Prepend a timestamp to the line (default True).
        """
        parts: list[str] = []
        if timestamp:
            t = datetime.now().strftime("%H:%M:%S")
            parts.append(f"[{t}]")
        if args:
            parts.append(" ".join(str(a) for a in args))
        if key is not None:
            parts.append(f"{key}={value}" if not args else f"| {key}={value}")
        line = " ".join(parts)

        # Write to file
        self._file.write(line + end)
        self._file.flush()

        # Write to stdout
        sys.stdout.write(line + end)
        if flush:
            sys.stdout.flush()

    def separator(self, char: str = "=", width: int = 60, timestamp: bool = False) -> None:
        """Print a separator line."""
        self.print(char * width, timestamp=timestamp)

    def header(self, title: str, char: str = "=", width: int = 60) -> None:
        """Print a centered header."""
        self.separator(char, width)
        padding = max(0, width - len(title) - 2) // 2
        self.print(f"{' ' * padding}{title}{' ' * padding}", timestamp=False)
        self.separator(char, width, timestamp=False)

    def print_table(self, **columns: object) -> None:
        """Print space-separated key=value pairs with timestamp prefix."""
        t = datetime.now().strftime("%H:%M:%S")
        parts = [f"[{t}]"] + [f"{k}={v}" for k, v in columns.items()]
        line = "  ".join(parts)
        # To file
        self._file.write(line + "\n")
        self._file.flush()
        # To stdout
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Global registry ─────────────────────────────────────────────────────

_loggers: dict[str, Logger] = {}


def get_logger(path: str | Path, mode: str = "w") -> Logger:
    """Get or create a Logger by file path.

    Note: ``mode`` only applies on the first call for a given path.
    Subsequent calls with the same path return the existing Logger.
    """
    key = str(Path(path).resolve())
    if key not in _loggers:
        _loggers[key] = Logger(path, mode=mode)
    return _loggers[key]
