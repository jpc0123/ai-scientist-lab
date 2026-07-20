from __future__ import annotations

from pathlib import Path


class LogStore:
    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

    def append(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(text)
            if not text.endswith("\n"):
                file.write("\n")

    def read_chunk(
        self, cursor: str | None = None, *, complete: bool
    ) -> tuple[str, str | None, str]:
        raw = self.log_path.read_text(encoding="utf-8")
        offset = int(cursor) if cursor and cursor.isdigit() else 0
        offset = max(0, min(offset, len(raw)))
        content = raw[offset:]
        next_cursor = str(len(raw))
        return content, next_cursor, content
