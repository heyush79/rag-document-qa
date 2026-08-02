from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.core.exceptions import UnsupportedFileError


class FileParser:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def parse(self, filename: str, content: bytes) -> str:
        extension = Path(filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(
                f"Unsupported file type '{extension or 'unknown'}'. Use TXT, MD, or PDF."
            )
        if extension in {".txt", ".md"}:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UnsupportedFileError("Text files must be UTF-8 encoded") from exc
        else:
            from pypdf import PdfReader

            try:
                reader = PdfReader(BytesIO(content))
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                raise UnsupportedFileError("The PDF could not be parsed") from exc

        if not text.strip():
            raise UnsupportedFileError("The uploaded document contains no extractable text")
        return text
