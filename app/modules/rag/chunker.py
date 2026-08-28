import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.*)", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    heading: str | None


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, section_body) pairs at each markdown
    heading. Any text before the first heading gets heading=None.
    """
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))

    return sections


def _split_into_paragraphs(section_body: str) -> list[str]:
    """Paragraphs are separated by a blank line - the natural next level
    of structure below a heading."""
    paragraphs = re.split(r"\n\s*\n", section_body)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """The fallback: only used when a single paragraph alone is longer
    than chunk_size. This is the ONLY place a hard character-count split
    happens - everywhere else, splits happen at real structural
    boundaries.
    """
    pieces = []
    start = 0
    while start < len(paragraph):
        end = start + chunk_size
        pieces.append(paragraph[start:end])
        if end >= len(paragraph):
            break
        start = end - chunk_overlap
    return pieces


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Structure-aware chunking: split into headed sections, then
    paragraphs, then greedily pack paragraphs into chunks up to
    chunk_size - only hard-splitting a paragraph that alone exceeds it.
    """
    chunks: list[Chunk] = []

    for heading, body in _split_into_sections(text):
        current = ""

        for paragraph in _split_into_paragraphs(body):
            if len(paragraph) > chunk_size:
                if current:
                    chunks.append(Chunk(text=current, heading=heading))
                    current = ""
                for piece in _split_long_paragraph(paragraph, chunk_size, chunk_overlap):
                    chunks.append(Chunk(text=piece, heading=heading))
                continue

            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) > chunk_size:
                chunks.append(Chunk(text=current, heading=heading))
                current = paragraph
            else:
                current = candidate

        if current:
            chunks.append(Chunk(text=current, heading=heading))

    return chunks
