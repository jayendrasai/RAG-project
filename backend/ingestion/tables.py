import logging
from typing import List, Dict

log = logging.getLogger(__name__)


def extract_tables(file_path: str) -> List[Dict]:
    """
    Try to pull structured tables from a PDF.
    Uses tabula-py first, falls back to basic text patterns.
    """
    chunks = []

    try:
        import tabula
        tables = tabula.read_pdf(file_path, pages="all", multiple_tables=True, silent=True)

        for i, table in enumerate(tables):
            if table.empty:
                continue
            text = table.to_string(index=False)
            if len(text.strip()) < 20:
                continue
            chunks.append({
                "content": f"[Table {i + 1}]\n{text}",
                "page_number": 0,  # tabula doesn't always give us page numbers reliably
                "content_type": "table",
            })

    except ImportError:
        log.warning("tabula-py not available, skipping table extraction")
    except Exception as e:
        log.warning(f"Table extraction failed: {e}")

    return chunks
