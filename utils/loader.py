"""
utils/loader.py

Document Loader for the Python Documentation Assistant.

Responsibilities:
- Recursively load HTML documentation
- Extract clean documentation text
- Attach metadata
- Return LangChain Document objects

This module does NOT:
- Split documents
- Generate embeddings
- Create vector databases
"""

from pathlib import Path
from typing import List
import re

from bs4 import BeautifulSoup
from bs4.element import Tag
from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_ROOT = PROJECT_ROOT / "knowledge_base" / "raw" / "python_docs"

IGNORE_DIRS = {
    "_static",
    "_images",
    "_sources",
    "_downloads",
}

IGNORE_FILES = {
    "searchindex.js",
    "objects.inv",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def should_skip(file_path: Path) -> bool:
    """Return True if a file should not be processed."""

    if any(part in IGNORE_DIRS for part in file_path.parts):
        return True

    if file_path.name in IGNORE_FILES:
        return True

    return file_path.suffix.lower() != ".html"


def normalize_text(text: str) -> str:
    """Normalize whitespace without altering content."""

    text = text.replace("\r", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def clean_html(html: str) -> tuple[str, str]:
    """
    Extract meaningful documentation content from a Sphinx HTML page.

    Returns
    -------
    tuple(title, cleaned_text)
    """

    soup = BeautifulSoup(html, "html.parser")

    # -------------------------------------------------------------
    # Remove unwanted tags
    # -------------------------------------------------------------

    for selector in (
        "nav",
        "header",
        "footer",
        "aside",
        "script",
        "style",
        "form",
    ):
        for tag in soup.select(selector):
            tag.decompose()

    # -------------------------------------------------------------
    # Remove common Sphinx UI elements
    # -------------------------------------------------------------

    REMOVE_CLASSES = [
        ".sphinxsidebar",
        ".sphinxsidebarwrapper",
        ".related",
        ".footer",
        ".breadcrumbs",
        ".breadcrumb",
        ".search",
        ".searchbox",
        ".wy-nav-side",
        ".wy-side-nav-search",
        ".wy-breadcrumbs",
        ".toc",
        ".toctree-wrapper",
        ".headerlink",
    ]

    for selector in REMOVE_CLASSES:
        for tag in soup.select(selector):
            tag.decompose()

    # -------------------------------------------------------------
    # Find the documentation body
    # -------------------------------------------------------------

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="body")
        or soup.find("div", class_="bodywrapper")
        or soup.find("div", class_="document")
        or soup.find("div", role="main")
        or soup.body
    )

    if not isinstance(main, Tag):
        return "", ""

    # -------------------------------------------------------------
    # Extract title
    # -------------------------------------------------------------

    title = ""

    if soup.title is not None:
        title = soup.title.get_text(" ", strip=True)

    # -------------------------------------------------------------
    # Extract text
    # -------------------------------------------------------------

    text = main.get_text(separator="\n", strip=True)

    text = normalize_text(text)

    return title, text


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_documents() -> List[Document]:
    """
    Load all documentation pages.

    Returns
    -------
    List[Document]
    """

    documents: List[Document] = []

    html_files = sorted(
        [
            path
            for path in DOCS_ROOT.rglob("*.html")
            if not should_skip(path)
        ]
    )

    for html_file in html_files:

        try:

            html = html_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            title, text = clean_html(html)

            if not text:
                continue

            relative_path = html_file.relative_to(DOCS_ROOT)

            section = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "root"
            )

            metadata = {
                "source": str(html_file.resolve()),
                "relative_path": str(relative_path),
                "title": title,
                "section": section,
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        except Exception as exc:
            print(f"Failed to load {html_file}")
            print(type(exc).__name__, exc)

    # -------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Document Loader Summary")
    print("=" * 60)
    print(f"HTML Files Loaded : {len(html_files)}")
    print(f"Documents Created : {len(documents)}")

    if documents:

        print("\nExample Metadata")
        print("-" * 60)
        print(documents[0].metadata)

        print("\nPreview")
        print("-" * 60)
        print(documents[0].page_content[:500])

    print("=" * 60 + "\n")

    return documents


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    load_documents()