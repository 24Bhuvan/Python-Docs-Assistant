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
import re
from typing import List

from bs4 import BeautifulSoup
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
# Helper Functions
# ---------------------------------------------------------------------

def should_skip(file_path: Path) -> bool:
    """
    Determine whether a file should be skipped.
    """

    if any(part in IGNORE_DIRS for part in file_path.parts):
        return True

    if file_path.name in IGNORE_FILES:
        return True

    if file_path.suffix.lower() != ".html":
        return True

    return False


def clean_html(html: str) -> tuple[str, str]:
    """
    Parse HTML and extract meaningful documentation text.

    Returns
    -------
    title : str
    text : str
    """

    soup = BeautifulSoup(html, "html.parser")

    # -------------------------------------------------------------
    # Remove unwanted elements
    # -------------------------------------------------------------

    for tag in soup.find_all([
        "nav",
        "header",
        "footer",
        "aside",
        "script",
        "style",
        "form",
    ]):
        tag.decompose()

    # Remove common Sphinx/navigation classes
    REMOVE_CLASSES = {
        "sphinxsidebar",
        "sphinxsidebarwrapper",
        "related",
        "footer",
        "breadcrumbs",
        "breadcrumb",
        "search",
        "searchbox",
        "toc",
        "toctree-wrapper",
    }

    for tag in soup.find_all(True):
        classes = tag.get("class", [])

        if any(cls in REMOVE_CLASSES for cls in classes):
            tag.decompose()

    # -------------------------------------------------------------
    # Locate main documentation area
    # -------------------------------------------------------------

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="body")
        or soup.find("div", class_="document")
        or soup.body
    )

    if main is None:
        return "", ""

    title = ""

    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    text = main.get_text(separator="\n")

    # -------------------------------------------------------------
    # Normalize whitespace
    # -------------------------------------------------------------

    text = re.sub(r"\r", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return title, text


# ---------------------------------------------------------------------
# Public Loader
# ---------------------------------------------------------------------

def load_documents() -> List[Document]:
    """
    Load the Python HTML documentation.

    Returns
    -------
    List[Document]
    """

    documents: List[Document] = []

    html_files = []

    for file_path in DOCS_ROOT.rglob("*.html"):

        if should_skip(file_path):
            continue

        html_files.append(file_path)

    html_files.sort()

    for html_file in html_files:

        try:

            html = html_file.read_text(
                encoding="utf-8",
                errors="ignore"
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

            print(f"Failed to load {html_file}: {exc}")

    # -----------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------

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
# Standalone Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    load_documents()