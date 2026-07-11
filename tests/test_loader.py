import pytest
from pathlib import Path
from langchain_core.documents import Document

# Adjust import mapping based on your directory layout
from utils.loader import (
    should_skip,
    normalize_text,
    clean_html,
    load_documents
)


# =====================================================================
# 1. Unit Tests for Helper Functions
# =====================================================================

@pytest.mark.parametrize(
    "path_str, expected",
    [
        ("tutorial/index.html", False),
        ("_static/custom.css", True),
        ("library/_images/plot.png", True),
        ("searchindex.js", True),
        ("library/objects.inv", True),
        ("tutorial/intro.txt", True),
    ]
)
def test_should_skip(path_str, expected):
    """Verify that file exclusions accurately target internal Sphinx items and non-HTML assets."""
    assert should_skip(Path(path_str)) == expected


def test_normalize_text():
    """Verify raw formatting spaces and excessive consecutive newlines are flattened correctly."""
    dirty_text = "Python   is \t great.\n\n\n\nNext line \r here."
    expected = "Python is great.\n\nNext line here."
    assert normalize_text(dirty_text) == expected


# =====================================================================
# 2. Unit Tests for HTML Content Extraction
# =====================================================================

def test_clean_html_extraction():
    """Verify title matching and document structural tag text extraction."""
    sample_html = """
    <html>
        <head><title>Documentation Index</title></head>
        <body>
            <main>
                <h1>Welcome to Python Docs</h1>
                <p>This is the core body content.</p>
            </main>
        </body>
    </html>
    """
    title, text = clean_html(sample_html)
    assert title == "Documentation Index"
    assert "Welcome to Python Docs" in text
    assert "This is the core body content." in text


def test_clean_html_strips_junk_elements():
    """Confirm UI chrome, structural scripts, sidebars, and navigation headers are stripped."""
    sample_html = """
    <html>
        <head><title>Target Page</title></head>
        <body>
            <header><nav>Link 1 | Link 2</nav></header>
            <div class="sphinxsidebar">Sidebar Content</div>
            <main>
                <p>Retained primary content.</p>
                <a class="headerlink" href="#target">¶</a>
            </main>
            <script>console.log('strip me');</script>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    title, text = clean_html(sample_html)
    
    assert "Retained primary content." in text
    assert "Link 1" not in text
    assert "Sidebar Content" not in text
    assert "¶" not in text
    assert "strip me" not in text
    assert "Copyright 2026" not in text


# =====================================================================
# 3. Execution Pipeline Integration Tests
# =====================================================================

def test_load_documents_functional_pipeline(tmp_path, monkeypatch):
    """
    Simulates a small documentation library layout inside a temporary workspace
    to verify end-to-end extraction logic and metadata payload mappings.
    """
    # Create temporary doc skeleton layout architecture
    docs_dir = tmp_path / "python_docs"
    docs_dir.mkdir(parents=True)
    
    # 1. Valid library module document
    lib_dir = docs_dir / "library"
    lib_dir.mkdir()
    valid_file = lib_dir / "os.html"
    valid_file.write_text(
        "<html><head><title>OS Module</title></head><body><main>OS functions.</main></body></html>", 
        encoding="utf-8"
    )
    
    # 2. Ignored asset layout tracking file
    ignored_dir = docs_dir / "_static"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "style.html"
    ignored_file.write_text("<html><body>Junk</body></html>", encoding="utf-8")
    
    # Monkeypatch the config path pointer inside loader.py to use our ephemeral folder context
    monkeypatch.setattr("utils.loader.DOCS_ROOT", docs_dir)
    
    # Run pipeline process tracking loop
    extracted_docs = load_documents()
    
    # Assertions
    assert len(extracted_docs) == 1
    doc = extracted_docs[0]
    
    assert isinstance(doc, Document)
    assert doc.page_content == "OS functions."
    assert doc.metadata["title"] == "OS Module"
    assert doc.metadata["section"] == "library"
    assert doc.metadata["relative_path"] == str(Path("library/os.html"))
    assert Path(doc.metadata["source"]).is_absolute()