import os
import tempfile

import pytest

# Point the API's stores at a throwaway database BEFORE server.main is imported,
# so tests never read or modify the real data/documents.db.
_TMP = tempfile.mkdtemp(prefix="kgrag-tests-")
os.environ["DOCUMENTS_DB"] = os.path.join(_TMP, "documents.db")
os.environ["UPLOAD_DIR"] = os.path.join(_TMP, "uploads")
os.environ["USAGE_DB"] = os.path.join(_TMP, "usage.db")

from server import auth  # noqa: E402


@pytest.fixture(autouse=True)
def _auth_off_and_clean_limits(monkeypatch):
    """Tests run with auth disabled unless they enable it, whatever the local .env says."""
    monkeypatch.setattr(auth, "settings", auth.Settings({}))
    auth._hits.clear()
    yield
    auth._hits.clear()
