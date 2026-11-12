import pytest

from server import auth


@pytest.fixture(autouse=True)
def _auth_off_and_clean_limits(monkeypatch):
    """Tests run with auth disabled unless they enable it, whatever the local .env says."""
    monkeypatch.setattr(auth, "settings", auth.Settings({}))
    auth._hits.clear()
    yield
    auth._hits.clear()
