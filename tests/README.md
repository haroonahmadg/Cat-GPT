# Tests

The repository uses pytest. Run the suite from the repository root:

```powershell
python -m pytest
```

`conftest.py` configures shared fixtures and temporary per-test data. Tests
are grouped by capability, including authentication/security, agent tools,
SQLite persistence, RAG and citations, image/media, voice, API hardening, and
the Codex frontend workspace behavior.

Keep tests isolated from real user data and third-party services. Prefer
temporary paths and mocked HTTP/model calls; never require production
credentials for the normal test suite.
