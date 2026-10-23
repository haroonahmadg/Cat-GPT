# Utilities

`security.py` contains shared file-path, file-type, file-size, and user-input
validation helpers used by document workflows. Keep validation consistent
with the API's upload limits and the supported document loaders.

Add cross-cutting helpers here only when they are genuinely shared; API
request handling and agent orchestration belong in their respective packages.
See the [project guide](../README.md) for architecture and setup.
