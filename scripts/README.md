# Maintenance scripts

Scripts in this directory are run manually and are not part of normal
application startup.

`migrate_existing_data.py` assigns older unscoped application data to one
selected user. It updates SQLite records and RAG metadata in place. Back up all
relevant databases and `data/` before running it, and verify the target user
and resulting records afterward.

The script requires the repository's Python dependencies and a configured
local environment. See the [root README](../README.md) for setup.
