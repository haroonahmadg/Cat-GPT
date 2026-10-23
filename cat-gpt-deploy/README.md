# Deployment references

This directory contains deployment examples and the provider-specific guide.
The active root Render Blueprint is `../render.yaml`; the `render.yaml` here
is a copy/reference, not a second service that must be deployed.

- `DEPLOYMENT.md` — deployment walkthrough and free-tier storage notes.
- `.env.example` — backend environment-variable template (also available at
  the repository root).
- `frontend.env.local.example` — local Next.js API URL template. Copy it to
  `forntend/.env.local` for local development.
- `render.yaml` and `runtime.txt` — reference deployment configuration.

Keep credentials out of these templates. Configure real secrets in the local
ignored environment files or hosting platform's secret settings.
