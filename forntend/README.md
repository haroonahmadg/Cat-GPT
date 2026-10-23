# Next.js frontend

This is the primary browser UI, built with Next.js App Router and React. The
repository directory is spelled `forntend`; deployment root-directory settings
must use that exact name.

## Development

```powershell
npm ci
Copy-Item ..\cat-gpt-deploy\frontend.env.local.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` in `.env.local` to the FastAPI base URL. Production
build and TypeScript validation:

```powershell
npm run build
```

## Code layout

- `app/` — route pages, root layout, and global styles.
- `components/` — chat, sidebar, scheduled reminders, and Codex workspace UI.
- `lib/api.ts` — typed HTTP and SSE client; keep event payloads aligned with
  the backend.
- `lib/useAuth.ts` — client authentication/session state.
- `lib/useChat.ts` — chat, thread, attachment, streaming, and voice state.
- `lib/types.ts` — shared frontend DTO and UI types.
- `public/` — static assets.

The Codex workspace uses user-keyed browser storage and remains separate from
the standard chat thread list. Avoid putting server-only secrets in frontend
environment variables; `NEXT_PUBLIC_` values are exposed to the browser.

See the [project guide](../README.md) for backend setup and deployment links.
