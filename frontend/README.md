# Veritas fact-checking frontend

## Requirements

- Node.js 20 or newer
- pnpm 10 or newer (npm also works if preferred)

## Run locally

```powershell
pnpm install
pnpm dev
```

Open `http://localhost:5173/` in the browser. Do not open `index.html` directly.

## Production build

```powershell
pnpm run build
pnpm run preview
```

The frontend currently uses mock analysis data. Set `VITE_API_BASE_URL` in `.env` when the FastAPI backend is available.
