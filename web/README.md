# Scientist Lab Web Console (v1.7)

Local single-user research workbench UI.

## Dev

```powershell
# terminal 1
cd D:\AI Scientist_tiao\scientist-lab
.\.venv\Scripts\scientist-lab.exe serve --host 127.0.0.1 --port 8787

# terminal 2
cd web
npm run dev
```

Open http://127.0.0.1:5173 — Vite proxies `/api` to the backend.

## Build

```powershell
cd web
npm run build
```

Then `scientist-lab serve` serves `web/dist` at `/`.

Legacy console: `/legacy`

## Stack

React + TypeScript + Vite + React Router + TanStack Query
