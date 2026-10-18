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

## Model audit visualizations

The **Model audit** navigation item presents the reproducible classical-ML audit:

- source-held validation and selected thresholds;
- held-out macro-F1 by dataset;
- ROC-AUC and Brier calibration metrics;
- contemporary diagnostic confusion matrices;
- false-positive/false-negative trade-offs;
- Random Forest validation-only tuning results.

The visualizations use a versioned static snapshot in
`src/data/trainingResults.js`. They do not expose local datasets or model binaries,
and they explicitly distinguish content classification from factual verification.

Run the data-contract checks and production build with:

```powershell
pnpm test
pnpm build
```
