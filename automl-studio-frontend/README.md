# AutoML Studio — Frontend

React enterprise dashboard for AutoML Studio.
Connects to the FastAPI backend to upload datasets, monitor training, view results, and run predictions.

---

## Tech Stack

| Package | Version | Purpose |
|---|---|---|
| react | ^18.2.0 | UI framework |
| react-dom | ^18.2.0 | DOM rendering |
| react-router-dom | ^6.22.0 | Client-side routing |
| @tanstack/react-query | ^5.28.0 | API calls + polling |
| axios | ^1.6.7 | HTTP client |
| zustand | ^4.5.2 | Global state management |
| recharts | ^2.12.2 | Bar charts on Results page |
| lucide-react | ^0.383.0 | Icons |
| tailwindcss | ^3.4.3 | Utility CSS framework |
| vite | ^5.2.8 | Build tool + dev server |

---

## Prerequisites

- Node.js 18+
- npm 9+
- AutoML Studio backend running on `http://localhost:8000`

---

## Setup

```bash
# 1. Navigate to frontend folder
cd automl-studio-frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Available Scripts

```bash
npm run dev       # Start dev server on port 5173 with hot reload
npm run build     # Production build → dist/
npm run preview   # Preview production build locally
```

---

## Project Structure

```
automl-studio-frontend/
├── index.html                      ← Entry HTML + Google Fonts
├── vite.config.js                  ← Vite config (port 5173)
├── tailwind.config.js              ← Design tokens + colors
├── postcss.config.js
├── package.json
└── src/
    ├── main.jsx                    ← React root + React Query provider
    ├── App.jsx                     ← Router with all 4 routes
    ├── index.css                   ← Global styles + component classes
    │
    ├── api/
    │   └── client.js               ← Axios instance + all API functions
    │
    ├── store/
    │   └── useStore.js             ← Zustand global state
    │
    ├── components/
    │   ├── Layout.jsx              ← Sidebar + topbar + app shell
    │   └── PageGuard.jsx           ← Protects pages needing prior steps
    │
    └── pages/
        ├── Upload.jsx              ← CSV upload + configuration
        ├── Train.jsx               ← Start training + live progress
        ├── Results.jsx             ← Leaderboard + charts + metrics
        ├── Predict.jsx             ← Dynamic prediction form
        └── NotFound.jsx            ← 404 page
```

---

## Pages

### Upload & Configure `/`
- Drag and drop CSV upload
- Dataset preview — rows, columns, missing values, column schema
- Target column dropdown (auto pre-selected by API suggestion)
- Task type auto-detected — classification or regression
- Test size slider (5%–50%)
- Optuna trials input (1–50)
- Save Configuration → navigate to Train

### Train Pipeline `/train`
- Training config summary (dataset, target, task type)
- Start Training button — calls `POST /experiments/start`
- Live progress bar (0–100%)
- Dynamic stage messages tied to progress percentage
- Scrollable log panel — auto-scrolls, color-coded by level
- Polls `GET /experiments/{id}/status` every 2 seconds
- Polling stops automatically on complete or failed
- Best model metrics on completion
- New Experiment button to reset and start over

### Results & Leaderboard `/results`
- Best model highlight card with primary metric callout
- Model selection comparison panel — Optuna selected vs test set best
- Full leaderboard table — best model row highlighted in teal
- Bar chart — sorted by metric direction (lower/higher is better)
- Metric direction badges (↓ lower / ↑ higher) for all metrics
- Best model all-metrics cards

### Predict `/predict`
- Auto-loads feature schema from `GET /experiments/{id}/features`
- Form built dynamically — numeric inputs + categorical dropdowns
- Columns always match the trained model exactly
- Reset to Defaults button
- Prediction result — large value display
- Confidence bar for classification models
- Clears result when any input changes

---

## Global State (Zustand)

All important state lives in `src/store/useStore.js`.
Nothing critical is stored in component local state.

| State | Set by | Used by |
|---|---|---|
| `filePath` | Upload response | Train (sent to API) |
| `columnNames` | Upload response | Upload (dropdown) |
| `targetColumn` | User selection | Train, sidebar |
| `experimentId` | Start response | All pages, topbar |
| `trainingStatus` | Polling | Train, sidebar dot |
| `bestMetrics` | Results response | Train, Results |
| `leaderboard` | Results response | Results |
| `featureColumns` | Features response | Predict |
| `lastPrediction` | Predict response | Predict |

---

## API Connection

All API calls are in `src/api/client.js`.

Default base URL: `http://localhost:8000`

To change the backend URL, edit line 9 in `src/api/client.js`:

```js
const BASE_URL = 'http://localhost:8000'
```

For Docker deployment, change this to a relative path `/` and configure Nginx to proxy API requests.

---

## Design System

Fonts loaded from Google Fonts:
- **DM Sans** — UI text, buttons, labels
- **JetBrains Mono** — metrics, IDs, log output, code
- **Syne** — page titles, large numbers, headings

Color tokens defined in `tailwind.config.js`:

```
bg-canvas   #05080f   deepest background
bg-base     #080d17   sidebar, cards
bg-surface  #0c1220   elevated surfaces
bg-border   #1a2540   borders
primary-400 #2dd4bf   teal accent, best model highlights
accent-green #10b981  success states
accent-gold  #f59e0b  warnings, regression badge
accent-red   #ef4444  errors, failed state
accent-blue  #38bdf8  classification badge, info
```

---

## Known Behaviors

**Experiment state is in-memory on the backend.**
If uvicorn is restarted, all experiment state is lost.
The frontend detects this and shows a "Session expired" banner.
Solution: go to Upload and retrain.

**Polling stops automatically.**
React Query polls `/status` every 2 seconds.
It stops when `status === 'complete'` or `status === 'failed'`.
No manual cleanup needed.

**Predict form columns are never hardcoded.**
They are always fetched from `GET /experiments/{id}/features`
which reads directly from the fitted pipeline artifact.
Column mismatch errors are physically impossible from the UI.

**Task type auto-detection.**
On upload, the API suggests a target column and task type.
The frontend re-detects task type live as the user changes the target dropdown.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Blank page on load | Check `npm install` completed without errors |
| API connection error | Make sure backend is running on port 8000 |
| 404 on page refresh | Expected in dev — use `npm run dev`, not `npm run preview` |
| "Session expired" banner | Restart uvicorn was detected — go to Upload and retrain |
| Predict form empty | Training must complete before features can be loaded |
| "Wrong target column" error | Pick a column appropriate for the task type |