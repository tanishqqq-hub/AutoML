/**
 * pages/Results.jsx
 *
 * Results & Leaderboard page.
 *
 * Shows:
 *   - Best model highlight card
 *   - Full model leaderboard table (best model row highlighted)
 *   - Bar chart comparing all models by primary metric
 *   - All metrics for the best model as individual cards
 */

import { useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, CartesianGrid,
} from 'recharts'
import {
  Trophy, Clock, Zap, ChevronRight, TrendingUp,
  AlertTriangle, Info, CheckCircle,
} from 'lucide-react'
import useStore from '../store/useStore'
import PageGuard from '../components/PageGuard'

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

function formatMetric(value) {
  if (typeof value !== 'number') return value
  if (value > 99999) return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (value > 999)   return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return value.toFixed(4)
}

function formatAxisTick(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000)     return `${(value / 1_000).toFixed(0)}k`
  return value.toFixed(2)
}

// ─────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────

/**
 * Best model highlight card at the top.
 */
// Priority order for picking the headline metric in BestModelCard.
// Matches the evaluator's primary_metric logic in config.yaml.
const HEADLINE_METRIC_PRIORITY = [
  'roc_auc', 'r2', 'r2_score',          // best single-number scores
  'accuracy', 'f1', 'f1_score',          // classification
  'rmse', 'mae',                         // regression error (shown with ↓)
]

function pickHeadlineMetric(metrics) {
  if (!metrics) return { key: null, value: null }
  for (const pm of HEADLINE_METRIC_PRIORITY) {
    if (pm in metrics) return { key: pm, value: metrics[pm] }
  }
  // Fallback — first key
  const key = Object.keys(metrics)[0]
  return { key, value: metrics[key] }
}

function BestModelCard({ bestModelName, bestMetrics, taskType, durationSeconds, experimentId }) {
  const { key: primaryMetric, value: primaryValue } = pickHeadlineMetric(bestMetrics)

  return (
    <div className="card border-primary-400/25 bg-primary-400/5">
      <div className="flex items-center gap-4">

        {/* Trophy icon */}
        <div className="w-12 h-12 rounded-xl bg-primary-400/15 border border-primary-400/30 flex items-center justify-center shrink-0">
          <Trophy size={20} className="text-primary-400" />
        </div>

        {/* Model info */}
        <div className="flex-1 min-w-0">
          <div className="text-xs text-text-muted uppercase tracking-widest mb-1">
            Best Model
          </div>
          <div
            className="text-xl font-bold text-text-primary"
            style={{ fontFamily: 'Syne, sans-serif' }}
          >
            {bestModelName?.replace(/_/g, ' ') || '—'}
          </div>
          <div className="flex items-center gap-3 mt-1.5 flex-wrap">
            {taskType && (
              <span className={taskType === 'classification' ? 'badge-blue' : 'badge-gold'}>
                {taskType}
              </span>
            )}
            {durationSeconds && (
              <span className="flex items-center gap-1 text-xs text-text-muted">
                <Clock size={11} />
                {durationSeconds.toFixed(1)}s
              </span>
            )}
            <span className="text-xs text-text-muted mono">
              {experimentId}
            </span>
          </div>
        </div>

        {/* Primary metric callout */}
        <div className="text-right shrink-0">
          <div className="text-xs text-text-muted uppercase tracking-widest mb-1">
            {primaryMetric} · test set
          </div>
          <div
            className="text-2xl font-bold text-primary-400"
            style={{ fontFamily: 'Syne, sans-serif' }}
          >
            {formatMetric(primaryValue)}
          </div>
          <div className="text-[10px] text-text-muted mt-1">
            selected by Optuna validation
          </div>
        </div>

      </div>
    </div>
  )
}

/**
 * Leaderboard table.
 * Best model row is highlighted with teal background + trophy icon.
 */
function LeaderboardTable({ leaderboard, bestModelName }) {
  if (!leaderboard?.length) return null

  const metricKeys = Object.keys(leaderboard[0]?.metrics || {})

  return (
    <div className="table-wrap">
      <div className="overflow-x-auto max-w-full">
        <table className="w-full text-sm min-w-[600px]">
          <thead>
            <tr>
              <th className="th w-10 text-center">#</th>
              <th className="th">Model</th>
              {metricKeys.map((m) => (
                <th key={m} className="th text-right">{m}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((row, i) => {
              const isBest = row.model_name === bestModelName
              return (
                <tr
                  key={row.model_name}
                  className={`
                    transition-colors duration-100
                    ${isBest
                      ? 'bg-primary-400/8 border-l-2 border-primary-400'
                      : 'hover:bg-bg-raised'
                    }
                  `}
                >
                  {/* Rank */}
                  <td className="td text-center">
                    {isBest ? (
                      <Trophy size={13} className="text-primary-400 mx-auto" />
                    ) : (
                      <span className="text-text-muted mono text-xs">{i + 1}</span>
                    )}
                  </td>

                  {/* Model name */}
                  <td className="td">
                    <span
                      className={`font-medium ${
                        isBest ? 'text-primary-400' : 'text-text-primary'
                      }`}
                    >
                      {row.model_name.replace(/_/g, ' ')}
                    </span>
                    {i === 0 && !isBest && (
                      <span className="ml-2 badge-blue text-[10px]">top test score</span>
                    )}
                    {isBest && (
                      <span className="ml-2 badge-teal text-[10px]">pipeline selected</span>
                    )}
                  </td>

                  {/* Metrics */}
                  {metricKeys.map((key) => (
                    <td
                      key={key}
                      className={`td text-right mono text-xs ${
                        isBest ? 'text-text-primary font-medium' : 'text-text-secondary'
                      }`}
                    >
                      {formatMetric(row.metrics[key])}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/**
 * Custom tooltip for Recharts bar chart.
 */
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-bg-raised border border-bg-border rounded-lg px-4 py-3 shadow-card-lg">
      <div className="text-xs text-text-muted mb-1.5 capitalize">
        {label?.replace(/_/g, ' ')}
      </div>
      <div
        className="text-lg font-bold text-text-primary"
        style={{ fontFamily: 'Syne, sans-serif' }}
      >
        {formatMetric(payload[0]?.value)}
      </div>
      <div className="text-[11px] text-text-muted mt-0.5">
        {payload[0]?.name}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Model Selection Comparison Panel
// Shows both Optuna-selected and test-set-best models side by side
// so the user can make an informed deployment decision.
// ─────────────────────────────────────────────────────────────────

function ModelSelectionPanel({
  optunaBestModel,
  testBestModel,
  testBestScore,
  testBestMetric,
  bestMetrics,
  selectionMismatch,
}) {
  // Only show if we have comparison data
  if (!testBestModel) return null

  const optunScore = bestMetrics?.[testBestMetric]

  return (
    <div className={`rounded-xl border p-5 space-y-4 ${
      selectionMismatch
        ? 'border-accent-gold/30 bg-accent-gold/5'
        : 'border-accent-green/30 bg-accent-green/5'
    }`}>

      {/* Header */}
      <div className="flex items-center gap-2">
        {selectionMismatch ? (
          <AlertTriangle size={15} className="text-accent-gold shrink-0" />
        ) : (
          <CheckCircle size={15} className="text-accent-green shrink-0" />
        )}
        <span
          className="font-semibold text-sm text-text-primary"
          style={{ fontFamily: 'Syne, sans-serif' }}
        >
          {selectionMismatch
            ? 'Model Selection Discrepancy Detected'
            : 'Model Selection Consistent'}
        </span>
      </div>

      {selectionMismatch && (
        <p className="text-xs text-text-secondary leading-relaxed">
          Optuna selected <span className="text-text-primary font-medium">{optunaBestModel?.replace(/_/g, ' ')}</span> during
          hyperparameter tuning using validation fold scores. However,{' '}
          <span className="text-text-primary font-medium">{testBestModel?.replace(/_/g, ' ')}</span> scored
          higher on the held-out test set. This can happen when a model
          overfits to the validation fold. The currently deployed model
          is <span className="text-primary-400 font-medium">{optunaBestModel?.replace(/_/g, ' ')}</span>.
        </p>
      )}

      {/* Comparison cards */}
      <div className="grid grid-cols-2 gap-4">

        {/* Optuna selected */}
        <div className={`rounded-lg border p-4 space-y-2 ${
          !selectionMismatch
            ? 'border-accent-green/30 bg-accent-green/5'
            : 'border-bg-border bg-bg-surface'
        }`}>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-primary-400/15 border border-primary-400/30 flex items-center justify-center">
              <Trophy size={12} className="text-primary-400" />
            </div>
            <span className="text-xs text-text-muted uppercase tracking-wider">
              Optuna Selected
            </span>
          </div>
          <div className="font-bold text-text-primary" style={{ fontFamily: 'Syne, sans-serif' }}>
            {optunaBestModel?.replace(/_/g, ' ')}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">{testBestMetric} · test set</span>
            <span className="mono text-sm font-semibold text-text-primary">
              {optunScore != null ? optunScore.toFixed(4) : '—'}
            </span>
          </div>
          <div className="badge-teal text-[10px] inline-flex">
            currently deployed
          </div>
        </div>

        {/* Test set best */}
        <div className={`rounded-lg border p-4 space-y-2 ${
          selectionMismatch
            ? 'border-accent-gold/30 bg-accent-gold/5'
            : 'border-bg-border bg-bg-surface'
        }`}>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-accent-blue/15 border border-accent-blue/30 flex items-center justify-center">
              <TrendingUp size={12} className="text-accent-blue" />
            </div>
            <span className="text-xs text-text-muted uppercase tracking-wider">
              Test Set Best
            </span>
          </div>
          <div className="font-bold text-text-primary" style={{ fontFamily: 'Syne, sans-serif' }}>
            {testBestModel?.replace(/_/g, ' ')}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">{testBestMetric} · test set</span>
            <span className="mono text-sm font-semibold text-text-primary">
              {testBestScore != null ? testBestScore.toFixed(4) : '—'}
            </span>
          </div>
          {selectionMismatch ? (
            <div className="badge-blue text-[10px] inline-flex">
              better on unseen data
            </div>
          ) : (
            <div className="badge-green text-[10px] inline-flex">
              same as deployed
            </div>
          )}
        </div>

      </div>

      {/* Advice */}
      {selectionMismatch && (
        <div className="flex items-start gap-2 pt-1">
          <Info size={13} className="text-text-muted shrink-0 mt-0.5" />
          <p className="text-xs text-text-muted leading-relaxed">
            <span className="text-text-secondary font-medium">What to do:</span>{' '}
            Increase Optuna trials (recommended: 20+) for better generalization,
            or retrain with <span className="mono">{testBestModel}</span> as the
            only enabled model to force its selection.
          </p>
        </div>
      )}

    </div>
  )
}

// ── Metric direction map ─────────────────────────────────────────
// Defines whether lower or higher is better for each metric,
// and provides a plain-English description.
const METRIC_DIRECTIONS = {
  // ── Regression error metrics — lower is better ──
  rmse:  { dir: 'lower', label: 'Root Mean Square Error — lower is better'  },
  mae:   { dir: 'lower', label: 'Mean Absolute Error — lower is better'     },
  mse:   { dir: 'lower', label: 'Mean Square Error — lower is better'       },
  mape:  { dir: 'lower', label: 'Mean Absolute % Error — lower is better'   },
  msle:  { dir: 'lower', label: 'Mean Squared Log Error — lower is better'  },
  // ── Regression score metrics — higher is better ──
  r2:    { dir: 'higher', label: 'R² Score — higher is better (max 1.0)'    },
  r2_score: { dir: 'higher', label: 'R² Score — higher is better (max 1.0)' },
  // ── Classification score metrics — higher is better ──
  accuracy:  { dir: 'higher', label: 'Accuracy — higher is better (max 1.0)'   },
  precision: { dir: 'higher', label: 'Precision — higher is better (max 1.0)'  },
  recall:    { dir: 'higher', label: 'Recall — higher is better (max 1.0)'     },
  f1:        { dir: 'higher', label: 'F1 Score — higher is better (max 1.0)'   },
  f1_score:  { dir: 'higher', label: 'F1 Score — higher is better (max 1.0)'   },
  roc_auc:   { dir: 'higher', label: 'ROC AUC — higher is better (max 1.0)'    },
  auc:       { dir: 'higher', label: 'AUC — higher is better (max 1.0)'        },
  log_loss:  { dir: 'lower',  label: 'Log Loss — lower is better'              },
  logloss:   { dir: 'lower',  label: 'Log Loss — lower is better'              },
}

/**
 * Shows a direction badge + description for any metric.
 * Falls back to a generic note if metric is unknown.
 */
function MetricDirectionNote({ metric }) {
  if (!metric) return null

  const key  = metric.toLowerCase().replace(/ /g, '_')
  const info = METRIC_DIRECTIONS[key]

  if (!info) {
    return (
      <span className="text-[11px] text-text-muted italic">
        {metric}
      </span>
    )
  }

  return (
    <div className="flex items-center gap-1.5">
      <span
        className={info.dir === 'lower' ? 'badge-blue' : 'badge-teal'}
      >
        {info.dir === 'lower' ? '↓ lower' : '↑ higher'}
      </span>
      <span className="text-[11px] text-text-muted hidden sm:inline">
        {info.label}
      </span>
    </div>
  )
}

/**
 * Bar chart comparing all models on the primary metric.
 */
function ModelBarChart({ leaderboard, bestModelName }) {
  if (!leaderboard?.length) return null

  // Use same priority as BestModelCard — not just the first key
  const { key: primaryMetric } = pickHeadlineMetric(leaderboard[0]?.metrics || {})
  if (!primaryMetric) return null

  // Determine direction for this metric
  const metricKey  = primaryMetric?.toLowerCase().replace(/ /g, '_')
  const metricInfo = METRIC_DIRECTIONS[metricKey]
  const isLowerBetter = metricInfo?.dir === 'lower'

  // Sort chart data so the BEST model is always on the LEFT
  // and bars flow in a visually intuitive direction:
  //   lower-is-better → ascending  (shortest = best = leftmost)
  //   higher-is-better → descending (tallest  = best = leftmost)
  const data = [...leaderboard]
    .map((row) => ({
      name:   row.model_name.replace(/_/g, ' '),
      value:  row.metrics[primaryMetric] ?? 0,
      isBest: row.model_name === bestModelName,
    }))
    .sort((a, b) =>
      isLowerBetter ? a.value - b.value : b.value - a.value
    )

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-5">
        <TrendingUp size={15} className="text-primary-400" />
        <span className="text-sm font-semibold text-text-primary" style={{ fontFamily: 'Syne, sans-serif' }}>
          Model Comparison
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="badge-neutral">{primaryMetric}</span>
          <MetricDirectionNote metric={primaryMetric} />
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={data}
          barCategoryGap="35%"
          margin={{ top: 4, right: 8, bottom: 4, left: 8 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1a2540"
            vertical={false}
          />
          <XAxis
            dataKey="name"
            tick={{
              fill: '#475569',
              fontSize: 11,
              fontFamily: 'JetBrains Mono, monospace',
            }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{
              fill: '#475569',
              fontSize: 10,
              fontFamily: 'JetBrains Mono, monospace',
            }}
            axisLine={false}
            tickLine={false}
            width={64}
            tickFormatter={formatAxisTick}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={64}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.isBest ? '#2dd4bf' : '#1e3a5f'}
                stroke={entry.isBest ? '#2dd4bf40' : 'transparent'}
                strokeWidth={1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 pt-3 border-t border-bg-border">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-primary-400" />
          <span className="text-xs text-text-muted">Best model</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-[#1e3a5f]" />
          <span className="text-xs text-text-muted">Other models</span>
        </div>
      </div>
    </div>
  )
}

/**
 * Individual metric cards for the best model.
 */
function MetricCards({ bestMetrics }) {
  if (!bestMetrics) return null
  const entries = Object.entries(bestMetrics)

  return (
    <div className={`grid gap-4 ${
      entries.length <= 3 ? 'grid-cols-3' :
      entries.length === 4 ? 'grid-cols-4' :
      'grid-cols-3'
    }`}>
      {entries.map(([key, value]) => (
        <div key={key} className="card text-center">
          <div
            className="text-2xl font-bold text-text-primary"
            style={{ fontFamily: 'Syne, sans-serif' }}
          >
            {formatMetric(value)}
          </div>
          <div className="text-xs text-text-muted mt-1.5 uppercase tracking-widest">
            {key}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────

export default function Results() {
  const navigate = useNavigate()

  const {
    experimentId, leaderboard, bestMetrics,
    bestModelName, taskType, durationSeconds,
    suggestedTaskType,
    optunaBestModel, testBestModel, testBestScore,
    testBestMetric, selectionMismatch,
  } = useStore()

  // Filter out 'auto' — same as Train page
  const resolvedTaskType =
    (taskType && taskType !== 'auto' ? taskType : null) ||
    (suggestedTaskType && suggestedTaskType !== 'auto' ? suggestedTaskType : null) ||
    null

  return (
    <PageGuard require="complete">
      <div className="space-y-8 animate-fade-in">

        {/* Page header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="page-title">Results & Leaderboard</h1>
            <p className="page-subtitle">
              All models trained and evaluated — best model selected automatically
            </p>
          </div>
          <button
            className="btn-primary shrink-0"
            onClick={() => navigate('/predict')}
          >
            <Zap size={14} />
            Run Prediction
            <ChevronRight size={14} />
          </button>
        </div>

        {/* Best model card */}
        <BestModelCard
          bestModelName={bestModelName}
          bestMetrics={bestMetrics}
          taskType={resolvedTaskType}
          durationSeconds={durationSeconds}
          experimentId={experimentId}
        />

        {/* Model selection comparison panel */}
        <ModelSelectionPanel
          optunaBestModel={optunaBestModel}
          testBestModel={testBestModel}
          testBestScore={testBestScore}
          testBestMetric={testBestMetric}
          bestMetrics={bestMetrics}
          selectionMismatch={selectionMismatch}
        />

        {/* Leaderboard table */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <p className="section-title mb-0">Model Leaderboard</p>
            <span className="text-[11px] text-text-muted">
              sorted by test set performance · pipeline selected model marked
            </span>
          </div>
          <LeaderboardTable
            leaderboard={leaderboard}
            bestModelName={bestModelName}
          />
        </div>

        {/* Bar chart */}
        <ModelBarChart
          leaderboard={leaderboard}
          bestModelName={bestModelName}
        />

        {/* All metrics for best model */}
        <div>
          <p className="section-title">
            Best Model Metrics —{' '}
            <span className="text-primary-400 normal-case tracking-normal">
              {bestModelName?.replace(/_/g, ' ')}
            </span>
          </p>
          <MetricCards bestMetrics={bestMetrics} />
        </div>

      </div>
    </PageGuard>
  )
}