/**
 * pages/Predict.jsx
 *
 * Predict page.
 *
 * Flow:
 *   1. On mount — auto-calls GET /experiments/{id}/features
 *   2. Builds form dynamically from feature_columns response
 *      - kind: 'numeric'      → number input (default: median of sample_values)
 *      - kind: 'categorical'  → select dropdown (options from sample_values)
 *   3. User fills form, clicks Predict
 *   4. POST /experiments/{id}/predict with form values as features dict
 *   5. Shows prediction result:
 *      - Regression:      large numeric value
 *      - Classification:  prediction label + confidence bar
 *
 * Key guarantee:
 *   Column names are NEVER hardcoded.
 *   They always come from the trained model's pipeline artifact.
 *   Column mismatch errors are physically impossible.
 */

import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Zap, AlertTriangle, RefreshCcw,
  TrendingUp, ChevronDown,
} from 'lucide-react'
import { getExperimentFeatures, runPrediction } from '../api/client'
import useStore from '../store/useStore'
import PageGuard from '../components/PageGuard'

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

/** Compute median of an array of values. */
function median(values) {
  if (!values?.length) return 0
  const nums = values
    .map(Number)
    .filter((n) => !isNaN(n))
    .sort((a, b) => a - b)
  if (!nums.length) return 0
  const mid = Math.floor(nums.length / 2)
  return nums.length % 2 !== 0
    ? nums[mid]
    : (nums[mid - 1] + nums[mid]) / 2
}

/** Format a prediction value for display. */
function formatPrediction(value) {
  if (typeof value !== 'number') return String(value)
  if (value > 99999)
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  if (value > 1)
    return value.toFixed(4)
  return value.toFixed(6)
}

/** Build default form values from feature columns. */
function buildDefaults(featureColumns) {
  const defaults = {}
  featureColumns.forEach((f) => {
    if (f.kind === 'numeric') {
      defaults[f.name] = median(f.sample_values)
    } else {
      defaults[f.name] = f.sample_values?.[0] ?? ''
    }
  })
  return defaults
}

// ─────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────

/**
 * Single feature input — numeric or categorical.
 */
function FeatureInput({ feature, value, onChange }) {
  const { name, kind, sample_values } = feature

  if (kind === 'categorical') {
    return (
      <div>
        <label className="label">{name}</label>
        <select
          className="input"
          value={value ?? sample_values?.[0] ?? ''}
          onChange={(e) => onChange(name, e.target.value)}
        >
          {(sample_values || []).map((v) => (
            <option key={String(v)} value={String(v)}>
              {String(v)}
            </option>
          ))}
        </select>
      </div>
    )
  }

  return (
    <div>
      <label className="label">{name}</label>
      <input
        type="number"
        step="any"
        className="input"
        value={value ?? ''}
        onChange={(e) => {
          const parsed = parseFloat(e.target.value)
          onChange(name, isNaN(parsed) ? '' : parsed)
        }}
      />
    </div>
  )
}

/**
 * Prediction result display.
 * Handles both regression (large number) and
 * classification (label + confidence bar).
 */
function PredictionResult({ result, taskType }) {
  const isClassification = taskType === 'classification'
  const hasProb          = result.probability != null

  return (
    <div className="card border-primary-400/25 space-y-5 animate-slide-up">

      {/* Header */}
      <div className="flex items-center gap-2 text-primary-400">
        <TrendingUp size={16} />
        <span
          className="font-semibold"
          style={{ fontFamily: 'Syne, sans-serif' }}
        >
          Prediction Result
        </span>
      </div>

      <div className="divider" />

      {/* Main prediction value */}
      <div className="text-center py-4">
        <div className="text-xs text-text-muted uppercase tracking-widest mb-3">
          {isClassification ? 'Predicted Class' : 'Predicted Value'}
        </div>
        <div
          className="text-4xl font-bold text-text-primary"
          style={{ fontFamily: 'Syne, sans-serif' }}
        >
          {formatPrediction(result.prediction)}
        </div>
      </div>

      {/* Confidence bar (classification only) */}
      {isClassification && hasProb && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted uppercase tracking-wider">
              Confidence
            </span>
            <span
              className="font-semibold text-text-primary mono"
            >
              {(result.probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-bg-border rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-400 rounded-full transition-all duration-700"
              style={{ width: `${(result.probability * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
      )}

      <div className="divider" />

      {/* Footer meta */}
      <div className="flex items-center justify-between text-[11px] text-text-muted mono">
        <span>model · {result.model_name}</span>
        <span>{result.experiment_id}</span>
      </div>

    </div>
  )
}

/**
 * Section header for numeric / categorical groups.
 */
function FeatureGroupHeader({ label, count }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-text-muted uppercase tracking-widest">
        {label}
      </span>
      <span className="badge-neutral">{count}</span>
      <div className="flex-1 h-px bg-bg-border" />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────

export default function Predict() {
  const {
    experimentId, taskType,
    featureColumns, featureTargetColumn,
    lastPrediction,
    setFeatureColumns, setPrediction,
  } = useStore()

  const [formValues, setFormValues] = useState({})

  // ── Fetch feature schema ────────────────────────────────────────
  const {
    data:     featuresData,
    isLoading: featuresLoading,
    isError:   featuresError,
    error:     featuresErr,
    refetch:   retryFeatures,
  } = useQuery({
    queryKey: ['features', experimentId],
    queryFn:  () => getExperimentFeatures(experimentId),
    enabled:  !!experimentId,
    // Don't refetch — feature schema never changes for a given experiment
    staleTime: Infinity,
  })

  // Store features in Zustand + init form defaults
  useEffect(() => {
    if (!featuresData) return
    setFeatureColumns(featuresData)
    setFormValues(buildDefaults(featuresData.feature_columns))
  }, [featuresData])

  // Always read target column from fresh API response — never from stale store
  const displayTarget = featuresData?.target_column || featureTargetColumn || null

  // If features already in store (page revisit), init form from store
  useEffect(() => {
    if (featureColumns?.length && !Object.keys(formValues).length) {
      setFormValues(buildDefaults(featureColumns))
    }
  }, [featureColumns])

  // ── Predict mutation ────────────────────────────────────────────
  const predictMut = useMutation({
    mutationFn: () => runPrediction(experimentId, formValues),
    onSuccess:  (data) => setPrediction(data),
  })

  const handleChange = (name, value) => {
    setFormValues((prev) => ({ ...prev, [name]: value }))
    // Clear previous prediction when user changes any input
    // so stale results are never shown alongside new inputs
    if (lastPrediction) setPrediction(null)
  }

  const handleReset = () => {
    const cols = featuresData?.feature_columns || featureColumns
    if (cols?.length) setFormValues(buildDefaults(cols))
  }

  // Resolve task type — use store value but also detect from model name
  // as a fallback when store is stale from a previous experiment
  const resolvedTaskType = (() => {
    if (taskType && taskType !== 'auto') return taskType
    // Detect from model name in last prediction
    const modelName = lastPrediction?.model_name?.toLowerCase() || ''
    if (modelName.includes('classifier')) return 'classification'
    if (modelName.includes('regressor'))  return 'regression'
    // Detect from prediction value type
    if (lastPrediction) {
      const val = lastPrediction.prediction
      if (lastPrediction.probability != null) return 'classification'
      if (typeof val === 'number' && !Number.isInteger(val) && val > 10) return 'regression'
    }
    return taskType || null
  })()

  // Use features from query response first, fall back to store
  const columns       = featuresData?.feature_columns || featureColumns || []
  const numericCols   = columns.filter((f) => f.kind === 'numeric')
  const categoricalCols = columns.filter((f) => f.kind === 'categorical')

  return (
    <PageGuard require="complete">
      <div className="space-y-8 animate-fade-in">

        {/* Page header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="page-title">Run Prediction</h1>
            <p className="page-subtitle">
              {featureTargetColumn
                ? `Predicting: ${featureTargetColumn}`
                : 'Fill in the feature values and run a prediction'}
            </p>
          </div>
          {columns.length > 0 && (
            <span className="badge-neutral shrink-0">
              {columns.length} features
            </span>
          )}
        </div>

        {/* Loading features */}
        {featuresLoading && (
          <div className="card flex items-center gap-3">
            <div className="w-4 h-4 border-2 border-primary-400 border-t-transparent rounded-full animate-spin shrink-0" />
            <span className="text-sm text-text-secondary">
              Loading feature schema from trained model...
            </span>
          </div>
        )}

        {/* Features error */}
        {featuresError && (
          <div className="card border-accent-red/30 space-y-3">
            <div className="flex items-center gap-2 text-accent-red">
              <AlertTriangle size={15} />
              <span className="text-sm font-medium">
                Could not load feature schema
              </span>
            </div>
            <p className="text-xs text-text-secondary">
              {featuresErr?.message}
            </p>
            <button
              className="btn-ghost"
              onClick={() => retryFeatures()}
            >
              <RefreshCcw size={13} />
              Retry
            </button>
          </div>
        )}

        {/* Dynamic feature form */}
        {columns.length > 0 && (
          <div className="space-y-8">

            {/* Numeric inputs */}
            {numericCols.length > 0 && (
              <div className="space-y-4">
                <FeatureGroupHeader
                  label="Numeric Features"
                  count={numericCols.length}
                />
                <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                  {numericCols.map((f) => (
                    <FeatureInput
                      key={f.name}
                      feature={f}
                      value={formValues[f.name]}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Categorical inputs */}
            {categoricalCols.length > 0 && (
              <div className="space-y-4">
                <FeatureGroupHeader
                  label="Categorical Features"
                  count={categoricalCols.length}
                />
                <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                  {categoricalCols.map((f) => (
                    <FeatureInput
                      key={f.name}
                      feature={f}
                      value={formValues[f.name]}
                      onChange={handleChange}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-3 pt-2">
              <button
                className="btn-primary"
                onClick={() => predictMut.mutate()}
                disabled={predictMut.isPending}
              >
                {predictMut.isPending ? (
                  <>
                    <div className="w-4 h-4 border-2 border-bg-canvas border-t-transparent rounded-full animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Zap size={15} />
                    Predict
                  </>
                )}
              </button>

              <button
                className="btn-ghost"
                onClick={handleReset}
                disabled={predictMut.isPending}
              >
                <RefreshCcw size={13} />
                Reset to Defaults
              </button>
            </div>

            {/* Predict error */}
            {predictMut.isError && (
              <div className="alert-danger animate-slide-up">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <p>{predictMut.error?.message}</p>
              </div>
            )}

          </div>
        )}

        {/* Prediction result */}
        {lastPrediction && (
          <PredictionResult
            result={lastPrediction}
            taskType={resolvedTaskType}
          />
        )}

      </div>
    </PageGuard>
  )
}