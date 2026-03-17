/**
 * api/client.js
 *
 * Axios instance + all API functions for AutoML Studio.
 * Every component imports from here — never calls axios directly.
 *
 * Base URL: http://localhost:8000
 * All functions return the response data directly (unwrapped).
 * All errors are normalized to a plain Error with a human-readable message.
 */

import axios from 'axios'

// ─────────────────────────────────────────────────────────────────
// Axios instance
// ─────────────────────────────────────────────────────────────────

const client = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 300_000, // 5 minutes — training jobs can be long
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Response interceptor ──────────────────────────────────────────
// Unwraps .data on success.
// Normalizes error detail into a plain Error message on failure.
client.interceptors.response.use(
  (response) => response.data,

  (error) => {
    // FastAPI returns errors as { detail: string } or { detail: [{msg}] }
    const detail = error.response?.data?.detail

    let message
    if (typeof detail === 'string') {
      message = detail
    } else if (Array.isArray(detail)) {
      // Pydantic validation errors come as an array
      message = detail.map((d) => d.msg || d.message || JSON.stringify(d)).join(' · ')
    } else if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      message = 'Cannot connect to AutoML Studio API. Make sure the backend is running on port 8000.'
    } else if (error.code === 'ECONNABORTED') {
      message = 'Request timed out. The server may be busy.'
    } else {
      message = error.message || 'An unexpected error occurred.'
    }

    return Promise.reject(new Error(message))
  }
)

// ─────────────────────────────────────────────────────────────────
// Health
// ─────────────────────────────────────────────────────────────────

/**
 * Check API liveness.
 * @returns {{ status: string, timestamp: number }}
 */
export const getHealth = () =>
  client.get('/health')

// ─────────────────────────────────────────────────────────────────
// Upload
// ─────────────────────────────────────────────────────────────────

/**
 * Upload a CSV file to the backend.
 *
 * @param {File} file - The CSV File object from the file input / drop zone.
 * @param {function} onUploadProgress - Optional progress callback (0-100).
 * @returns {Promise<UploadResponse>}
 *
 * UploadResponse shape:
 * {
 *   file_path:           string   — server-side save path
 *   filename:            string
 *   rows:                number
 *   columns:             number
 *   column_names:        string[]
 *   missing_values:      number
 *   dtypes:              { [col]: string }
 *   suggested_target:    string   — API best guess for target column
 *   suggested_task_type: string   — 'classification' | 'regression'
 * }
 */
export const uploadDataset = (file, onUploadProgress) => {
  const form = new FormData()
  form.append('file', file)

  return client.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onUploadProgress
      ? (e) => {
          const percent = Math.round((e.loaded * 100) / (e.total || 1))
          onUploadProgress(percent)
        }
      : undefined,
  })
}

// ─────────────────────────────────────────────────────────────────
// Experiments
// ─────────────────────────────────────────────────────────────────

/**
 * Start a new training experiment.
 *
 * @param {{ file_path, target_column, test_size, n_trials }} payload
 * @returns {Promise<{ experiment_id, status, message }>}
 */
export const startExperiment = (payload) =>
  client.post('/experiments/start', payload)

/**
 * Poll the status of a running experiment.
 * React Query calls this every 2 seconds until status is terminal.
 *
 * @param {string} experimentId
 * @returns {Promise<StatusResponse>}
 *
 * StatusResponse shape:
 * {
 *   experiment_id:  string
 *   status:         'pending' | 'running' | 'complete' | 'failed'
 *   progress:       number   (0-100)
 *   current_stage:  string
 *   logs:           string[]
 *   error:          string | null
 * }
 */
export const getExperimentStatus = (experimentId) =>
  client.get(`/experiments/${experimentId}/status`)

/**
 * Fetch full results for a completed experiment.
 * Only call this when status === 'complete'.
 *
 * @param {string} experimentId
 * @returns {Promise<ResultsResponse>}
 *
 * ResultsResponse shape:
 * {
 *   experiment_id:   string
 *   best_model_name: string
 *   best_score:      number
 *   best_metrics:    { [metric]: number }
 *   leaderboard:     [{ model_name, metrics: { [metric]: number } }]
 *   task_type:       string
 *   duration_seconds: number
 * }
 */
export const getExperimentResults = (experimentId) =>
  client.get(`/experiments/${experimentId}/results`)

/**
 * Fetch feature schema for the predict form.
 * Returns exactly which columns the trained model expects.
 *
 * @param {string} experimentId
 * @returns {Promise<FeaturesResponse>}
 *
 * FeaturesResponse shape:
 * {
 *   experiment_id:   string
 *   target_column:   string
 *   total_features:  number
 *   feature_columns: [{
 *     name:          string
 *     dtype:         string
 *     kind:          'numeric' | 'categorical'
 *     sample_values: any[]
 *   }]
 * }
 */
export const getExperimentFeatures = (experimentId) =>
  client.get(`/experiments/${experimentId}/features`)

/**
 * Run a prediction using the best model from an experiment.
 *
 * @param {string} experimentId
 * @param {{ [featureName]: any }} features
 * @returns {Promise<PredictionResponse>}
 *
 * PredictionResponse shape:
 * {
 *   prediction:    any
 *   probability:   number | null
 *   model_name:    string
 *   experiment_id: string
 * }
 */
export const runPrediction = (experimentId, features) =>
  client.post(`/experiments/${experimentId}/predict`, { features })