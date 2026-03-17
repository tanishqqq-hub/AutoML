/**
 * store/useStore.js
 *
 * Zustand global store for AutoML Studio.
 *
 * State flows in one direction:
 *   Upload → Configure → Train → Results → Predict
 *
 * Every page reads from this store.
 * No page ever stores important state locally —
 * everything that needs to survive navigation lives here.
 *
 * Key rule: experimentId is NEVER typed by the user.
 * It is set automatically by setExperimentStarted()
 * after POST /experiments/start returns.
 */

import { create } from 'zustand'

const useStore = create((set, get) => ({

  // ─────────────────────────────────────────────────────────────
  // UPLOAD STATE
  // Set by setUploadResult() after POST /upload succeeds.
  // ─────────────────────────────────────────────────────────────

  /** Server-side path returned by /upload. Passed to /experiments/start. */
  filePath: null,

  /** Original filename shown in the UI. */
  filename: null,

  /** Number of rows in the uploaded dataset. */
  rowCount: null,

  /** Number of columns in the uploaded dataset. */
  columnCount: null,

  /** Ordered list of all column names — used to populate target dropdown. */
  columnNames: [],

  /** Map of column name → pandas dtype string — used for task type detection. */
  dtypes: {},

  /** Total missing values across the dataset. */
  missingValues: null,

  /**
   * API's best guess at the target column.
   * Pre-selected in the dropdown — user can override.
   */
  suggestedTarget: null,

  /**
   * API's inferred task type based on suggestedTarget.
   * 'classification' | 'regression'
   */
  suggestedTaskType: null,

  // ─────────────────────────────────────────────────────────────
  // CONFIGURE STATE
  // Set by saveConfig() when the user clicks Save Configuration.
  // ─────────────────────────────────────────────────────────────

  /** Column the model will predict. Set from dropdown selection. */
  targetColumn: null,

  /** Test/train split fraction. From slider (0.05–0.5). Default: 0.2 */
  testSize: 0.2,

  /** Number of Optuna trials per model. From number input. Default: 5 */
  nTrials: 5,

  /** True once the user has clicked Save Configuration. */
  configSaved: false,

  // ─────────────────────────────────────────────────────────────
  // EXPERIMENT STATE
  // Set automatically — user never provides these manually.
  // ─────────────────────────────────────────────────────────────

  /**
   * Unique experiment ID returned by POST /experiments/start.
   * NEVER typed by the user.
   * Format: YYYYMMDD_HHMMSS
   */
  experimentId: null,

  /**
   * Current training status.
   * 'pending' | 'running' | 'complete' | 'failed'
   */
  trainingStatus: null,

  /** Latest progress value (0–100) from /status polling. */
  trainingProgress: 0,

  /** Latest stage label from /status polling. */
  trainingStage: '',

  /** Accumulated log lines from /status polling. */
  trainingLogs: [],

  /** Error message if trainingStatus === 'failed'. */
  trainingError: null,

  // ─────────────────────────────────────────────────────────────
  // RESULTS STATE
  // Set by setResults() after GET /experiments/{id}/results.
  // ─────────────────────────────────────────────────────────────

  /** Name of the winning model. */
  bestModelName: null,

  /** Primary metric score of the best model. */
  bestScore: null,

  /** Full metrics dict for the best model. { [metric]: number } */
  bestMetrics: null,

  /**
   * Leaderboard array — all models ranked by primary metric.
   * [{ model_name: string, metrics: { [metric]: number } }]
   */
  leaderboard: [],

  /** Resolved task type for this experiment. */
  taskType: null,

  /** Pipeline wall-clock duration in seconds. */
  durationSeconds: null,

  // ── Model selection comparison ──────────────────────────────────
  /** Model Optuna selected based on validation score. */
  optunaBestModel: null,

  /** Model that scored best on the held-out test set. */
  testBestModel: null,

  /** Test set score of the test-best model. */
  testBestScore: null,

  /** Metric used for test-set comparison. */
  testBestMetric: null,

  /**
   * True when Optuna's choice differs from the test-set best model.
   * Indicates potential overfitting to the validation fold.
   */
  selectionMismatch: false,

  // ─────────────────────────────────────────────────────────────
  // FEATURES / PREDICT STATE
  // Set by setFeatureColumns() after GET /experiments/{id}/features.
  // ─────────────────────────────────────────────────────────────

  /**
   * Feature columns the model was trained on.
   * [{ name, dtype, kind: 'numeric'|'categorical', sample_values }]
   * Built dynamically from the pipeline artifact — never hardcoded.
   */
  featureColumns: [],

  /** Target column name for this experiment (from /features response). */
  featureTargetColumn: null,

  /** Last prediction result from POST /experiments/{id}/predict. */
  lastPrediction: null,

  // ─────────────────────────────────────────────────────────────
  // ACTIONS
  // ─────────────────────────────────────────────────────────────

  /**
   * Called after POST /upload succeeds.
   * Stores all upload metadata and resets all downstream state
   * so a new upload starts fresh.
   *
   * @param {object} data - UploadResponse from API
   */
  setUploadResult: (data) =>
    set({
      filePath:          data.file_path,
      filename:          data.filename,
      rowCount:          data.rows,
      columnCount:       data.columns,
      columnNames:       data.column_names,
      dtypes:            data.dtypes,
      missingValues:     data.missing_values,
      suggestedTarget:   data.suggested_target,
      suggestedTaskType: data.suggested_task_type,

      // Pre-select suggested target as starting value for configure
      targetColumn: data.suggested_target,

      // Reset all downstream state
      configSaved:       false,
      experimentId:      null,
      trainingStatus:    null,
      trainingProgress:  0,
      trainingStage:     '',
      trainingLogs:      [],
      trainingError:     null,
      bestModelName:     null,
      bestScore:         null,
      bestMetrics:       null,
      leaderboard:       [],
      taskType:          null,
      durationSeconds:   null,
      featureColumns:    [],
      featureTargetColumn: null,
      lastPrediction:    null,
    }),

  /**
   * Called when the user clicks Save Configuration.
   *
   * @param {string} targetColumn
   * @param {number} testSize
   * @param {number} nTrials
   */
  saveConfig: (targetColumn, testSize, nTrials) =>
    set({
      targetColumn,
      testSize,
      nTrials,
      configSaved: true,
    }),

  /**
   * Called immediately after POST /experiments/start returns.
   * Stores the experiment ID — the only time it is ever set.
   *
   * @param {string} experimentId
   */
  setExperimentStarted: (experimentId) =>
    set({
      experimentId,
      trainingStatus:   'pending',
      trainingProgress: 0,
      trainingStage:    'Queued...',
      trainingLogs:     [],
      trainingError:    null,
    }),

  /**
   * Called every 2 seconds by React Query while training is running.
   * Updates progress, stage label, and log buffer.
   *
   * @param {object} data - StatusResponse from API
   */
  updateTrainingStatus: (data) =>
    set({
      trainingStatus:   data.status,
      trainingProgress: data.progress,
      trainingStage:    data.current_stage,
      trainingLogs:     data.logs,
      trainingError:    data.error || null,
    }),

  /**
   * Called after GET /experiments/{id}/results succeeds.
   *
   * @param {object} data - ResultsResponse from API
   */
  setResults: (data) =>
    set({
      bestModelName:    data.best_model_name,
      bestScore:        data.best_score,
      bestMetrics:      data.best_metrics,
      leaderboard:      data.leaderboard,
      taskType:         data.task_type,
      durationSeconds:  data.duration_seconds,
      trainingStatus:   'complete',
      // Model selection comparison
      optunaBestModel:  data.optuna_selected       || data.best_model_name,
      testBestModel:    data.test_best_model_name  || null,
      testBestScore:    data.test_best_score       || null,
      testBestMetric:   data.test_best_metric_name || null,
      selectionMismatch: data.selection_mismatch   || false,
    }),

  /**
   * Called after GET /experiments/{id}/features succeeds.
   *
   * @param {object} data - FeaturesResponse from API
   */
  setFeatureColumns: (data) =>
    set({
      featureColumns:      data.feature_columns,
      featureTargetColumn: data.target_column,
    }),

  /**
   * Called after POST /experiments/{id}/predict succeeds.
   *
   * @param {object} result - PredictionResponse from API
   */
  setPrediction: (result) =>
    set({ lastPrediction: result }),

  /**
   * Hard reset — clears everything.
   * Called when the user wants to start over with a new dataset.
   */
  reset: () =>
    set({
      filePath: null, filename: null, rowCount: null, columnCount: null,
      columnNames: [], dtypes: {}, missingValues: null,
      suggestedTarget: null, suggestedTaskType: null,
      targetColumn: null, testSize: 0.2, nTrials: 5, configSaved: false,
      experimentId: null, trainingStatus: null, trainingProgress: 0,
      trainingStage: '', trainingLogs: [], trainingError: null,
      bestModelName: null, bestScore: null, bestMetrics: null,
      leaderboard: [], taskType: null, durationSeconds: null,
      featureColumns: [], featureTargetColumn: null, lastPrediction: null,
      optunaBestModel: null, testBestModel: null, testBestScore: null, testBestMetric: null, selectionMismatch: false,
    }),
}))

export default useStore