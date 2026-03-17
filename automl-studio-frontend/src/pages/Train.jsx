/**
 * pages/Train.jsx
 */

import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Play, ChevronRight, AlertTriangle, CheckCircle,
  Terminal, RefreshCcw, Clock, Cpu,
} from 'lucide-react'
import {
  startExperiment,
  getExperimentStatus,
  getExperimentResults,
} from '../api/client'
import useStore from '../store/useStore'
import PageGuard from '../components/PageGuard'

// ─────────────────────────────────────────────────────────────────
// Dynamic stage messages tied to progress percentage
// ─────────────────────────────────────────────────────────────────

const PROGRESS_MESSAGES = [
  { at: 0,  message: 'Initialising experiment environment...' },
  { at: 5,  message: 'Reading dataset from disk...'           },
  { at: 10, message: 'Dataset loaded. Running validation...'  },
  { at: 25, message: 'Validation passed. Building preprocessing pipeline...' },
  { at: 40, message: 'Features encoded. Splitting train/test sets...' },
  { at: 50, message: 'Starting Optuna hyperparameter search...' },
  { at: 60, message: 'Training models — this may take a moment...' },
  { at: 75, message: 'Comparing model performance across trials...' },
  { at: 85, message: 'Selecting best model. Running final evaluation...' },
  { at: 92, message: 'Saving artifacts and generating leaderboard...' },
  { at: 97, message: 'Almost done — writing experiment results...' },
]

function getDynamicMessage(progress) {
  // Walk backwards to find the highest threshold we've passed
  const matched = [...PROGRESS_MESSAGES]
    .reverse()
    .find((m) => progress >= m.at)
  return matched?.message || 'Initialising...'
}

// ─────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────

function ConfigSummary({ filename, targetColumn, taskType }) {
  return (
    <div className="grid grid-cols-3 gap-4">

      <div className="card">
        <div className="text-xs text-text-muted uppercase tracking-widest mb-1.5">Dataset</div>
        <div className="text-sm font-semibold text-text-primary truncate" title={filename}>
          {filename || '—'}
        </div>
      </div>

      <div className="card">
        <div className="text-xs text-text-muted uppercase tracking-widest mb-1.5">Target</div>
        <div className="text-sm font-semibold text-text-primary truncate" title={targetColumn}>
          {targetColumn || '—'}
        </div>
      </div>

      <div className="card">
        <div className="text-xs text-text-muted uppercase tracking-widest mb-1.5">Task</div>
        {taskType ? (
          <span className={taskType === 'classification' ? 'badge-blue' : 'badge-gold'}>
            {taskType}
          </span>
        ) : (
          <span className="text-sm text-text-muted">—</span>
        )}
      </div>

    </div>
  )
}

function ProgressBar({ progress, status }) {
  const color =
    status === 'complete' ? 'bg-accent-green' :
    status === 'failed'   ? 'bg-accent-red'   :
    'bg-primary-400'

  const glow =
    status === 'running'
      ? 'shadow-[0_0_12px_rgba(45,212,191,0.5)]'
      : ''

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-xs">
        <span className="text-text-muted">Progress</span>
        <span className="mono text-text-secondary">{progress}%</span>
      </div>
      <div className="h-2 bg-bg-border rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${color} ${glow}`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  )
}

function LogPanel({ logs }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs?.length])

  return (
    <div className="rounded-xl border border-bg-border overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-bg-raised border-b border-bg-border">
        <Terminal size={13} className="text-text-muted" />
        <span className="text-xs text-text-muted">Pipeline logs</span>
        <span className="ml-auto text-[11px] text-text-muted mono">
          {logs?.length || 0} lines
        </span>
      </div>
      <div className="h-56 overflow-y-auto p-4 space-y-1 bg-bg-base">
        {!logs?.length ? (
          <div className="text-xs text-text-muted italic mono">
            Waiting for first log line...
          </div>
        ) : (
          logs.map((line, i) => {
            const isError   = line.includes('ERROR')
            const isWarning = line.includes('WARNING')
            return (
              <div
                key={i}
                className={`text-[11px] leading-relaxed mono
                  ${isError   ? 'text-accent-red'  :
                    isWarning ? 'text-accent-gold'  :
                    'text-text-secondary'}`}
              >
                {line}
              </div>
            )
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function MetricCard({ label, value }) {
  const formatted =
    typeof value !== 'number' ? value :
    value > 9999 ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) :
    value.toFixed(4)

  return (
    <div className="card text-center">
      <div
        className="text-xl font-bold text-text-primary"
        style={{ fontFamily: 'Syne, sans-serif' }}
      >
        {formatted}
      </div>
      <div className="text-xs text-text-muted mt-1 uppercase tracking-widest">
        {label}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────

export default function Train() {
  const navigate = useNavigate()

  const {
    filePath, filename, targetColumn, testSize, nTrials,
    suggestedTaskType, taskType,
    experimentId, trainingStatus, trainingProgress,
    trainingStage, trainingLogs, trainingError,
    bestMetrics, bestModelName, durationSeconds,
    setExperimentStarted, updateTrainingStatus, setResults,
  } = useStore()

  const [pollingEnabled, setPollingEnabled] = useState(false)

  // Resolve task type — prefer stored result, fall back to suggestion
  // Filter out 'auto' — that means the pipeline hasn't resolved it yet.
  // After training completes, taskType is set from results.task_type
  // which is always 'classification' or 'regression'.
  const resolvedTaskType =
    (taskType && taskType !== 'auto' ? taskType : null) ||
    (suggestedTaskType && suggestedTaskType !== 'auto' ? suggestedTaskType : null) ||
    null

  // ── Start experiment ─────────────────────────────────────────
  const startMut = useMutation({
    mutationFn: () =>
      startExperiment({
        file_path:     filePath,
        target_column: targetColumn,
        test_size:     testSize,
        n_trials:      nTrials,
      }),
    onSuccess: (data) => {
      setExperimentStarted(data.experiment_id)
      setPollingEnabled(true)
    },
  })

  // ── Status polling ───────────────────────────────────────────
  const { data: statusData } = useQuery({
    queryKey:        ['status', experimentId],
    queryFn:         () => getExperimentStatus(experimentId),
    enabled:         !!experimentId && pollingEnabled,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      if (s === 'complete' || s === 'failed') return false
      return 2000
    },
  })

  useEffect(() => {
    if (!statusData) return
    updateTrainingStatus(statusData)
    if (statusData.status === 'complete' || statusData.status === 'failed') {
      setPollingEnabled(false)
    }
    if (statusData.status === 'complete') {
      getExperimentResults(experimentId).then((res) => setResults(res))
    }
  }, [statusData])

  useEffect(() => {
    if (
      experimentId &&
      trainingStatus !== 'complete' &&
      trainingStatus !== 'failed'
    ) {
      setPollingEnabled(true)
    }
  }, [experimentId])

  const isRunning  = trainingStatus === 'pending' || trainingStatus === 'running'
  const isComplete = trainingStatus === 'complete'
  const isFailed   = trainingStatus === 'failed'

  // Dynamic message based on current progress
  const dynamicMessage = getDynamicMessage(trainingProgress)

  return (
    <PageGuard require="config">
      <div className="space-y-8 animate-fade-in">

        {/* Header */}
        <div>
          <h1 className="page-title">Train Pipeline</h1>
          <p className="page-subtitle">
            AutoML will train and tune all models, then select the best one
          </p>
        </div>

        {/* Config summary */}
        <ConfigSummary
          filename={filename}
          targetColumn={targetColumn}
          taskType={resolvedTaskType}
        />

        {/* Start button — hidden once experiment is started */}
        {!experimentId && (
          <div className="space-y-3">
            <button
              className="btn-primary"
              onClick={() => startMut.mutate()}
              disabled={startMut.isPending}
            >
              {startMut.isPending ? (
                <>
                  <div className="w-4 h-4 border-2 border-bg-canvas border-t-transparent rounded-full animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play size={15} />
                  Start Training
                </>
              )}
            </button>

            {startMut.isError && (
              <div className="alert-danger animate-slide-up">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <p>{startMut.error?.message}</p>
              </div>
            )}
          </div>
        )}

        {/* Live training progress */}
        {experimentId && isRunning && (
          <div className="card space-y-5 animate-slide-up">

            {/* Dynamic stage message */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2.5">
                <Cpu size={15} className="text-primary-400 shrink-0" />
                <span className="text-sm font-medium text-text-primary">
                  {dynamicMessage}
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className="w-2 h-2 rounded-full bg-primary-400 animate-pulse-dot" />
                <span className="text-xs text-text-muted capitalize">
                  {trainingStatus}
                </span>
              </div>
            </div>

            <ProgressBar progress={trainingProgress} status={trainingStatus} />

            <LogPanel logs={trainingLogs} />

            <div className="text-[11px] text-text-muted mono">
              experiment · {experimentId}
            </div>
          </div>
        )}

        {/* Failed */}
        {isFailed && (
          <div className="card border-accent-red/30 space-y-4 animate-slide-up">
            <div className="flex items-center gap-2.5 text-accent-red">
              <AlertTriangle size={16} />
              <span className="font-semibold" style={{ fontFamily: 'Syne, sans-serif' }}>
                Training Failed
              </span>
            </div>
            <p className="text-sm text-text-secondary">
              {trainingError || 'An unknown error occurred during training.'}
            </p>
            <LogPanel logs={trainingLogs} />
            <button
              className="btn-ghost"
              onClick={() => { useStore.getState().reset(); navigate('/') }}
            >
              <RefreshCcw size={13} />
              Start Over
            </button>
          </div>
        )}

        {/* Complete */}
        {isComplete && bestMetrics && (
          <div className="space-y-6 animate-slide-up">

            {/* Success banner */}
            <div className="card border-accent-green/30 flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-accent-green/15 border border-accent-green/25 flex items-center justify-center shrink-0">
                <CheckCircle size={20} className="text-accent-green" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-text-primary" style={{ fontFamily: 'Syne, sans-serif' }}>
                  Training Complete
                </div>
                <div className="text-xs text-text-muted mt-0.5">
                  Best model:{' '}
                  <span className="text-text-secondary">
                    {bestModelName?.replace(/_/g, ' ')}
                  </span>
                  {durationSeconds && (
                    <span className="inline-flex items-center gap-1 ml-2">
                      <Clock size={11} />
                      {durationSeconds.toFixed(1)}s
                    </span>
                  )}
                </div>
              </div>
              <div className="text-[11px] text-text-muted mono shrink-0">
                {experimentId}
              </div>
            </div>

            {/* Metrics */}
            <div>
              <p className="section-title">Best Model Metrics</p>
              <div className={`grid gap-4 ${
                Object.keys(bestMetrics).length <= 3 ? 'grid-cols-3' : 'grid-cols-4'
              }`}>
                {Object.entries(bestMetrics).map(([key, val]) => (
                  <MetricCard key={key} label={key} value={val} />
                ))}
              </div>
            </div>

            {/* Nav */}
            <div className="flex items-center gap-3 flex-wrap">
              <button className="btn-primary" onClick={() => navigate('/results')}>
                View Leaderboard <ChevronRight size={15} />
              </button>
              <button className="btn-ghost" onClick={() => navigate('/predict')}>
                Go to Predict <ChevronRight size={15} />
              </button>
              <button
                className="btn-ghost ml-auto text-xs"
                onClick={() => { useStore.getState().reset(); navigate('/') }}
              >
                New Experiment
              </button>
            </div>

          </div>
        )}

      </div>
    </PageGuard>
  )
}