/**
 * pages/Upload.jsx
 *
 * Upload & Configure page.
 *
 * Flow:
 *   1. User drags/drops or browses a CSV file
 *   2. File uploaded to POST /upload
 *   3. Response populates dataset stats + column schema
 *   4. User picks target column from dropdown
 *      (pre-selected with API suggested_target)
 *   5. Task type auto-detected live as user changes target
 *   6. User sets test size (slider) + n_trials (number input)
 *   7. Save Configuration → stored in Zustand
 *   8. Navigate to Train page
 */

import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import {
  Upload as UploadIcon,
  CheckCircle,
  AlertTriangle,
  ChevronRight,
  Info,
  RotateCcw,
} from 'lucide-react'
import { uploadDataset } from '../api/client'
import useStore from '../store/useStore'

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

function inferTaskType(dtypes, colName, suggestedTarget, suggestedTaskType) {
  if (!colName || !dtypes) return null
  if (colName === suggestedTarget && suggestedTaskType) return suggestedTaskType
  const dtype = dtypes[colName]
  if (!dtype) return null
  if (dtype === 'object' || dtype === 'bool') return 'classification'
  return 'regression'
}

// ─────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────

function DatasetStats({ rowCount, columnCount, missingValues }) {
  const stats = [
    { label: 'Rows',           value: rowCount?.toLocaleString() },
    { label: 'Columns',        value: columnCount },
    { label: 'Missing Values', value: missingValues?.toLocaleString() },
  ]
  return (
    <div className="grid grid-cols-3 gap-4">
      {stats.map(({ label, value }) => (
        <div key={label} className="card text-center">
          <div className="text-2xl font-bold text-text-primary" style={{ fontFamily: 'Syne, sans-serif' }}>
            {value ?? '—'}
          </div>
          <div className="text-xs text-text-muted mt-1 uppercase tracking-widest">{label}</div>
        </div>
      ))}
    </div>
  )
}

function ColumnSchemaTable({ columnNames, dtypes }) {
  if (!columnNames?.length) return null

  const badgeClass = (dtype) => {
    if (dtype === 'object')                                     return 'badge-blue'
    if (dtype === 'bool')                                       return 'badge-gold'
    if (dtype?.startsWith('int') || dtype?.startsWith('float')) return 'badge-teal'
    return 'badge-neutral'
  }

  return (
    <div className="table-wrap">
      <div className="overflow-y-auto max-h-56">
        <table className="w-full text-sm">
          <thead className="sticky top-0">
            <tr>
              <th className="th w-10">#</th>
              <th className="th">Column</th>
              <th className="th">Dtype</th>
            </tr>
          </thead>
          <tbody>
            {columnNames.map((col, i) => (
              <tr key={col} className="hover:bg-bg-raised transition-colors duration-100">
                <td className="td text-text-muted text-xs mono">{i}</td>
                <td className="td font-medium text-text-primary">{col}</td>
                <td className="td">
                  <span className={badgeClass(dtypes?.[col])}>
                    {dtypes?.[col] ?? '—'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TaskTypeBadge({ taskType }) {
  if (!taskType) return null
  return (
    <div className="flex items-center gap-2 mt-2.5">
      <Info size={13} className="text-text-muted shrink-0" />
      <span className="text-xs text-text-muted">Detected task type:</span>
      <span className={taskType === 'classification' ? 'badge-blue' : 'badge-gold'}>
        {taskType}
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────

export default function Upload() {
  const navigate = useNavigate()

  const {
    setUploadResult, saveConfig, reset,
    filePath, filename, rowCount, columnCount, missingValues,
    columnNames, dtypes, suggestedTarget, suggestedTaskType,
    targetColumn, testSize, nTrials, configSaved,
  } = useStore()

  const [dragOver,      setDragOver]      = useState(false)
  const [uploadPercent, setUploadPercent] = useState(0)
  const [localTarget,   setLocalTarget]   = useState(targetColumn || '')
  const [localTestSize, setLocalTestSize] = useState(testSize)
  const [localNTrials,  setLocalNTrials]  = useState(nTrials)
  const [savedBanner,   setSavedBanner]   = useState(configSaved)
  const [notCsvError,   setNotCsvError]   = useState(false)

  const fileInputRef = useRef(null)

  const hasUpload = !!filePath
  const taskType  = inferTaskType(dtypes, localTarget, suggestedTarget, suggestedTaskType)

  // ── Upload mutation ──────────────────────────────────────────
  const uploadMut = useMutation({
    mutationFn: (file) => uploadDataset(file, (pct) => setUploadPercent(pct)),
    onSuccess: (data) => {
      setUploadResult(data)
      setLocalTarget(data.suggested_target || data.column_names[0] || '')
      setSavedBanner(false)
      setUploadPercent(0)
    },
    onError: () => setUploadPercent(0),
  })

  // ── File handling ────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return
    setNotCsvError(false)
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setNotCsvError(true)
      return
    }
    setSavedBanner(false)
    uploadMut.reset()
    uploadMut.mutate(file)
  }, [uploadMut])

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const onFileChange = (e) => {
    handleFile(e.target.files[0])
    e.target.value = ''
  }

  // ── Save config ──────────────────────────────────────────────
  const handleSaveConfig = () => {
    const trials = Math.min(50, Math.max(1, parseInt(localNTrials) || 5))
    saveConfig(localTarget, localTestSize, trials)
    setSavedBanner(true)
  }

  // ── Handle reset ────────────────────────────────────────────
  const handleReset = () => {
    reset()
    uploadMut.reset()
    setSavedBanner(false)
    setNotCsvError(false)
    setLocalTarget('')
    setLocalTestSize(0.2)
    setLocalNTrials(5)
    setUploadPercent(0)
  }

  return (
    <div className="space-y-8 animate-fade-in">

      {/* Page header */}
      <div>
        <h1 className="page-title">Upload & Configure</h1>
        <p className="page-subtitle">
          Upload a CSV dataset and configure your training settings
        </p>
      </div>

      {/* Drop zone */}
      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={onFileChange}
        />

        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`
            relative border-2 border-dashed rounded-xl p-12
            flex flex-col items-center justify-center gap-4
            cursor-pointer transition-all duration-200 select-none
            ${dragOver
              ? 'border-primary-400 bg-primary-400/5 scale-[1.01]'
              : hasUpload
              ? 'border-accent-green/50 bg-accent-green/5'
              : 'border-bg-border hover:border-bg-hover hover:bg-bg-surface/40'
            }
          `}
        >
          {/* Uploading */}
          {uploadMut.isPending && (
            <div className="flex flex-col items-center gap-4">
              <div className="w-10 h-10 border-2 border-primary-400 border-t-transparent rounded-full animate-spin" />
              <div className="text-center space-y-2">
                <p className="text-sm text-text-secondary font-medium">Uploading and validating...</p>
                <div className="w-48 h-1.5 bg-bg-border rounded-full overflow-hidden mx-auto">
                  <div
                    className="h-full bg-primary-400 rounded-full transition-all duration-300"
                    style={{ width: `${uploadPercent}%` }}
                  />
                </div>
                <p className="text-xs text-text-muted mono">{uploadPercent}%</p>
              </div>
            </div>
          )}

          {/* Success */}
          {!uploadMut.isPending && hasUpload && (
            <div className="flex flex-col items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-accent-green/15 border border-accent-green/30 flex items-center justify-center">
                <CheckCircle size={22} className="text-accent-green" />
              </div>
              <div className="text-center space-y-0.5">
                <p className="text-sm font-semibold text-text-primary">{filename}</p>
                <p className="text-xs text-text-muted">
                  {rowCount?.toLocaleString()} rows · {columnCount} columns · Click to replace
                </p>
              </div>
            </div>
          )}

          {/* Idle */}
          {!uploadMut.isPending && !hasUpload && (
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-bg-surface border border-bg-border flex items-center justify-center">
                <UploadIcon size={22} className="text-text-muted" />
              </div>
              <div className="text-center space-y-1.5">
                <p className="text-sm font-medium text-text-secondary">
                  Drag and drop your CSV file here
                </p>
                <p className="text-xs text-text-muted">
                  or click to browse &nbsp;·&nbsp; CSV only &nbsp;·&nbsp; max 50 MB
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Non-CSV error */}
        {notCsvError && (
          <div className="alert-danger mt-3 animate-slide-up">
            <AlertTriangle size={15} className="shrink-0 mt-0.5" />
            <p>Only CSV files are accepted. Please upload a <span className="mono">.csv</span> file.</p>
          </div>
        )}

        {/* Upload API error */}
        {uploadMut.isError && (
          <div className="alert-danger mt-3 animate-slide-up">
            <AlertTriangle size={15} className="shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-xs uppercase tracking-wider mb-0.5">Upload failed</p>
              <p>{uploadMut.error?.message}</p>
            </div>
          </div>
        )}
      </div>

      {/* Dataset info + config (shown after upload) */}
      {hasUpload && (
        <div className="space-y-6 animate-slide-up">

          {/* Stats row */}
          <DatasetStats
            rowCount={rowCount}
            columnCount={columnCount}
            missingValues={missingValues}
          />

          {/* Column schema */}
          <div>
            <p className="section-title">Column Schema</p>
            <ColumnSchemaTable columnNames={columnNames} dtypes={dtypes} />
          </div>

          {/* Configuration card */}
          <div className="card space-y-6">

            {/* Card header */}
            <div className="flex items-center justify-between">
              <p className="section-title mb-0">Training Configuration</p>
              {savedBanner && (
                <div className="flex items-center gap-1.5 text-xs text-accent-green">
                  <CheckCircle size={13} />
                  <span>Configuration saved</span>
                </div>
              )}
            </div>

            <div className="divider" />

            {/* Target column */}
            <div>
              <label className="label">Target Column</label>
              <select
                className="input"
                value={localTarget}
                onChange={(e) => { setLocalTarget(e.target.value); setSavedBanner(false) }}
              >
                {columnNames.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <TaskTypeBadge taskType={taskType} />
            </div>

            {/* Test size */}
            <div>
              <label className="label">
                Test Size —{' '}
                <span className="text-primary-400 normal-case tracking-normal mono">
                  {Math.round(localTestSize * 100)}%
                </span>
              </label>
              <input
                type="range"
                min="0.05" max="0.5" step="0.01"
                value={localTestSize}
                onChange={(e) => { setLocalTestSize(parseFloat(e.target.value)); setSavedBanner(false) }}
              />
              <div className="flex justify-between text-[11px] text-text-muted mt-1.5">
                <span>5% — less test data</span>
                <span>50% — more test data</span>
              </div>
            </div>

            {/* Optuna trials */}
            <div>
              <label className="label">Optuna Trials per Model</label>
              <div className="flex items-start gap-4">
                <input
                  type="number"
                  min="1" max="50"
                  value={localNTrials}
                  onChange={(e) => { setLocalNTrials(e.target.value); setSavedBanner(false) }}
                  className="input w-28"
                  placeholder="5"
                />
                <p className="text-xs text-text-muted pt-2 leading-relaxed">
                  Higher = better tuning, slower training<br />
                  Recommended: 5–20
                </p>
              </div>
            </div>

            <div className="divider" />

            {/* Actions */}
            <div className="flex items-center gap-3 flex-wrap">
              <button
                className="btn-primary"
                onClick={handleSaveConfig}
                disabled={!localTarget}
              >
                <CheckCircle size={15} />
                Save Configuration
              </button>

              {savedBanner && (
                <button
                  className="btn-ghost"
                  onClick={() => navigate('/train')}
                >
                  Go to Train
                  <ChevronRight size={15} />
                </button>
              )}

              <button
                className="btn-ghost ml-auto"
                onClick={handleReset}
              >
                <RotateCcw size={13} />
                Reset
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  )
}