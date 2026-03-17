/**
 * components/Layout.jsx
 *
 * Main app shell — persistent sidebar + content area.
 *
 * Sidebar shows:
 *   - Brand logo
 *   - Navigation with step completion indicators
 *   - Live experiment status dot
 *   - Session info footer (dataset, target, experiment ID)
 *   - Stale session banner when backend restarts
 */

import { NavLink } from 'react-router-dom'
import { useState } from 'react'
import {
  Upload, Cpu, BarChart2, Zap,
  Activity, Circle, AlertTriangle, X,
  CheckCircle,
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { getExperimentStatus } from '../api/client'
import useStore from '../store/useStore'

// ── Navigation items ──────────────────────────────────────────────
const NAV_ITEMS = [
  { to: '/',        icon: Upload,   label: 'Upload & Configure', end: true  },
  { to: '/train',   icon: Cpu,      label: 'Train Pipeline',     end: false },
  { to: '/results', icon: BarChart2, label: 'Results',           end: false },
  { to: '/predict', icon: Zap,      label: 'Predict',            end: false },
]

// ── Step completion dot ───────────────────────────────────────────
function StepDot({ complete }) {
  if (!complete) return (
    <span className="w-1.5 h-1.5 rounded-full bg-bg-border shrink-0" />
  )
  return (
    <CheckCircle size={12} className="text-primary-400 shrink-0" />
  )
}

// ── Status dot ────────────────────────────────────────────────────
function StatusDot({ status }) {
  if (!status) return null
  const styles = {
    pending:  'bg-accent-gold animate-pulse-dot',
    running:  'bg-primary-400 animate-pulse-dot',
    complete: 'bg-accent-green',
    failed:   'bg-accent-red',
  }
  return (
    <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${styles[status] || 'bg-text-muted'}`} />
  )
}

// ── Sidebar info row ──────────────────────────────────────────────
function InfoRow({ label, value, mono = false }) {
  if (!value) return null
  return (
    <div className="space-y-0.5 min-w-0">
      <div className="text-[10px] text-text-muted uppercase tracking-widest">{label}</div>
      <div
        className={`text-xs text-text-secondary truncate ${mono ? 'mono text-[11px]' : ''}`}
        title={value}
      >
        {value}
      </div>
    </div>
  )
}

// ── Stale session banner ──────────────────────────────────────────
function StaleBanner({ experimentId }) {
  const [dismissed, setDismissed] = useState(false)

  const { error } = useQuery({
    queryKey:  ['session-check', experimentId],
    queryFn:   () => getExperimentStatus(experimentId),
    retry:     false,
    staleTime: 30_000,
    enabled:   !!experimentId,
  })

  const isStale = error?.message?.toLowerCase().includes('not found')
  if (!isStale || dismissed) return null

  return (
    <div className="mx-6 mt-4 animate-slide-up">
      <div className="alert-warning text-xs">
        <AlertTriangle size={13} className="shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="font-medium mb-0.5">Session expired</p>
          <p className="text-text-secondary">
            Backend was restarted. Go to Upload and retrain.
          </p>
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 p-0.5 hover:opacity-70 transition-opacity"
        >
          <X size={12} />
        </button>
      </div>
    </div>
  )
}

// ── Main Layout ───────────────────────────────────────────────────
export default function Layout({ children }) {
  const {
    filename,
    targetColumn,
    experimentId,
    trainingStatus,
    configSaved,
    bestMetrics,
    featureColumns,
  } = useStore()

  // Step completion states
  const stepDone = {
    upload:  !!filename,
    config:  configSaved,
    train:   trainingStatus === 'complete',
    results: !!bestMetrics,
    predict: featureColumns?.length > 0,
  }

  const navCompletion = {
    '/':        stepDone.upload && stepDone.config,
    '/train':   stepDone.train,
    '/results': stepDone.results,
    '/predict': stepDone.predict,
  }

  return (
    <div className="flex min-h-screen bg-bg-canvas">

      {/* ── Sidebar ── */}
      <aside className="w-[220px] shrink-0 flex flex-col bg-bg-base border-r border-bg-border sticky top-0 h-screen overflow-y-auto">

        {/* Brand */}
        <div className="px-5 py-5 border-b border-bg-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-primary-400/15 border border-primary-400/30">
              <Activity size={15} className="text-primary-400" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-bold text-text-primary leading-none" style={{ fontFamily: 'Syne, sans-serif' }}>
                AutoML Studio
              </div>
              <div className="text-[10px] text-text-muted mt-0.5 tracking-widest uppercase">
                Enterprise · v2.0
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={15} className="shrink-0" />
              <span className="flex-1 truncate">{label}</span>
              <StepDot complete={navCompletion[to]} />
            </NavLink>
          ))}
        </nav>

        {/* Session info */}
        <div className="px-4 py-4 border-t border-bg-border space-y-3">
          {trainingStatus && (
            <div className="flex items-center gap-2">
              <StatusDot status={trainingStatus} />
              <span className="text-xs text-text-muted capitalize">{trainingStatus}</span>
            </div>
          )}
          <InfoRow label="Dataset"    value={filename}      />
          <InfoRow label="Target"     value={targetColumn}  />
          <InfoRow label="Experiment" value={experimentId} mono />
        </div>

      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 min-w-0 overflow-auto">

        {/* Top bar */}
        <div className="sticky top-0 z-10 h-14 flex items-center px-8 bg-bg-canvas/80 backdrop-blur-sm border-b border-bg-border/50">
          <div className="flex items-center gap-2 text-text-muted">
            <Circle size={7} className="fill-primary-400 text-primary-400" />
            <span className="text-xs mono">
              {experimentId ? `exp · ${experimentId}` : 'no active experiment'}
            </span>
          </div>
        </div>

        {/* Stale session banner */}
        {experimentId && trainingStatus === 'complete' && (
          <StaleBanner experimentId={experimentId} />
        )}

        {/* Page content */}
        <div className="px-8 py-8 max-w-5xl mx-auto">
          {children}
        </div>

      </main>
    </div>
  )
}