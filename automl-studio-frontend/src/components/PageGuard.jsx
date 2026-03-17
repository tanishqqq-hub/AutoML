/**
 * components/PageGuard.jsx
 *
 * Protects pages that depend on prior pipeline steps.
 * Shows a clear, actionable message instead of crashing
 * when required state is missing from the store.
 *
 * Usage:
 *   <PageGuard require="config">
 *     <TrainPage />
 *   </PageGuard>
 *
 * Guard levels:
 *   'upload'     — needs a file uploaded (filePath must exist)
 *   'config'     — needs upload + config saved
 *   'experiment' — needs upload + config + experiment started
 *   'complete'   — needs upload + config + training complete
 */

import { useNavigate } from 'react-router-dom'
import { AlertTriangle, ArrowRight } from 'lucide-react'
import useStore from '../store/useStore'

// ── Guard level definitions ───────────────────────────────────────
const GUARDS = {
  upload: {
    check:   (s) => !!s.filePath,
    message: 'You need to upload a dataset first.',
    action:  'Go to Upload',
    path:    '/',
  },
  config: {
    check:   (s) => !!s.filePath && s.configSaved,
    message: 'You need to upload a dataset and save your configuration first.',
    action:  'Go to Upload & Configure',
    path:    '/',
  },
  experiment: {
    check:   (s) => !!s.filePath && s.configSaved && !!s.experimentId,
    message: 'You need to run the training pipeline first.',
    action:  'Go to Train',
    path:    '/train',
  },
  complete: {
    check:   (s) => !!s.experimentId && s.trainingStatus === 'complete',
    message: 'Training must be complete before accessing this page.',
    action:  'Go to Train',
    path:    '/train',
  },
}

export default function PageGuard({ require: level, children }) {
  const navigate  = useNavigate()
  const state     = useStore()
  const guard     = GUARDS[level]

  // No guard defined or check passes — render children
  if (!guard || guard.check(state)) {
    return children
  }

  return (
    <div className="animate-fade-in space-y-6">
      <div className="card border-accent-gold/30 flex flex-col gap-4 max-w-lg">

        {/* Icon + message */}
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-accent-gold/15 border border-accent-gold/25 flex items-center justify-center shrink-0">
            <AlertTriangle size={16} className="text-accent-gold" />
          </div>
          <div className="space-y-1 pt-0.5">
            <div
              className="text-sm font-semibold text-text-primary"
              style={{ fontFamily: 'Syne, sans-serif' }}
            >
              Step required
            </div>
            <p className="text-sm text-text-secondary">
              {guard.message}
            </p>
          </div>
        </div>

        {/* Action button */}
        <button
          className="btn-ghost self-start flex items-center gap-2"
          onClick={() => navigate(guard.path)}
        >
          {guard.action}
          <ArrowRight size={14} />
        </button>

      </div>
    </div>
  )
}