/**
 * pages/NotFound.jsx
 * 404 page for unknown routes.
 */

import { useNavigate } from 'react-router-dom'
import { AlertTriangle, ArrowLeft } from 'lucide-react'

export default function NotFound() {
  const navigate = useNavigate()
  return (
    <div className="min-h-[60vh] flex items-center justify-center animate-fade-in">
      <div className="text-center space-y-6 max-w-sm">
        <div className="w-16 h-16 rounded-2xl bg-accent-gold/10 border border-accent-gold/25 flex items-center justify-center mx-auto">
          <AlertTriangle size={28} className="text-accent-gold" />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-text-primary mono mb-2">404</h1>
          <p className="text-text-secondary">This page does not exist.</p>
        </div>
        <button className="btn-ghost" onClick={() => navigate('/')}>
          <ArrowLeft size={14} />
          Back to Upload
        </button>
      </div>
    </div>
  )
}