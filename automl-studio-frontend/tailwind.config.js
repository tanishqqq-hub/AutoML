/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans:    ['"DM Sans"',         'sans-serif'],
        mono:    ['"JetBrains Mono"',  'monospace'],
        display: ['"Syne"',            'sans-serif'],
      },
      colors: {
        bg: {
          canvas:  '#05080f',
          base:    '#080d17',
          surface: '#0c1220',
          raised:  '#111827',
          border:  '#1a2540',
          hover:   '#1e2d4a',
        },
        primary: {
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
        },
        accent: {
          blue:    '#38bdf8',
          gold:    '#f59e0b',
          green:   '#10b981',
          red:     '#ef4444',
          purple:  '#a78bfa',
        },
        text: {
          primary:   '#f1f5f9',
          secondary: '#94a3b8',
          muted:     '#475569',
          faint:     '#1e3a5f',
        },
      },
      boxShadow: {
        'card':       '0 1px 3px rgba(0,0,0,0.5)',
        'card-lg':    '0 4px 16px rgba(0,0,0,0.6)',
        'glow-teal':  '0 0 20px rgba(45,212,191,0.2)',
        'glow-blue':  '0 0 20px rgba(56,189,248,0.2)',
      },
      animation: {
        'fade-in':   'fadeIn 0.35s ease forwards',
        'slide-up':  'slideUp 0.3s ease forwards',
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:   { from: { opacity: 0 },                        to: { opacity: 1 } },
        slideUp:  { from: { opacity: 0, transform: 'translateY(10px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        pulseDot: {
          '0%,100%': { opacity: 1,   transform: 'scale(1)'    },
          '50%':     { opacity: 0.4, transform: 'scale(0.85)' },
        },
      },
    },
  },
  plugins: [],
}