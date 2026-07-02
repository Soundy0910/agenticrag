/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          50:  '#f8f9fb',
          100: '#f0f2f6',
          200: '#e4e8ef',
          800: '#171b27',
          825: '#131720',
          850: '#10141c',
          900: '#0b0e15',
          950: '#060810',
        },
        brand: {
          400: '#7c8df8',
          500: '#5b6cf4',
          600: '#4254e0',
        },
        accent: {
          400: '#34d399',
          500: '#10b981',
        },
        warn: {
          400: '#fbbf24',
          500: '#f59e0b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        '3xs': ['0.5625rem', { lineHeight: '0.875rem' }],
        '2xs': ['0.625rem',  { lineHeight: '1rem'    }],
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
      },
      keyframes: {
        fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(8px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
