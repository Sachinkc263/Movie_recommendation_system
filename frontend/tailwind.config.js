/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f0f0f',
        card: '#1a1a1a',
        accent: '#e50914',
        'accent-hover': '#b20710',
        surface: '#242424',
        border: '#2a2a2a',
        muted: '#6b7280',
        'text-primary': '#f9fafb',
        'text-secondary': '#9ca3af',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
