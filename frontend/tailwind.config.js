const markdownElements = [
  'div',
  'p',
  'span',

  'video',
  'img',

  'abbr',
  'acronym',
  'b',
  'blockquote',
  'code',
  'em',
  'i',
  'li',
  'ol',
  'ul',
  'strong',
  'table',
  'tr',
  'td',
  'th',

  'details',
  'summary',
  'kbd',
  'samp',
  'sub',
  'sup',
  'ins',
  'del',
  'var',
  'q',
  'dl',
  'dt',
  'dd',
  'ruby',
  'rt',
  'rp',

  'br',
  'hr',

  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',

  'thead',
  'tbody',
  'tfoot',
  'u',
  's',
  'a',
  'pre',
  'cite',
]

const markdownPseudoElements = ['::marker', '::before', '::after']

const tableElements = ['table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot']

const proseStyles = {
  color: 'inherit',
}

const tableProseStyles = {
  ...proseStyles,
  borderWidth: 'thin',
  borderColor: '#d2d2d5',
}

const elementsStyles = markdownElements.reduce((acc, element) => {
  let styles = proseStyles
  if (tableElements.includes(element)) styles = tableProseStyles

  acc[element] = styles
  markdownPseudoElements.forEach((pseudo) => {
    acc[element + pseudo] = styles
  })
  return acc
}, {})

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            color: 'inherit',
            fontSize: 'inherit',
            ...elementsStyles,
          },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
