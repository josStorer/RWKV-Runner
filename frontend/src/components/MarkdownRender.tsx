import 'katex/dist/katex.min.css'
import { FC } from 'react'
import ReactMarkdown from 'react-markdown'
import { ReactMarkdownOptions } from 'react-markdown/lib/react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { BrowserOpenURL } from '../../wailsjs/runtime'

const Hyperlink: FC<any> = ({ href, children }) => {
  return (
    <span
      style={{ color: '#8ab4f8', cursor: 'pointer' }}
      onClick={() => {
        BrowserOpenURL(href)
      }}
    >
      {/*@ts-ignore*/}
      {children}
    </span>
  )
}

const MarkdownRender: FC<ReactMarkdownOptions & { disabled?: boolean }> = (
  props
) => {
  return (
    <div
      dir="auto"
      className="markdown-body prose"
      style={{ maxWidth: '100%' }}
    >
      {props.disabled ? (
        <div style={{ whiteSpace: 'pre-wrap' }} className={props.className}>
          {props.children}
        </div>
      ) : (
        <ReactMarkdown
          className={props.className}
          allowedElements={[
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
          ]}
          unwrapDisallowed={true}
          remarkPlugins={[remarkMath, remarkGfm, remarkBreaks]}
          rehypePlugins={[
            rehypeKatex,
            rehypeRaw,
            [
              rehypeHighlight,
              {
                detect: true,
                ignoreMissing: true,
              },
            ],
          ]}
          components={{
            a: Hyperlink,
          }}
        >
          {props.children}
        </ReactMarkdown>
      )}
    </div>
  )
}

export default MarkdownRender
