import 'katex/dist/katex.min.css'
import { FC, useEffect, useMemo, useRef, useState } from 'react'
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
} from '@fluentui/react-components'
import { MermaidDiagram } from '@lightenna/react-mermaid-diagram'
import classNames from 'classnames'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import { ReactMarkdownOptions } from 'react-markdown/lib/react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { BrowserOpenURL } from '../../wailsjs/runtime'
import commonStore from '../stores/commonStore'

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

const ThinkComponent: FC<any> = ({ node, children, ...props }) => {
  const { t } = useTranslation()
  const ref = useRef<HTMLDivElement>(null)
  const isEnded = ref.current?.parentElement?.classList.contains('think-ended')
  const isEmpty =
    !children ||
    (Array.isArray(children) &&
      (children.length === 0 ||
        (children.length === 1 &&
          typeof children[0] === 'string' &&
          children[0].trim() === '')))

  return isEmpty ? (
    <></>
  ) : (
    <div ref={ref} {...props}>
      <Accordion collapsible defaultOpenItems={['default']}>
        <AccordionItem value="default">
          <AccordionHeader size="small">
            {isEnded ? t('Thinking Ended') : t('Thinking...')}
          </AccordionHeader>
          <AccordionPanel>
            <div
              style={{ whiteSpace: 'pre-wrap' }}
              className={classNames(
                commonStore.settings.darkMode
                  ? 'text-[#d6d6d6]'
                  : 'text-[#777777]'
              )}
            >
              {children}
            </div>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </div>
  )
}

const extractText = (node: any): string => {
  if (node === null || node === undefined) return ''
  if (typeof node === 'string') return node
  if (Array.isArray(node)) return node.map(extractText).join('')
  if (typeof node === 'object' && 'props' in node)
    return extractText(node.props?.children)
  if (typeof node === 'object' && 'children' in node)
    return extractText(node.children)
  return ''
}

const MermaidComponent: FC<any> = ({ children }) => {
  const chart = extractText(children).trim()
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setError(null)
  }, [chart])
  if (!chart) return <></>
  const theme = commonStore.settings.darkMode ? 'dark' : 'default'
  if (error) {
    return (
      <div className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-700">
        <div className="font-semibold">Mermaid render error</div>
        <div className="whitespace-pre-wrap">{error}</div>
        <pre className="mt-2 whitespace-pre-wrap rounded bg-white/70 p-2 text-xs text-neutral-700">
          {chart}
        </pre>
      </div>
    )
  }
  return (
    <MermaidDiagram
      theme={theme}
      suppressErrorRendering={true}
      onError={(err) => {
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'string'
              ? err
              : String(err)
        setError(message)
      }}
    >
      {chart}
    </MermaidDiagram>
  )
}

const wrapMermaidLabels = (content: string) =>
  content.replace(/\[([^\]\n]*)\]/g, (match, inner) => {
    const trimmed = inner.trim()
    if (
      (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
      (trimmed.startsWith("'") && trimmed.endsWith("'"))
    )
      return match
    if (!/[:;]|\\n|["“”]/.test(inner)) return match
    const escapedInner = inner.replace(/"/g, '\\"')
    return `["${escapedInner}"]`
  })

const convertHtmlCommentsToMermaid = (content: string) =>
  content.replace(/<!--([\s\S]*?)-->/g, (_, inner: string) => {
    const compact = inner.replace(/\s+/g, ' ').trim()
    return compact ? `\n%% ${compact}\n` : ''
  })

const stripHtmlTags = (content: string) =>
  content.replace(/<[^>]+>/g, '').replace(/&nbsp;/gi, ' ')

const normalizeMermaidBlocks = (value: string) =>
  value.replace(/```mermaid([\s\S]*?)```/gi, (match, content) => {
    const normalizedContent = wrapMermaidLabels(
      stripHtmlTags(
        convertHtmlCommentsToMermaid(content.replace(/<br\s*\/?>/gi, '\\n'))
      )
    )
    const trimmed = normalizedContent.replace(/^\s+|\s+$/g, '')
    return `\n\n<mermaid>\n\n${trimmed}\n\n</mermaid>\n\n`
  })

const MarkdownRender: FC<
  ReactMarkdownOptions & { disabled?: boolean; thinkEnded?: boolean }
> = (props) => {
  const markdownContent = useMemo(() => {
    if (typeof props.children !== 'string') return props.children
    return normalizeMermaidBlocks(props.children)
  }, [props.children])

  return (
    <div
      dir="auto"
      className={classNames(
        'markdown-body prose',
        props.thinkEnded ? 'think-ended' : ''
      )}
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

            'think',
            'mermaid',

            'svg',
            'style',
            'defs',
            'g',
            'path',
            'line',
            'polyline',
            'polygon',
            'rect',
            'circle',
            'ellipse',
            'marker',
            'text',
            'tspan',
            'foreignObject',
            'desc',
            'title',
            'linearGradient',
            'radialGradient',
            'stop',
            'clipPath',
            'mask',
            'pattern',
            'use',
            'filter',
            'feDropShadow',
            'feGaussianBlur',
            'feColorMatrix',
            'feOffset',
            'feBlend',
            'feComponentTransfer',
            'feFuncR',
            'feFuncG',
            'feFuncB',
            'feFuncA',
            'feComposite',
            'feFlood',
            'feImage',
            'feMerge',
            'feMergeNode',
            'feMorphology',
            'feTurbulence',
            'feDisplacementMap',
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
            // @ts-ignore
            think: ThinkComponent,
            mermaid: MermaidComponent,
          }}
        >
          {markdownContent}
        </ReactMarkdown>
      )}
    </div>
  )
}

export default MarkdownRender
