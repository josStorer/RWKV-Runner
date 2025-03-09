import 'katex/dist/katex.min.css'
import { FC, useRef } from 'react'
import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
} from '@fluentui/react-components'
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
            {isEnded ? t('思考结束') : t('思考中...')}
          </AccordionHeader>
          <AccordionPanel>
            <div
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

const MarkdownRender: FC<
  ReactMarkdownOptions & { disabled?: boolean; thinkEnded?: boolean }
> = (props) => {
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
          }}
        >
          {props.children}
        </ReactMarkdown>
      )}
    </div>
  )
}

export default MarkdownRender
