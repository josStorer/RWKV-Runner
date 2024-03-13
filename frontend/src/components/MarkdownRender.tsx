import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { FC } from 'react';
import { ReactMarkdownOptions } from 'react-markdown/lib/react-markdown';
import { BrowserOpenURL } from '../../wailsjs/runtime';

const Hyperlink: FC<any> = ({ href, children }) => {
  return (
    <span
      style={{ color: '#8ab4f8', cursor: 'pointer' }}
      onClick={() => {
        BrowserOpenURL(href);
      }}
    >
      {/*@ts-ignore*/}
      {children}
    </span>
  );
};

const MarkdownRender: FC<ReactMarkdownOptions & { disabled?: boolean }> = (props) => {
  return (
    <div dir="auto" className="prose markdown-body" style={{ maxWidth: '100%' }}>
      {props.disabled ?
        <div style={{ whiteSpace: 'pre-wrap' }}>
          {props.children}
        </div> :
        <ReactMarkdown
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
            'cite'
          ]}
          unwrapDisallowed={true}
          remarkPlugins={[remarkGfm, remarkBreaks]}
          rehypePlugins={[
            rehypeRaw,
            [
              rehypeHighlight,
              {
                detect: true,
                ignoreMissing: true
              }
            ]
          ]}
          components={{
            a: Hyperlink
          }}
        >
          {props.children}
        </ReactMarkdown>
      }
    </div>
  );
};

export default MarkdownRender;
