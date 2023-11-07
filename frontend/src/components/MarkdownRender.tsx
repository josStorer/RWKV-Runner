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

const MarkdownRender: FC<ReactMarkdownOptions> = (props) => {
  return (
    <div dir="auto" className="markdown-body">
      <ReactMarkdown
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
    </div>
  );
};

export default MarkdownRender;
