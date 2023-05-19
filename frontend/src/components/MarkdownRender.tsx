import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import {FC} from 'react';
import {ReactMarkdownOptions} from 'react-markdown/lib/react-markdown';

export const MarkdownRender: FC<ReactMarkdownOptions> = (props) => {
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
      >
        {props.children}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRender;
