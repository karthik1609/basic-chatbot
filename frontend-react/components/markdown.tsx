import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        code(props) {
          const { className, children } = props as unknown as { className?: string; children?: React.ReactNode };
          const text = String(children ?? '').replace(/\n$/, '');
          // heuristic: treat as block if contains newline
          const isBlock = /\n/.test(text);
          if (isBlock) {
            return (
              <pre className="relative group">
                <code className={className}>{text}</code>
                <button
                  type="button"
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-xs bg-secondary px-2 py-1 rounded"
                  onClick={() => navigator.clipboard.writeText(text)}
                >
                  Copy
                </button>
              </pre>
            );
          }
          return <code className={className}>{text}</code>;
        }
      }}
    >{children}</ReactMarkdown>
  );
}
