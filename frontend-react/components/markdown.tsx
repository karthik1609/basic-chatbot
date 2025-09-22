import React from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CitationTooltip } from './citation-tooltip';

type CitationLite = { tag: string; text?: string; query?: string };

type MarkdownProps = {
  children: string;
  citations?: CitationLite[];
};

/** --- Local, wider prop types for renderers (compatible with react-markdown) --- */
type ParagraphRendererProps = React.ComponentPropsWithoutRef<'p'> & {
  node?: unknown;
  children?: React.ReactNode;
};

type CodeRendererProps = React.ComponentPropsWithoutRef<'code'> & {
  inline?: boolean;
  node?: unknown;
  className?: string;
  children?: React.ReactNode;
};

// Generic renderer props helper for various tags
type ElementRendererProps<T extends keyof React.JSX.IntrinsicElements> = React.ComponentPropsWithoutRef<T> & {
  node?: unknown;
  children?: React.ReactNode;
  className?: string;
};

/** --- Utilities (strictly typed) --- */
function isReactElement(
  node: React.ReactNode
): node is React.ReactElement<Record<string, unknown>> {
  return React.isValidElement(node);
}

function getElementChildren(
  el: React.ReactElement<Record<string, unknown>>
): React.ReactNode | undefined {
  const props = el.props as Readonly<Record<string, unknown>>;
  return props.children as React.ReactNode | undefined;
}

function nodeToPlainText(node: React.ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(nodeToPlainText).join('');
  if (isReactElement(node)) return nodeToPlainText(getElementChildren(node));
  return '';
}

/** Replace [Dx]/[Sx] tokens with <CitationTooltip/>, recursing into inline children */
function renderWithTooltips(
  nodes: React.ReactNode,
  citations: ReadonlyArray<CitationLite>
): React.ReactNode {
  return React.Children.map(nodes, (node, outerIdx) => {
    if (typeof node === 'string') {
      // Split on bracketed citations starting with D#/S# and allow optional extra text inside the brackets
      const parts = node.split(/(\[(?:D|S)\d+[^\]]*\])/g);
      return parts.map((part, idx) => {
        const m = /^\[((?:D|S)\d+)(?:[^\]]*)\]$/.exec(part);
        if (m) {
          const tagKey = m[1];
          const displayLabel = part; // preserve any section suffix like "[D1 §2.23]"
          const c = citations.find((x) => x.tag === tagKey || `[${x.tag}]` === `[${tagKey}]`);
          const content = c?.text ?? c?.query ?? 'See citations panel';
          return <CitationTooltip key={`${outerIdx}-${idx}`} label={displayLabel} content={content} />;
        }
        return <React.Fragment key={`${outerIdx}-${idx}`}>{part}</React.Fragment>;
      });
    }

    if (isReactElement(node)) {
      const child = getElementChildren(node);
      return React.cloneElement(node, undefined, child ? renderWithTooltips(child, citations) : child);
    }

    return node;
  });
}

export function Markdown({ children, citations = [] }: MarkdownProps) {
  const components: Components = {
    p({ children: nodeChildren }: ParagraphRendererProps) {
      return <p>{renderWithTooltips(nodeChildren, citations)}</p>;
    },

    blockquote({ children: nodeChildren, className }: ElementRendererProps<'blockquote'>) {
      return <blockquote className={className}>{renderWithTooltips(nodeChildren, citations)}</blockquote>;
    },

    strong({ children: nodeChildren, className }: ElementRendererProps<'strong'>) {
      return <strong className={className}>{renderWithTooltips(nodeChildren, citations)}</strong>;
    },

    em({ children: nodeChildren, className }: ElementRendererProps<'em'>) {
      return <em className={className}>{renderWithTooltips(nodeChildren, citations)}</em>;
    },

    li({ children: nodeChildren, className }: ElementRendererProps<'li'>) {
      return <li className={className}>{renderWithTooltips(nodeChildren, citations)}</li>;
    },

    h1({ children: nodeChildren, className }: ElementRendererProps<'h1'>) {
      return <h1 className={className}>{renderWithTooltips(nodeChildren, citations)}</h1>;
    },
    h2({ children: nodeChildren, className }: ElementRendererProps<'h2'>) {
      return <h2 className={className}>{renderWithTooltips(nodeChildren, citations)}</h2>;
    },
    h3({ children: nodeChildren, className }: ElementRendererProps<'h3'>) {
      return <h3 className={className}>{renderWithTooltips(nodeChildren, citations)}</h3>;
    },
    h4({ children: nodeChildren, className }: ElementRendererProps<'h4'>) {
      return <h4 className={className}>{renderWithTooltips(nodeChildren, citations)}</h4>;
    },
    h5({ children: nodeChildren, className }: ElementRendererProps<'h5'>) {
      return <h5 className={className}>{renderWithTooltips(nodeChildren, citations)}</h5>;
    },
    h6({ children: nodeChildren, className }: ElementRendererProps<'h6'>) {
      return <h6 className={className}>{renderWithTooltips(nodeChildren, citations)}</h6>;
    },

    td({ children: nodeChildren, className }: ElementRendererProps<'td'>) {
      return <td className={className}>{renderWithTooltips(nodeChildren, citations)}</td>;
    },

    th({ children: nodeChildren, className }: ElementRendererProps<'th'>) {
      return <th className={className}>{renderWithTooltips(nodeChildren, citations)}</th>;
    },

    code({ inline, className, children: nodeChildren }: CodeRendererProps) {
      const isInline = Boolean(inline);
      const text = nodeToPlainText(nodeChildren).replace(/\n$/, '');

      if (!isInline) {
        return (
          <pre className="relative group">
            <code className={className}>{text}</code>
            <button
              type="button"
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-xs bg-secondary px-2 py-1 rounded"
              onClick={() => {
                if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
                  navigator.clipboard.writeText(text);
                }
              }}
              aria-label="Copy code"
            >
              Copy
            </button>
          </pre>
        );
      }

      return <code className={className}>{nodeChildren}</code>;
    },
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={components}
    >
      {children}
    </ReactMarkdown>
  );
}
