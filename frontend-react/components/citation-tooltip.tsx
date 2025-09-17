import { useState } from 'react';

type Props = {
  label: string;
  content: string;
};

export function CitationTooltip({ label, content }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <span
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      className="relative underline decoration-dotted cursor-help"
      aria-label={`${label} citation`}
    >
      {label}
      {open && (
        <span
          role="tooltip"
          className="absolute left-0 top-full mt-1 z-50 bg-popover text-popover-foreground text-xs p-2 rounded border border-border max-w-sm whitespace-pre-wrap shadow"
        >
          {content}
        </span>
      )}
    </span>
  );
}
