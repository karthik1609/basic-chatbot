import { useEffect, useId, useRef, useState } from 'react';

type Props = {
  label: string;
  content: string;
};

export function CitationTooltip({ label, content }: Props) {
  const [open, setOpen] = useState(false);
  const [locked, setLocked] = useState(false);
  const [progress, setProgress] = useState(0); // 0..1 dwell progress
  const dwellMs = 3000;
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number | null>(null);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const tooltipId = useId();

  function stopDwell(resetProgress: boolean) {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    startRef.current = null;
    if (resetProgress) setProgress(0);
  }

  function startDwell() {
    if (locked) return;
    stopDwell(false);
    startRef.current = performance.now();
    const tick = () => {
      if (startRef.current == null) return;
      const elapsed = performance.now() - startRef.current;
      const p = Math.min(1, elapsed / dwellMs);
      setProgress(p);
      if (p >= 1) {
        setLocked(true);
        setProgress(1);
        stopDwell(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  // Outside click to close when locked/open
  useEffect(() => {
    if (!(open && locked)) return;
    function onDocDown(e: MouseEvent) {
      const el = rootRef.current;
      if (!el) return;
      if (!el.contains(e.target as Node)) {
        setLocked(false);
        setOpen(false);
        setProgress(0);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setLocked(false);
        setOpen(false);
        setProgress(0);
      }
    }
    document.addEventListener('mousedown', onDocDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, locked]);

  return (
    <span
      ref={rootRef}
      tabIndex={0}
      onMouseEnter={() => {
        setOpen(true);
        startDwell();
      }}
      onMouseLeave={() => {
        if (!locked) {
          setOpen(false);
          stopDwell(true);
        }
      }}
      onFocus={() => {
        setOpen(true);
        startDwell();
      }}
      onBlur={() => {
        if (!locked) {
          setOpen(false);
          stopDwell(true);
        }
      }}
      onPointerDown={() => {
        setOpen(true);
        startDwell();
      }}
      onPointerUp={() => {
        if (!locked) stopDwell(false);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          setLocked(false);
          setOpen(false);
          setProgress(0);
        }
      }}
      className="relative underline decoration-dotted cursor-help outline-none focus-visible:ring-2 focus-visible:ring-ring/50 rounded"
      aria-label={`${label} citation`}
      aria-describedby={open ? tooltipId : undefined}
    >
      {label}
      {!locked && open ? (
        <span
          aria-hidden
          className="absolute -right-4 -top-2 size-4 rounded-full border border-border bg-background shadow overflow-hidden"
          style={{
            background: `conic-gradient(currentColor ${Math.round(progress * 360)}deg, transparent 0)`,
          }}
        />
      ) : null}
      {open && (
        <span
          id={tooltipId}
          role="tooltip"
          className="absolute left-0 top-full mt-1 z-50 bg-popover text-popover-foreground text-xs p-3 rounded border border-border shadow overflow-auto w-[min(80vw,50rem)] max-h-96 whitespace-pre-wrap"
        >
          {content}
        </span>
      )}
    </span>
  );
}
