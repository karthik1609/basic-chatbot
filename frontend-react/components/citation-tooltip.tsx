import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

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
  const [portalEl, setPortalEl] = useState<HTMLElement | null>(null);
  const [coords, setCoords] = useState<{ top: number; left: number; maxWidth: number }>({ top: 0, left: 0, maxWidth: 960 });

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

  // Setup portal host once on client
  useEffect(() => {
    if (typeof document === 'undefined') return;
    const el = document.createElement('div');
    el.setAttribute('data-citation-portal', '');
    document.body.appendChild(el);
    setPortalEl(el);
    return () => {
      document.body.removeChild(el);
    };
  }, []);

  // Position tooltip near the trigger using viewport coordinates
  useEffect(() => {
    if (!open) return;
    function compute() {
      const host = rootRef.current;
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const margin = 8;
      const vw = Math.max(document.documentElement.clientWidth, window.innerWidth || 0);
      const desiredWidth = Math.min(vw * 0.9, 960);
      let left = rect.left;
      if (left + desiredWidth > vw - margin) left = Math.max(margin, vw - desiredWidth - margin);
      const top = rect.bottom + margin;
      setCoords({ top, left, maxWidth: desiredWidth });
    }
    compute();
    window.addEventListener('resize', compute);
    window.addEventListener('scroll', compute, true);
    return () => {
      window.removeEventListener('resize', compute);
      window.removeEventListener('scroll', compute, true);
    };
  }, [open]);

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
      {open && portalEl
        ? createPortal(
            <div
              id={tooltipId}
              role="tooltip"
              style={{ position: 'fixed', top: coords.top, left: coords.left, maxWidth: coords.maxWidth }}
              className="z-[9999] bg-popover text-popover-foreground text-xs p-3 rounded border border-border shadow overflow-auto w-[min(90vw,60rem)] max-h-[70vh] whitespace-pre-wrap"
            >
              {content}
            </div>,
            portalEl
          )
        : null}
    </span>
  );
}
