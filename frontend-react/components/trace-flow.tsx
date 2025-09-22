"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { ReactFlow, Background, Controls, MiniMap, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";

export type RFTraceEntry = { stage: string } & Record<string, unknown>;
export type RFTrace = RFTraceEntry[];

const elk = new ELK();

function kindForStage(stage: string): string {
  const s = stage.toLowerCase();
  if (s.includes("normalize")) return "normalize";
  if (s.includes("extract")) return "extract";
  if (s.includes("evidence") || s.includes("retrieve")) return "docs";
  if (s === "sql" || s.includes("sql")) return "sql";
  if (s.includes("compose")) return "compose";
  if (s.includes("nitpicker")) return "nitpicker";
  if (s.includes("validate")) return "validate";
  if (s.includes("ask")) return "ask";
  return "stage";
}

function styleFor(kind: string): React.CSSProperties {
  // Light backgrounds; consistent dark text
  const base: React.CSSProperties = {
    padding: 8,
    borderRadius: 6,
    color: "#111",
    border: "1px solid #CBD5E1",
  };
  const bgMap: Record<string, string> = {
    normalize: "#E6F7FF", // light blue
    extract: "#E9F5FF", // very light blue
    docs: "#E8F5F7", // light teal
    sql: "#E8F5E9", // light green
    compose: "#FFF7E6", // light orange
    nitpicker: "#F3E8FF", // light purple
    validate: "#F1F5F9", // slate-100
    ask: "#FFF1F2", // rose-50
    stage: "#F8FAFC", // very light gray
    detail: "#F5F5F5",
  };
  return { ...base, backgroundColor: bgMap[kind] ?? bgMap.stage };
}

function toGraph(trace: RFTrace) {
  const stageNodes = trace.map((t, i) => ({
    id: `stage-${i}`,
    data: { label: t.stage, full: t },
    position: { x: 0, y: 0 },
    style: styleFor(kindForStage(t.stage)),
  }));

  const detailNodes: Node[] = [];
  const edges: Edge[] = [];

  trace.forEach((t, i) => {
    const parentId = `stage-${i}`;
    const nit = (t as unknown as { nitpicker?: { rounds?: Array<Record<string, unknown>> } }).nitpicker;
    if (nit?.rounds && Array.isArray(nit.rounds)) {
      (nit.rounds as Array<Record<string, unknown>>).forEach((r, j) => {
        const id = `${parentId}-round-${j}`;
        detailNodes.push({
          id,
          data: { label: `nitpicker • round ${j + 1} • score ${r.score ?? "?"}`, full: r },
          position: { x: 0, y: 0 },
          style: styleFor("nitpicker"),
        });
        edges.push({ id: `${parentId}->${id}`, source: parentId, target: id });
      });
    }
    if (i < trace.length - 1) {
      edges.push({ id: `${parentId}->stage-${i + 1}`, source: parentId, target: `stage-${i + 1}` });
    }
  });

  return { nodes: [...stageNodes, ...detailNodes], edges };
}

async function elkLayout(nodes: Node[], edges: Edge[]) {
  const graph: { id: string; layoutOptions: Record<string,string>; children: Array<{id:string;width:number;height:number}>; edges: Array<{id:string;sources:string[];targets:string[]}> } = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.layered.spacing.nodeNodeBetweenLayers": "40",
      "elk.spacing.nodeNode": "24",
      "elk.direction": "DOWN",
    },
    children: nodes.map((n) => ({ id: n.id, width: 200, height: 56 })),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };
  const res = await elk.layout(graph);
  const positions: Record<string, { x: number; y: number }> = {};
  (res.children || []).forEach((c: { id: string; x: number; y: number }) => {
    positions[c.id] = { x: c.x, y: c.y };
  });
  return nodes.map((n) => ({ ...n, position: positions[n.id] || n.position }));
}

export default function TraceFlow({ trace }: { trace: RFTrace }) {
  const { nodes: rawNodes, edges } = useMemo(() => toGraph(trace || []), [trace]);
  const [nodes, setNodes] = useState<Node[]>(rawNodes);
  const [selected, setSelected] = useState<RFTraceEntry | null>(null);
  const [open, setOpen] = useState(false);
  const [locked, setLocked] = useState(false);
  const dwellRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const asideRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    elkLayout(rawNodes, edges).then(setNodes).catch(() => setNodes(rawNodes));
  }, [trace]);

  // helpers
  function clearDwell() {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
  }

  function startDwell(next: RFTraceEntry) {
    setSelected(next);
    setOpen(true);
    setLocked(false);
    clearDwell();
    timerRef.current = window.setTimeout(() => {
      setLocked(true);
    }, 3000);
  }

  // Close on Escape or outside click (pane click handled separately)
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        clearDwell();
        setLocked(false);
        setOpen(false);
        setSelected(null);
      }
    }
    function onDocDown(e: MouseEvent) {
      const el = asideRef.current;
      if (!el) return;
      if (!el.contains(e.target as Node)) {
        clearDwell();
        setLocked(false);
        setOpen(false);
        setSelected(null);
      }
    }
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDocDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDocDown);
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      {(!trace || trace.length === 0) ? (
        <div className="text-xs opacity-70">No flow available for this message.</div>
      ) : null}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        onNodeMouseEnter={(_e: unknown, node: Node) => {
          const full = (node as unknown as { data?: { full?: RFTraceEntry } }).data?.full;
          if (full) startDwell(full);
        }}
        onNodeMouseLeave={() => {
          if (!locked) {
            clearDwell();
            setOpen(false);
            setSelected(null);
          }
        }}
        onPaneClick={() => {
          clearDwell();
          setLocked(false);
          setOpen(false);
          setSelected(null);
        }}
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
      {open && selected ? (
        <aside ref={asideRef} className="absolute right-2 top-2 w-[320px] max-h-[85vh] overflow-auto bg-popover text-popover-foreground border border-border rounded p-3 shadow">
          <div className="font-medium mb-2">{selected.stage || "Details"}</div>
          <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(selected, null, 2)}</pre>
        </aside>
      ) : null}
    </div>
  );
}


