"use client";
import React, { useEffect, useMemo, useState } from "react";
import { ReactFlow, Background, Controls, MiniMap, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";

export type RFTraceEntry = { stage: string } & Record<string, unknown>;
export type RFTrace = RFTraceEntry[];

const elk = new ELK();

function toGraph(trace: RFTrace) {
  const stageNodes = trace.map((t, i) => ({
    id: `stage-${i}`,
    data: { label: t.stage, full: t },
    position: { x: 0, y: 0 },
    style: { padding: 8, borderRadius: 6 },
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
          style: { padding: 6, borderRadius: 6 },
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

  useEffect(() => {
    elkLayout(rawNodes, edges).then(setNodes).catch(() => setNodes(rawNodes));
  }, [trace]);

  return (
    <div className="relative w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        onNodeClick={(_ev: unknown, node: Node) => setSelected((node as unknown as { data?: { full?: RFTraceEntry } }).data?.full ?? null)}
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
      {selected ? (
        <aside className="absolute right-2 top-2 w-[320px] max-h-[85vh] overflow-auto bg-popover text-popover-foreground border border-border rounded p-3 shadow">
          <div className="font-medium mb-2">{selected.stage || "Details"}</div>
          <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(selected, null, 2)}</pre>
        </aside>
      ) : null}
    </div>
  );
}


