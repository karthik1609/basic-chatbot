declare module "@xyflow/react" {
  export const ReactFlow: any;
  export const Background: any;
  export const Controls: any;
  export const MiniMap: any;
  export type Node = any;
  export type Edge = any;
}

declare module "elkjs" {
  export default class ELK {
    layout(graph: any): Promise<any>;
  }
}

declare module "elkjs/lib/elk.bundled.js" {
  export default class ELK {
    layout(graph: any): Promise<any>;
  }
}


