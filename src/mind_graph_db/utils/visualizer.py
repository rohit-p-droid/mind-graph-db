"""Graph visualizer and debugging utility for Mind Graph DB."""

import json
import os
from typing import Any, Dict, List, Optional, Set

from mind_graph_db.interfaces.graph_store import GraphStore


class GraphVisualizer:
    """Utility for rendering ASCII trees, exporting JSON/DOT, and building interactive HTML visualizers."""

    @staticmethod
    def render_ascii(
        graph_store: GraphStore,
        seed_node_ids: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> str:
        """Render a human-readable ASCII graph tree representation.

        Args:
            graph_store: GraphStore instance.
            seed_node_ids: Optional list of starting seed node IDs.
            max_depth: Traversal depth.

        Returns:
            Formatted ASCII graph string.
        """
        lines: List[str] = ["=== Mind Graph DB Knowledge Graph Visualizer ==="]

        subgraph = graph_store.query_subgraph(start_node_ids=seed_node_ids or [], depth=max_depth)
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        if not nodes:
            lines.append(" (Empty graph - no nodes found)")
            return "\n".join(lines)

        lines.append(f"Nodes ({len(nodes)} total):")
        node_map = {n.id: n for n in nodes}
        for n in nodes:
            lines.append(f"  • [{n.node_type}] ID: {n.id} | Label: '{n.label}'")

        lines.append(f"\nRelationships ({len(relationships)} total):")
        for r in relationships:
            src_label = node_map[r.source_id].label if r.source_id in node_map else r.source_id
            tgt_label = node_map[r.target_id].label if r.target_id in node_map else r.target_id
            evidence_info = f" [Evidence: '{r.evidence_text}']" if r.evidence_text else ""
            lines.append(
                f"  • '{src_label}' --({r.relation_type}, conf={r.confidence:.2f})--> '{tgt_label}'{evidence_info}"
            )

        return "\n".join(lines)

    @staticmethod
    def export_json(graph_store: GraphStore) -> Dict[str, Any]:
        """Export graph structure as JSON-serializable dictionary.

        Args:
            graph_store: GraphStore instance.

        Returns:
            Dictionary containing 'nodes' and 'relationships'.
        """
        subgraph = graph_store.query_subgraph([], depth=5)
        return {
            "nodes": [n.model_dump() for n in subgraph.get("nodes", [])],
            "relationships": [r.model_dump() for r in subgraph.get("relationships", [])],
        }

    @staticmethod
    def export_dot(graph_store: GraphStore) -> str:
        """Export graph structure in Graphviz DOT format.

        Args:
            graph_store: GraphStore instance.

        Returns:
            Graphviz DOT syntax string.
        """
        lines: List[str] = ["digraph MindGraphDB {", '  rankdir="LR";', '  node [shape=ellipse, style=filled, fillcolor=lightcyan];']

        subgraph = graph_store.query_subgraph([], depth=5)
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        for n in nodes:
            shape = "box" if n.node_type == "DOCUMENT" else "ellipse"
            color = "lightyellow" if n.node_type == "DOCUMENT" else "lightcyan"
            lines.append(f'  "{n.id}" [label="{n.label}", shape={shape}, fillcolor={color}];')

        for r in relationships:
            label = f"{r.relation_type} ({r.confidence:.2f})"
            lines.append(f'  "{r.source_id}" -> "{r.target_id}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def export_html(
        graph_store: GraphStore,
        output_path: Optional[str] = "graph.html",
        title: str = "Mind Graph DB Interactive Visualizer",
    ) -> str:
        """Generate a self-contained, interactive web-based Vis.js graph visualization HTML document.

        Args:
            graph_store: GraphStore instance.
            output_path: Optional file path to write HTML output.
            title: Web page title header.

        Returns:
            HTML string of the visual graph application.
        """
        subgraph = graph_store.query_subgraph([], depth=5)
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        vis_nodes = []
        for n in nodes:
            is_doc = n.node_type == "DOCUMENT"
            bg_color = "#4f46e5" if is_doc else "#0284c7"
            border_color = "#818cf8" if is_doc else "#38bdf8"
            shape = "box" if is_doc else "dot"
            size = 25 if is_doc else 16

            vis_nodes.append(
                {
                    "id": n.id,
                    "label": n.label[:35] + ("..." if len(n.label) > 35 else ""),
                    "full_label": n.label,
                    "group": n.node_type,
                    "shape": shape,
                    "size": size,
                    "color": {
                        "background": bg_color,
                        "border": border_color,
                        "highlight": {"background": "#6366f1", "border": "#a5b4fc"},
                    },
                    "font": {"color": "#f8fafc", "face": "Inter, sans-serif", "size": 13},
                    "properties": n.properties,
                }
            )

        vis_edges = []
        for r in relationships:
            rel_type = r.relation_type
            color_map = {
                "MENTIONS": "#38bdf8",
                "SIMILAR_TO": "#c084fc",
                "USES_STORAGE": "#34d399",
            }
            edge_color = color_map.get(rel_type, "#94a3b8")

            vis_edges.append(
                {
                    "id": r.id,
                    "from": r.source_id,
                    "to": r.target_id,
                    "label": rel_type,
                    "title": f"Relation: {rel_type}\nConfidence: {r.confidence:.2f}\nEvidence: {r.evidence_text or 'N/A'}",
                    "confidence": r.confidence,
                    "evidence_text": r.evidence_text or "No explicit provenance logged.",
                    "relation_type": rel_type,
                    "arrows": "to",
                    "color": {"color": edge_color, "highlight": "#f43f5e"},
                    "font": {"color": "#cbd5e1", "size": 11, "align": "top"},
                    "width": max(1, int(r.confidence * 3)),
                }
            )

        json_nodes = json.dumps(vis_nodes)
        json_edges = json.dumps(vis_edges)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #090d16;
      color: #f8fafc;
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 20;
    }}
    .logo-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-icon {{
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, #6366f1, #0ea5e9);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 16px;
    }}
    h1 {{ font-size: 18px; font-weight: 600; letter-spacing: -0.02em; }}
    .badge {{
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.3);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    input, select, button {{
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #f8fafc;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      transition: all 0.2s;
    }}
    input:focus, select:focus {{
      border-color: #6366f1;
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }}
    button {{
      cursor: pointer;
      background: #4f46e5;
      font-weight: 500;
      border: none;
    }}
    button:hover {{ background: #6366f1; }}
    #main-container {{
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }}
    #mynetwork {{
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at 50% 50%, #0f172a 0%, #090d16 100%);
    }}
    #side-panel {{
      width: 380px;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(16px);
      border-left: 1px solid rgba(255, 255, 255, 0.08);
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: -8px 0 24px rgba(0, 0, 0, 0.3);
      z-index: 10;
    }}
    .panel-header {{
      font-size: 16px;
      font-weight: 600;
      color: #f8fafc;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 12px;
    }}
    .property-card {{
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .prop-label {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; }}
    .prop-val {{ font-size: 13px; color: #e2e8f0; line-height: 1.5; word-break: break-word; }}
    .evidence-box {{
      background: rgba(14, 165, 233, 0.1);
      border: 1px solid rgba(14, 165, 233, 0.3);
      border-radius: 8px;
      padding: 12px;
      color: #38bdf8;
      font-size: 13px;
      line-height: 1.4;
    }}
    .legend {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 12px 16px;
      display: flex;
      gap: 16px;
      z-index: 5;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: #cbd5e1; }}
    .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <div class="logo-icon">M</div>
      <h1>Mind Graph DB Explorer</h1>
      <span class="badge" id="node-count-badge">0 Nodes</span>
      <span class="badge" id="edge-count-badge">0 Edges</span>
    </div>
    <div class="toolbar">
      <input type="text" id="search-input" placeholder="Search node label..." onkeyup="filterNodes()" />
      <button onclick="resetZoom()">Fit View</button>
      <button onclick="togglePhysics()">Toggle Physics</button>
    </div>
  </header>

  <div id="main-container">
    <div id="mynetwork"></div>

    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4f46e5;"></div> Document Node</div>
      <div class="legend-item"><div class="legend-dot" style="background:#0284c7; border-radius:50%;"></div> Entity Node</div>
      <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> MENTIONS Edge</div>
      <div class="legend-item"><div class="legend-dot" style="background:#c084fc;"></div> SIMILAR_TO Edge</div>
    </div>

    <div id="side-panel">
      <div class="panel-header" id="panel-title">Graph Node Inspector</div>
      <div id="panel-content">
        <div class="property-card">
          <div class="prop-val" style="color: #94a3b8; text-align: center;">Click any node or edge in the visual graph to inspect details, properties, and relationship evidence provenance.</div>
        </div>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const rawNodes = {json_nodes};
    const rawEdges = {json_edges};

    document.getElementById("node-count-badge").innerText = rawNodes.length + " Nodes";
    document.getElementById("edge-count-badge").innerText = rawEdges.length + " Edges";

    const container = document.getElementById("mynetwork");
    const data = {{
      nodes: new vis.DataSet(rawNodes),
      edges: new vis.DataSet(rawEdges)
    }};

    const options = {{
      nodes: {{
        borderWidth: 2,
        shadow: true
      }},
      edges: {{
        smooth: {{ type: "continuous" }},
        shadow: true
      }},
      physics: {{
        barnesHut: {{
          gravitationalConstant: -3000,
          centralGravity: 0.3,
          springLength: 120
        }},
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100
      }}
    }};

    const network = new vis.Network(container, data, options);

    let physicsEnabled = true;
    function togglePhysics() {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
    }}

    function resetZoom() {{
      network.fit({{ animation: true }});
    }}

    function filterNodes() {{
      const query = document.getElementById("search-input").value.toLowerCase();
      if (!query) {{
        data.nodes.update(rawNodes);
        return;
      }}
      const matchedIds = rawNodes.filter(n => n.full_label.toLowerCase().includes(query)).map(n => n.id);
      if (matchedIds.length > 0) {{
        network.selectNodes(matchedIds);
      }}
    }}

    network.on("click", function (params) {{
      const content = document.getElementById("panel-content");
      const title = document.getElementById("panel-title");

      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = rawNodes.find(n => n.id === nodeId);
        if (node) {{
          title.innerText = "[" + node.group + "] Node Inspector";
          let html = `
            <div class="property-card">
              <div class="prop-label">Node ID</div>
              <div class="prop-val"><code>${{node.id}}</code></div>
            </div>
            <div class="property-card">
              <div class="prop-label">Label / Name</div>
              <div class="prop-val">${{node.full_label}}</div>
            </div>
            <div class="property-card">
              <div class="prop-label">Node Type</div>
              <div class="prop-val"><b>${{node.group}}</b></div>
            </div>
          `;
          if (node.properties && Object.keys(node.properties).length > 0) {{
            html += `
              <div class="property-card">
                <div class="prop-label">Metadata & Properties</div>
                <div class="prop-val"><pre style="font-size:12px;">${{JSON.stringify(node.properties, null, 2)}}</pre></div>
              </div>
            `;
          }}
          content.innerHTML = html;
        }}
      }} else if (params.edges.length > 0) {{
        const edgeId = params.edges[0];
        const edge = rawEdges.find(e => e.id === edgeId);
        if (edge) {{
          title.innerText = "Relationship Edge Inspector";
          const srcNode = rawNodes.find(n => n.id === edge.from);
          const tgtNode = rawNodes.find(n => n.id === edge.to);
          content.innerHTML = `
            <div class="property-card">
              <div class="prop-label">Relation Type</div>
              <div class="prop-val"><b>${{edge.relation_type}}</b></div>
            </div>
            <div class="property-card">
              <div class="prop-label">Source Node</div>
              <div class="prop-val">${{srcNode ? srcNode.full_label : edge.from}}</div>
            </div>
            <div class="property-card">
              <div class="prop-label">Target Node</div>
              <div class="prop-val">${{tgtNode ? tgtNode.full_label : edge.to}}</div>
            </div>
            <div class="property-card">
              <div class="prop-label">Confidence Score</div>
              <div class="prop-val"><b>${{(edge.confidence * 100).toFixed(1)}}%</b></div>
            </div>
            <div class="evidence-box">
              <div class="prop-label" style="color:#0284c7; margin-bottom:4px;">Evidence Provenance</div>
              "${{edge.evidence_text}}"
            </div>
          `;
        }}
      }} else {{
        title.innerText = "Graph Node Inspector";
        content.innerHTML = `
          <div class="property-card">
            <div class="prop-val" style="color: #94a3b8; text-align: center;">Click any node or edge in the visual graph to inspect details, properties, and relationship evidence provenance.</div>
          </div>
        `;
      }}
    }});
  </script>
</body>
</html>
"""

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content
