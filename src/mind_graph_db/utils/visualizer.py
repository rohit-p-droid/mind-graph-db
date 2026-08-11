"""Graph visualizer and debugging utility for Mind Graph DB."""

import json
import os
from typing import Any, Dict, List, Optional

from mind_graph_db.interfaces.graph_store import GraphStore


class GraphVisualizer:
    """Utility for rendering ASCII trees, exporting JSON/DOT, and building visual graph HTML."""

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
        title: str = "Mind Graph DB Explorer",
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
            ntype = (n.node_type or "ENTITY").upper()
            is_doc = ntype == "DOCUMENT"
            is_concept = ntype == "CONCEPT"

            if is_doc:
                icon = "📄 "
                shape = "box"
                bg_color = "#1e1b4b"
                border_color = "#6366f1"
                highlight_bg = "#312e81"
                font_color = "#e0e7ff"
                size = 26
                truncated_label = n.label[:42] + ("..." if len(n.label) > 42 else "")
            elif is_concept:
                icon = "💡 "
                shape = "ellipse"
                bg_color = "#064e3b"
                border_color = "#10b981"
                highlight_bg = "#047857"
                font_color = "#ecfdf5"
                size = 20
                truncated_label = n.label
            else:
                icon = "🏷️ "
                shape = "ellipse"
                bg_color = "#0c4a6e"
                border_color = "#0ea5e9"
                highlight_bg = "#0369a1"
                font_color = "#f0f9ff"
                size = 22
                truncated_label = n.label

            display_label = f"{icon}{truncated_label}"

            vis_nodes.append(
                {
                    "id": n.id,
                    "label": display_label,
                    "full_label": n.label,
                    "node_type": ntype,
                    "shape": shape,
                    "size": size,
                    "color": {
                        "background": bg_color,
                        "border": border_color,
                        "highlight": {"background": highlight_bg, "border": "#38bdf8"},
                        "hover": {"background": highlight_bg, "border": "#818cf8"}
                    },
                    "font": {
                        "color": font_color,
                        "face": "Inter, system-ui, sans-serif",
                        "size": 13,
                        "bold": True,
                    },
                    "properties": n.properties,
                    "margin": 12,
                }
            )

        vis_edges = []
        for r in relationships:
            rel_type = (r.relation_type or "RELATED_TO").upper()
            color_map = {
                "MENTIONS": "#38bdf8",
                "SIMILAR_TO": "#c084fc",
                "USES_STORAGE": "#34d399",
                "DEPENDS_ON": "#f59e0b",
                "WRITTEN_IN": "#ec4899",
            }
            edge_color = color_map.get(rel_type, "#94a3b8")
            conf_pct = int(r.confidence * 100)

            vis_edges.append(
                {
                    "id": r.id,
                    "from": r.source_id,
                    "to": r.target_id,
                    "label": f"{rel_type} ({conf_pct}%)",
                    "title": f"Relation: {rel_type}\nConfidence: {conf_pct}%\nEvidence: {r.evidence_text or 'N/A'}",
                    "confidence": r.confidence,
                    "evidence_text": r.evidence_text or "No explicit provenance logged.",
                    "relation_type": rel_type,
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                    "color": {"color": edge_color, "highlight": "#f43f5e", "hover": "#fb7185"},
                    "font": {
                        "color": "#cbd5e1",
                        "size": 11,
                        "align": "top",
                        "face": "Inter, sans-serif",
                        "background": "rgba(15, 23, 42, 0.75)"
                    },
                    "width": max(2, int(r.confidence * 4)),
                    "selectionWidth": 4,
                }
            )

        json_nodes = json.dumps(vis_nodes)
        json_edges = json.dumps(vis_edges)

        doc_count = sum(1 for n in nodes if n.node_type == "DOCUMENT")
        entity_count = sum(1 for n in nodes if n.node_type != "DOCUMENT")

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
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background-color: #0b0f19;
      color: #f8fafc;
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 20;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .logo-group {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .logo-icon {{
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #6366f1, #0ea5e9);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 18px;
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }}
    h1 {{ font-size: 19px; font-weight: 600; letter-spacing: -0.02em; color: #f8fafc; }}
    .stats-group {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .badge {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .badge b {{ color: #38bdf8; font-weight: 600; }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    input[type="text"] {{
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f8fafc;
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 13px;
      width: 260px;
      outline: none;
      transition: all 0.2s;
    }}
    input[type="text"]:focus {{
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
    }}
    button {{
      cursor: pointer;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #f8fafc;
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      outline: none;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    button:hover {{
      background: rgba(99, 102, 241, 0.2);
      border-color: #6366f1;
      color: #ffffff;
    }}
    #main-container {{
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }}
    #mynetwork {{
      flex: 1;
      height: 100%;
      background: radial-gradient(circle at 50% 50%, #0f172a 0%, #060911 100%);
    }}
    #side-panel {{
      width: 420px;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(24px);
      border-left: 1px solid rgba(255, 255, 255, 0.08);
      padding: 28px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
      box-shadow: -10px 0 30px rgba(0, 0, 0, 0.4);
      z-index: 10;
    }}
    .panel-header {{
      font-size: 17px;
      font-weight: 600;
      color: #f8fafc;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .card {{
      background: rgba(30, 41, 59, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .card-label {{
      font-size: 11px;
      text-transform: uppercase;
      color: #94a3b8;
      font-weight: 600;
      letter-spacing: 0.06em;
    }}
    .card-val {{
      font-size: 14px;
      color: #f1f5f9;
      line-height: 1.5;
      word-break: break-word;
    }}
    .type-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.04em;
    }}
    .badge-doc {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }}
    .badge-ent {{ background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }}
    .badge-cnc {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .evidence-box {{
      background: rgba(14, 165, 233, 0.08);
      border: 1px solid rgba(14, 165, 233, 0.25);
      border-radius: 10px;
      padding: 14px;
      color: #38bdf8;
      font-size: 13px;
      line-height: 1.5;
      font-style: italic;
    }}
    .confidence-bar-bg {{
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 4px;
    }}
    .confidence-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #0ea5e9, #6366f1);
      border-radius: 4px;
    }}
    .legend {{
      position: absolute;
      bottom: 24px;
      left: 24px;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      gap: 20px;
      z-index: 5;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #cbd5e1;
      cursor: pointer;
      user-select: none;
      transition: opacity 0.2s;
    }}
    .legend-item:hover {{ opacity: 0.8; }}
    .legend-dot {{ width: 14px; height: 14px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <div class="logo-group">
      <div class="logo-icon">M</div>
      <h1>Mind Graph DB Explorer</h1>
      <div class="stats-group">
        <div class="badge">📄 <b>{doc_count}</b> Documents</div>
        <div class="badge">🏷️ <b>{entity_count}</b> Entities</div>
        <div class="badge">🔗 <b>{len(relationships)}</b> Edges</div>
      </div>
    </div>

    <div class="toolbar">
      <input type="text" id="search-input" placeholder="Search node label..." onkeyup="filterNodes()" />
      <button onclick="resetZoom()">🔍 Fit View</button>
      <button onclick="togglePhysics()">⚡ Toggle Physics</button>
    </div>
  </header>

  <div id="main-container">
    <div id="mynetwork"></div>

    <div class="legend">
      <div class="legend-item" onclick="filterGroup('DOCUMENT')">
        <div class="legend-dot" style="background:#6366f1;"></div> 📄 Document Node
      </div>
      <div class="legend-item" onclick="filterGroup('ENTITY')">
        <div class="legend-dot" style="background:#0ea5e9; border-radius:50%;"></div> 🏷️ Entity Node
      </div>
      <div class="legend-item" onclick="filterGroup('CONCEPT')">
        <div class="legend-dot" style="background:#10b981; border-radius:50%;"></div> 💡 Concept Node
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#38bdf8;"></div> MENTIONS Edge
      </div>
    </div>

    <div id="side-panel">
      <div class="panel-header" id="panel-title">Graph Node Inspector</div>
      <div id="panel-content">
        <div class="card">
          <div class="card-val" style="color: #94a3b8; text-align: center; padding: 20px 0;">
            Click any node or relationship edge in the graph canvas to inspect full text, properties, and evidence provenance.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const rawNodes = {json_nodes};
    const rawEdges = {json_edges};

    const container = document.getElementById("mynetwork");
    const data = {{
      nodes: new vis.DataSet(rawNodes),
      edges: new vis.DataSet(rawEdges)
    }};

    const options = {{
      nodes: {{
        borderWidth: 2,
        shadow: {{ enabled: true, color: "rgba(0,0,0,0.5)", size: 10, x: 0, y: 4 }}
      }},
      edges: {{
        smooth: {{ type: "continuous", roundness: 0.2 }},
        shadow: {{ enabled: true, color: "rgba(0,0,0,0.3)", size: 6, x: 0, y: 2 }}
      }},
      physics: {{
        solver: "barnesHut",
        barnesHut: {{
          gravitationalConstant: -4000,
          centralGravity: 0.25,
          springLength: 160,
          springConstant: 0.04,
          damping: 0.09
        }},
        stabilization: {{ iterations: 200 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragNodes: true
      }}
    }};

    const network = new vis.Network(container, data, options);

    let physicsEnabled = true;
    function togglePhysics() {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
    }}

    function resetZoom() {{
      network.fit({{ animation: {{ duration: 500, easingFunction: "easeInOutQuad" }} }});
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

    let activeFilter = null;
    function filterGroup(groupType) {{
      if (activeFilter === groupType) {{
        activeFilter = null;
        data.nodes.update(rawNodes);
      }} else {{
        activeFilter = groupType;
        const filtered = rawNodes.filter(n => n.node_type === groupType);
        data.nodes.clear();
        data.nodes.add(filtered);
      }}
    }}

    network.on("click", function (params) {{
      const content = document.getElementById("panel-content");
      const title = document.getElementById("panel-title");

      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = rawNodes.find(n => n.id === nodeId);
        if (node) {{
          let badgeClass = node.node_type === "DOCUMENT" ? "badge-doc" : (node.node_type === "CONCEPT" ? "badge-cnc" : "badge-ent");
          let icon = node.node_type === "DOCUMENT" ? "📄" : (node.node_type === "CONCEPT" ? "💡" : "🏷️");
          
          title.innerHTML = `${{icon}} Node Details`;
          
          let html = `
            <div class="card">
              <div class="card-label">Node Type</div>
              <div><span class="type-badge ${{badgeClass}}">${{node.node_type}}</span></div>
            </div>
            <div class="card">
              <div class="card-label">Canonical Label / Text</div>
              <div class="card-val" style="font-weight:600; color:#ffffff;">${{node.full_label}}</div>
            </div>
            <div class="card">
              <div class="card-label">Node Identifier</div>
              <div class="card-val"><code style="color:#38bdf8;">${{node.id}}</code></div>
            </div>
          `;

          if (node.properties && Object.keys(node.properties).length > 0) {{
            html += `
              <div class="card">
                <div class="card-label">Properties & Metadata</div>
                <div class="card-val"><pre style="font-size:12px; color:#cbd5e1;">${{JSON.stringify(node.properties, null, 2)}}</pre></div>
              </div>
            `;
          }}

          // Find connected edges
          const connectedEdges = rawEdges.filter(e => e.from === node.id || e.to === node.id);
          if (connectedEdges.length > 0) {{
            html += `
              <div class="card">
                <div class="card-label">Connected Relationships (${{connectedEdges.length}})</div>
                <div style="display:flex; flex-direction:column; gap:6px; margin-top:4px;">
            `;
            for (const edge of connectedEdges) {{
              const otherId = edge.from === node.id ? edge.to : edge.from;
              const otherNode = rawNodes.find(n => n.id === otherId);
              const targetLabel = otherNode ? otherNode.full_label : otherId;
              const dirText = edge.from === node.id ? "➡️ Outgoing" : "⬅️ Incoming";
              html += `
                <div style="font-size:12px; background:rgba(255,255,255,0.03); padding:8px; border-radius:6px;">
                  <b>${{dirText}}</b>: <span style="color:#38bdf8;">${{edge.relation_type}}</span> to <b>${{targetLabel}}</b>
                </div>
              `;
            }}
            html += `</div></div>`;
          }}

          content.innerHTML = html;
        }}
      }} else if (params.edges.length > 0) {{
        const edgeId = params.edges[0];
        const edge = rawEdges.find(e => e.id === edgeId);
        if (edge) {{
          title.innerHTML = "🔗 Relationship Edge Inspector";
          const srcNode = rawNodes.find(n => n.id === edge.from);
          const tgtNode = rawNodes.find(n => n.id === edge.to);
          const confPct = Math.round(edge.confidence * 100);

          content.innerHTML = `
            <div class="card">
              <div class="card-label">Relation Type</div>
              <div class="card-val"><b style="color:#38bdf8; font-size:16px;">${{edge.relation_type}}</b></div>
            </div>
            <div class="card">
              <div class="card-label">Source Node</div>
              <div class="card-val"><b>${{srcNode ? srcNode.full_label : edge.from}}</b></div>
            </div>
            <div class="card">
              <div class="card-label">Target Node</div>
              <div class="card-val"><b>${{tgtNode ? tgtNode.full_label : edge.to}}</b></div>
            </div>
            <div class="card">
              <div class="card-label">Confidence Rating: ${{confPct}}%</div>
              <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: ${{confPct}}%;"></div>
              </div>
            </div>
            <div class="evidence-box">
              <div class="card-label" style="color:#0ea5e9; margin-bottom:6px;">Evidence Provenance</div>
              "${{edge.evidence_text}}"
            </div>
          `;
        }}
      }} else {{
        title.innerText = "Graph Node Inspector";
        content.innerHTML = `
          <div class="card">
            <div class="card-val" style="color: #94a3b8; text-align: center; padding: 20px 0;">
              Click any node or relationship edge in the graph canvas to inspect full text, properties, and evidence provenance.
            </div>
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
