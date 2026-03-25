import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class Knowledge3DVisualizer:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def create_3d_knowledge_graph(self, width=1200, height=800):
        """Create clear, visible 3D knowledge graph for black backgrounds"""
        try:
            graph_data = self.kb.get_ontology_graph()

            # Create 3D positions using spring layout with better spacing
            G = nx.Graph()
            for node in graph_data["nodes"]:
                G.add_node(node["id"], **node)
            for link in graph_data["links"]:
                G.add_edge(link["source"], link["target"], weight=link.get("value", 1))

            # Generate 3D positions with much better spacing
            pos = nx.spring_layout(G, dim=3, k=10, iterations=150, seed=42)

            # Extract node positions and properties
            node_x, node_y, node_z = [], [], []
            node_text, node_size, node_color = [], [], []
            node_types = []

            for node in G.nodes():
                x, y, z = pos[node]
                # Scale up positions for spreading
                node_x.append(x * 2.0)
                node_y.append(y * 2.0)
                node_z.append(z * 2.0)

                node_data = G.nodes[node]
                node_text.append(node_data.get("name", node))
                node_size.append(node_data.get("size", 18)) # Slightly smaller markers
                node_color.append(node_data.get("color", "#FF6B6B"))
                node_types.append(node_data.get("type", "unknown"))

            # Create node trace with balanced labels
            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers+text", # Restored text for usability
                marker=dict(
                    size=node_size,
                    color=node_color,
                    opacity=0.9,
                    line=dict(width=1.5, color="white"),
                    sizemode="diameter",
                ),
                text=node_text,
                textposition="top center",
                hoverinfo="text",
                name="Knowledge Nodes",
                hovertemplate="<b>%{text}</b><br>Type: %{customdata}<extra></extra>",
                customdata=node_types,
                textfont=dict(
                    size=10, # Smaller text to avoid clutter
                    color="white",
                    family="Arial",
                ),
            )

            # Create edge traces with matching scale
            edge_traces = []
            for edge in G.edges():
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]

                edge_trace = go.Scatter3d(
                    x=[x0 * 2.0, x1 * 2.0, None], # Matched node scale
                    y=[y0 * 2.0, y1 * 2.0, None],
                    z=[z0 * 2.0, z1 * 2.0, None],
                    mode="lines",
                    line=dict(
                        width=G.edges[edge].get("weight", 1) * 2 + 1,
                        color="rgba(200, 200, 200, 0.5)", # Slightly dimmer to de-clutter
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
                edge_traces.append(edge_trace)

            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace])

            # Enhanced layout for black backgrounds
            fig.update_layout(
                title=dict(
                    text="🌱 3D KNOWLEDGE GRAPH: PLANT BREEDING ONTOLOGY",
                    x=0.5,
                    font=dict(size=24, color="white", family="Arial Black"),
                    y=0.95,
                ),
                scene=dict(
                    xaxis=dict(
                        showbackground=False,
                        showticklabels=False,
                        title="",
                        gridcolor="rgba(255, 255, 255, 0.1)", # Very subtle grid
                        zerolinecolor="rgba(255, 255, 255, 0.1)",
                    ),
                    yaxis=dict(
                        showbackground=False,
                        showticklabels=False,
                        title="",
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)",
                    ),
                    zaxis=dict(
                        showbackground=False,
                        showticklabels=False,
                        title="",
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)",
                    ),
                    bgcolor="rgba(0, 0, 0, 0.8)",  # Dark but visible background
                    camera=dict(
                        eye=dict(x=1.8, y=1.8, z=1.8),  # Better viewing angle
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                    ),
                ),
                margin=dict(l=0, r=0, b=0, t=80),
                paper_bgcolor="rgba(0, 0, 0, 1)",  # Pure black paper
                plot_bgcolor="rgba(0, 0, 0, 1)",  # Pure black plot
                font=dict(color="white", size=12, family="Arial"),
                width=width,
                height=height,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(0, 0, 0, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(size=12, color="white"),
                ),
            )

            # Add enhanced legend
            self._add_enhanced_legend(fig, node_types)

            logger.info("✅ Enhanced 3D Knowledge Graph created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating enhanced 3D knowledge graph: {e}")
            return self._create_error_figure()

    def _add_enhanced_legend(self, fig, node_types):
        """Add enhanced legend with better visibility"""
        type_colors = {
            "trait_category": "#FF6B6B",  # Bright Red
            "trait": "#4ECDC4",  # Bright Teal
            "gene": "#45B7D1",  # Bright Blue
            "strategy": "#96CEB4",  # Bright Green
            "condition": "#FFD166",  # Bright Yellow
            "recommendation": "#06D6A0",  # Bright Green
            "unknown": "#999999",  # Medium Gray
        }

        # Add invisible traces for legend with larger markers
        for node_type, color in type_colors.items():
            if node_type in [t.lower() for t in node_types]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[None],
                        y=[None],
                        z=[None],
                        mode="markers",
                        marker=dict(
                            size=15, color=color, line=dict(width=2, color="white")
                        ),
                        name=node_type.replace("_", " ").title(),
                        showlegend=True,
                        legendgroup=node_type,
                    )
                )

    def create_3d_rule_network(self, recommendations, width=1200, height=700):
        """Create clear 3D visualization of rule inference network"""
        try:
            if not recommendations:
                return self._create_empty_figure("No recommendations available")

            # Create nodes with better spacing
            nodes = []
            node_positions = {}

            for i, rec in enumerate(recommendations):
                # Condition node
                cond_id = f"cond_{i}"
                nodes.append(
                    {
                        "id": cond_id,
                        "name": rec.get("reasoning", "Condition"),
                        "type": "condition",
                        "confidence": rec["confidence"],
                    }
                )
                node_positions[cond_id] = [i * 4, 0, rec["confidence"]]

                # Recommendation node
                rec_id = f"rec_{i}"
                nodes.append(
                    {
                        "id": rec_id,
                        "name": rec["recommendation"][:80]
                        + ("..." if len(rec["recommendation"]) > 80 else ""),
                        "type": "recommendation",
                        "confidence": rec["confidence"],
                    }
                )
                node_positions[rec_id] = [i * 4, 3, rec["confidence"]]

            # Create 3D scatter plot with enhanced visibility
            node_x = [pos[0] for pos in node_positions.values()]
            node_y = [pos[1] for pos in node_positions.values()]
            node_z = [pos[2] for pos in node_positions.values()]

            node_colors = []
            node_sizes = []
            node_text = []
            node_customdata = []

            for node in nodes:
                if node["type"] == "condition":
                    node_colors.append("#FFD166")  # Bright Yellow
                    node_sizes.append(25)
                else:
                    node_colors.append("#06D6A0")  # Bright Green
                    node_sizes.append(30)
                node_text.append(node["name"])
                node_customdata.append(f"Confidence: {node['confidence']:.2f}")

            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode="markers+text", # Restored text
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.9,
                    line=dict(width=1.5, color="white"),
                ),
                text=node_text,
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
                customdata=node_customdata,
                name="Rule Nodes",
                textfont=dict(size=9, color="rgba(255,255,255,0.8)"), # Subdued text
            )

            # Create edges with better visibility
            edge_x, edge_y, edge_z = [], [], []
            for i in range(len(recommendations)):
                cond_id = f"cond_{i}"
                rec_id = f"rec_{i}"

                if cond_id in node_positions and rec_id in node_positions:
                    cond_pos = node_positions[cond_id]
                    rec_pos = node_positions[rec_id]

                    edge_x.extend([cond_pos[0], rec_pos[0], None])
                    edge_y.extend([cond_pos[1], rec_pos[1], None])
                    edge_z.extend([cond_pos[2], rec_pos[2], None])

            edge_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(
                    width=6, color="rgba(255, 255, 255, 0.8)"
                ),  # Thicker, brighter
                hoverinfo="none",
                name="Inference Path",
            )

            fig = go.Figure(data=[edge_trace, node_trace])

            # Enhanced layout
            fig.update_layout(
                title=dict(
                    text="🧠 3D RULE INFERENCE NETWORK",
                    x=0.5,
                    font=dict(size=22, color="white", family="Arial Black"),
                    y=0.95,
                ),
                scene=dict(
                    xaxis_title="Rule Index",
                    yaxis_title="Inference Level",
                    zaxis_title="Confidence Score",
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                    ),
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                    zaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                ),
                margin=dict(l=0, r=0, b=0, t=80),
                paper_bgcolor="rgba(0, 0, 0, 1)",
                plot_bgcolor="rgba(0, 0, 0, 1)",
                font=dict(color="white", family="Arial"),
                width=width,
                height=height,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(0, 0, 0, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                ),
            )

            logger.info("✅ Enhanced 3D Rule Network created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating enhanced 3D rule network: {e}")
            return self._create_error_figure()

    def create_trait_correlation_3d(self, df, traits=None, width=1200, height=700):
        """Create clear 3D scatter plot of trait correlations"""
        try:
            if traits is None:
                traits = ["Yield_per_plant", "Height", "Grain_weight"]

            # Ensure traits exist in dataframe
            available_traits = [trait for trait in traits if trait in df.columns]
            if len(available_traits) < 3:
                logger.warning(
                    f"Not enough traits available. Needed: 3, Found: {available_traits}"
                )
                return self._create_empty_figure("Not enough trait data available")

            x_trait, y_trait, z_trait = available_traits[:3]

            # Create color scale based on available data
            color_column = None
            if "Drought_Tolerance" in df.columns:
                color_column = "Drought_Tolerance"
            elif "Variety" in df.columns:
                color_column = "Variety"
            else:
                color_column = x_trait

            fig = px.scatter_3d(
                df,
                x=x_trait,
                y=y_trait,
                z=z_trait,
                color=color_column,
                hover_name="Variety" if "Variety" in df.columns else None,
                title=f"3D TRAIT CORRELATION: {x_trait} vs {y_trait} vs {z_trait}",
                opacity=0.8,
                size_max=18,
            )

            # Enhanced styling for black background
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=2, color="white"),  # White borders for visibility
                    opacity=0.8,
                ),
                selector=dict(mode="markers"),
            )

            fig.update_layout(
                title=dict(
                    x=0.5,
                    font=dict(size=20, color="white", family="Arial Black"),
                    y=0.95,
                ),
                paper_bgcolor="rgba(0, 0, 0, 1)",
                plot_bgcolor="rgba(0, 0, 0, 1)",
                font=dict(color="white", family="Arial"),
                scene=dict(
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                    zaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.3)",
                        gridwidth=2,
                        title_font=dict(color="white", size=14),
                    ),
                ),
                margin=dict(l=0, r=0, b=0, t=80),
                width=width,
                height=height,
                legend=dict(
                    bgcolor="rgba(0, 0, 0, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(color="white"),
                ),
            )

            logger.info("✅ Enhanced 3D Trait Correlation plot created successfully")
            return fig

        except Exception as e:
            logger.error(f"Error creating enhanced 3D trait correlation: {e}")
            return self._create_error_figure()

    def _create_empty_figure(self, message):
        """Create an empty figure with message"""
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=message, x=0.5, y=0.5, font=dict(color="white", size=16)),
            paper_bgcolor="rgba(0, 0, 0, 1)",
            plot_bgcolor="rgba(0, 0, 0, 1)",
            font=dict(color="white"),
            width=400,
            height=200,
        )
        return fig

    def _create_error_figure(self):
        """Create an error figure"""
        return self._create_empty_figure("Error creating visualization")
