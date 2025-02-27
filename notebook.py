import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json
import uuid
from IPython.display import display, HTML

class MathReasoningGraph:
    def __init__(self, title="Math Reasoning Graph"):
        self.title = title
        self.graph = nx.DiGraph()
        self.node_types = {
            "theorem": {"color": "#FFD700", "shape": "ellipse"},      # Gold
            "axiom": {"color": "#FF8C00", "shape": "box"},            # Dark Orange
            "fact": {"color": "#20B2AA", "shape": "box"},             # Light Sea Green
            "observation": {"color": "#87CEFA", "shape": "ellipse"},  # Light Sky Blue
            "question": {"color": "#BA55D3", "shape": "diamond"},     # Medium Orchid
            "insight": {"color": "#3CB371", "shape": "ellipse"},      # Medium Sea Green
            "conclusion": {"color": "#FF6347", "shape": "box"},       # Tomato
            "hypothesis": {"color": "#9370DB", "shape": "ellipse"},   # Medium Purple
            "step": {"color": "#A9A9A9", "shape": "ellipse"},         # Dark Gray
            "property": {"color": "#6495ED", "shape": "box"},         # Cornflower Blue
            "definition": {"color": "#2E8B57", "shape": "box"},       # Sea Green
            "example": {"color": "#DAA520", "shape": "box"},          # Goldenrod
            "counterexample": {"color": "#DC143C", "shape": "box"},   # Crimson
        }
        
        self.edge_types = {
            "implies": {"color": "black", "style": "solid", "weight": 1.0},
            "contradicts": {"color": "red", "style": "dashed", "weight": 1.0},
            "supports": {"color": "green", "style": "solid", "weight": 1.0},
            "follows_from": {"color": "blue", "style": "solid", "weight": 1.0},
            "part_of": {"color": "purple", "style": "dotted", "weight": 1.0},
            "example_of": {"color": "orange", "style": "dotted", "weight": 0.5},
            "defined_as": {"color": "brown", "style": "solid", "weight": 1.0},
            "specializes": {"color": "teal", "style": "solid", "weight": 0.75},
            "generalizes": {"color": "olive", "style": "solid", "weight": 0.75},
            "leads_to": {"color": "grey", "style": "solid", "weight": 1.0},
        }
        
        self.reasoning_steps = []
        self.conclusions = []
        
        # Default positions for nodes
        self.pos = None
        
    def add_node(self, content, node_type="observation", node_id=None, location=None):
        """
        Add a node to the reasoning graph.
        
        Args:
            content (str): The content of the node
            node_type (str): Type of node (theorem, fact, observation, etc.)
            node_id (str, optional): Custom ID for the node. If None, a UUID will be generated.
            location (tuple, optional): (x, y) coordinates for the node. If None, it will be positioned automatically.
        
        Returns:
            str: The ID of the newly created node
        """
        if node_id is None:
            node_id = str(uuid.uuid4())[:8]
            
        if node_type not in self.node_types:
            raise ValueError(f"Node type {node_type} not recognized. Available types: {list(self.node_types.keys())}")
            
        self.graph.add_node(node_id, 
                           content=content, 
                           type=node_type, 
                           color=self.node_types[node_type]["color"],
                           shape=self.node_types[node_type]["shape"])
        
        # Store fixed position if provided
        if location is not None:
            if self.pos is None:
                self.pos = {}
            self.pos[node_id] = location
        
        self.reasoning_steps.append(f"Added {node_type} node: {content}")
        
        if node_type == "conclusion":
            self.conclusions.append({
                "id": node_id,
                "content": content
            })
        
        return node_id
    
    def add_edge(self, source_id, target_id, edge_type="implies", label=""):
        """
        Add a directed edge between two nodes.
        
        Args:
            source_id (str): The ID of the source node
            target_id (str): The ID of the target node
            edge_type (str): Type of edge (implies, contradicts, supports, etc.)
            label (str, optional): Optional label for the edge
            
        Returns:
            tuple: (source_id, target_id)
        """
        if source_id not in self.graph.nodes:
            raise ValueError(f"Source node {source_id} does not exist in the graph")
            
        if target_id not in self.graph.nodes:
            raise ValueError(f"Target node {target_id} does not exist in the graph")
            
        if edge_type not in self.edge_types:
            raise ValueError(f"Edge type {edge_type} not recognized. Available types: {list(self.edge_types.keys())}")
        
        self.graph.add_edge(source_id, 
                           target_id, 
                           type=edge_type,
                           color=self.edge_types[edge_type]["color"],
                           style=self.edge_types[edge_type]["style"],
                           weight=self.edge_types[edge_type]["weight"],
                           label=label)
        
        source_content = self.graph.nodes[source_id]["content"]
        target_content = self.graph.nodes[target_id]["content"]
        
        self.reasoning_steps.append(f"Connected '{source_content}' to '{target_content}' with {edge_type} relationship")
        
        return (source_id, target_id)
    
    def add_reasoning_step(self, text):
        """Add a reasoning step without modifying the graph structure."""
        self.reasoning_steps.append(text)
    
    def get_node_neighbors(self, node_id):
        """Get all immediate neighbors of a node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")
            
        successors = list(self.graph.successors(node_id))
        predecessors = list(self.graph.predecessors(node_id))
        
        return {
            "outgoing": [{"id": succ, 
                         "content": self.graph.nodes[succ]["content"], 
                         "type": self.graph.nodes[succ]["type"],
                         "edge_type": self.graph.edges[node_id, succ]["type"]} 
                        for succ in successors],
            "incoming": [{"id": pred, 
                         "content": self.graph.nodes[pred]["content"], 
                         "type": self.graph.nodes[pred]["type"],
                         "edge_type": self.graph.edges[pred, node_id]["type"]} 
                        for pred in predecessors]
        }
    
    def find_nodes_by_content(self, search_term):
        """Find nodes whose content contains the search term."""
        results = []
        for node_id in self.graph.nodes:
            content = self.graph.nodes[node_id]["content"]
            if search_term.lower() in content.lower():
                results.append({
                    "id": node_id,
                    "content": content,
                    "type": self.graph.nodes[node_id]["type"]
                })
        return results
    
    def find_paths(self, start_id, end_id):
        """Find all paths between two nodes."""
        if start_id not in self.graph.nodes:
            raise ValueError(f"Start node {start_id} does not exist in the graph")
            
        if end_id not in self.graph.nodes:
            raise ValueError(f"End node {end_id} does not exist in the graph")
            
        try:
            paths = list(nx.all_simple_paths(self.graph, start_id, end_id))
            
            formatted_paths = []
            for path in paths:
                path_steps = []
                for i in range(len(path)):
                    node_id = path[i]
                    node_info = {
                        "id": node_id,
                        "content": self.graph.nodes[node_id]["content"],
                        "type": self.graph.nodes[node_id]["type"]
                    }
                    
                    # Add edge info if not the last node
                    if i < len(path) - 1:
                        next_id = path[i + 1]
                        edge_info = {
                            "type": self.graph.edges[node_id, next_id]["type"],
                            "label": self.graph.edges[node_id, next_id].get("label", "")
                        }
                        node_info["edge"] = edge_info
                        
                    path_steps.append(node_info)
                    
                formatted_paths.append(path_steps)
                
            return formatted_paths
        except nx.NetworkXNoPath:
            return []
    
    def get_sources_and_sinks(self):
        """Identify source nodes (no incoming edges) and sink nodes (no outgoing edges)."""
        sources = []
        sinks = []
        
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                sources.append({
                    "id": node,
                    "content": self.graph.nodes[node]["content"],
                    "type": self.graph.nodes[node]["type"]
                })
                
            if self.graph.out_degree(node) == 0:
                sinks.append({
                    "id": node,
                    "content": self.graph.nodes[node]["content"],
                    "type": self.graph.nodes[node]["type"]
                })
                
        return {"sources": sources, "sinks": sinks}
    
    def plot(self, figsize=(12, 10), node_size=2000, font_size=10, edge_label_font_size=8, 
             show_edge_labels=True, layout="spring"):
        """
        Plot the reasoning graph.
        
        Args:
            figsize (tuple): Figure size
            node_size (int): Size of nodes
            font_size (int): Font size for node labels
            edge_label_font_size (int): Font size for edge labels
            show_edge_labels (bool): Whether to show edge labels
            layout (str): Type of layout algorithm to use ('spring', 'circular', 'kamada_kawai', 'planar')
        """
        plt.figure(figsize=figsize)
        
        # If we have fixed positions, use them; otherwise compute layout
        if self.pos is None or len(self.pos) != len(self.graph.nodes):
            if layout == "spring":
                self.pos = nx.spring_layout(self.graph, seed=42)
            elif layout == "circular":
                self.pos = nx.circular_layout(self.graph)
            elif layout == "kamada_kawai":
                self.pos = nx.kamada_kawai_layout(self.graph)
            elif layout == "planar":
                try:
                    self.pos = nx.planar_layout(self.graph)
                except nx.NetworkXException:
                    self.pos = nx.spring_layout(self.graph, seed=42)
            else:
                self.pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes with their specific colors and shapes
        for node_type in self.node_types:
            node_list = [node for node in self.graph.nodes 
                        if self.graph.nodes[node]["type"] == node_type]
            
            if not node_list:
                continue
                
            color = self.node_types[node_type]["color"]
            shape = self.node_types[node_type]["shape"]
            
            if shape == "ellipse":
                nx.draw_networkx_nodes(self.graph, self.pos, nodelist=node_list, 
                                      node_color=color, node_size=node_size, 
                                      node_shape='o', alpha=0.8)
            elif shape == "box":
                nx.draw_networkx_nodes(self.graph, self.pos, nodelist=node_list, 
                                      node_color=color, node_size=node_size, 
                                      node_shape='s', alpha=0.8)
            elif shape == "diamond":
                nx.draw_networkx_nodes(self.graph, self.pos, nodelist=node_list, 
                                      node_color=color, node_size=node_size, 
                                      node_shape='d', alpha=0.8)
        
        # Draw edges with their specific styles and colors
        for edge_type in self.edge_types:
            edge_list = [(u, v) for u, v, data in self.graph.edges(data=True) 
                        if data["type"] == edge_type]
            
            if not edge_list:
                continue
                
            color = self.edge_types[edge_type]["color"]
            style = self.edge_types[edge_type]["style"]
            width = self.edge_types[edge_type]["weight"] * 2  # Scale for visibility
            
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=edge_list, 
                                 width=width, edge_color=color, style=style, 
                                 arrowsize=20, alpha=0.7)
        
        # Draw node labels
        labels = {node: self.graph.nodes[node]["content"] 
                for node in self.graph.nodes}
        wrapped_labels = {node: '\n'.join(textwrap.wrap(text, width=20)) 
                         for node, text in labels.items()}
        nx.draw_networkx_labels(self.graph, self.pos, labels=wrapped_labels, 
                               font_size=font_size, font_family='sans-serif', 
                               font_weight='bold')
        
        # Draw edge labels if requested
        if show_edge_labels:
            edge_labels = {(u, v): data.get("label", "") if data.get("label") else data["type"] 
                          for u, v, data in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, 
                                       font_size=edge_label_font_size)
        
        plt.title(self.title, size=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_reasoning_steps(self):
        """Print all reasoning steps."""
        print("\n=== Reasoning Steps ===")
        for i, step in enumerate(self.reasoning_steps, 1):
            print(f"Step {i}: {step}")
            
    def print_conclusions(self):
        """Print all conclusions."""
        if not self.conclusions:
            print("\nNo conclusions recorded yet.")
            return
            
        print("\n=== Conclusions ===")
        for i, conclusion in enumerate(self.conclusions, 1):
            print(f"Conclusion {i}: {conclusion['content']}")
    
    def export_to_json(self, filename="reasoning_graph.json"):
        """Export the graph to a JSON file."""
        data = {
            "title": self.title,
            "nodes": [],
            "edges": [],
            "reasoning_steps": self.reasoning_steps,
            "conclusions": self.conclusions
        }
        
        # Export nodes
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            position = self.pos[node_id] if self.pos and node_id in self.pos else [0, 0]
            
            data["nodes"].append({
                "id": node_id,
                "content": node_data["content"],
                "type": node_data["type"],
                "color": node_data["color"],
                "shape": node_data["shape"],
                "position": {"x": float(position[0]), "y": float(position[1])}
            })
            
        # Export edges
        for source, target, edge_data in self.graph.edges(data=True):
            data["edges"].append({
                "source": source,
                "target": target,
                "type": edge_data["type"],
                "color": edge_data["color"],
                "style": edge_data["style"],
                "weight": edge_data["weight"],
                "label": edge_data.get("label", "")
            })
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filename
    
    def import_from_json(self, filename):
        """Import a graph from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Create a new graph
        self.graph = nx.DiGraph()
        self.title = data["title"]
        self.reasoning_steps = data["reasoning_steps"]
        self.conclusions = data.get("conclusions", [])
        self.pos = {}
        
        # Import nodes
        for node in data["nodes"]:
            self.graph.add_node(node["id"], 
                               content=node["content"], 
                               type=node["type"],
                               color=node["color"],
                               shape=node["shape"])
            
            if "position" in node:
                self.pos[node["id"]] = (node["position"]["x"], node["position"]["y"])
            
        # Import edges
        for edge in data["edges"]:
            self.graph.add_edge(edge["source"], 
                               edge["target"], 
                               type=edge["type"],
                               color=edge["color"],
                               style=edge["style"],
                               weight=edge["weight"],
                               label=edge.get("label", ""))
            
        return self
    
    def generate_subgraph(self, node_ids):
        """Create a subgraph containing only the specified nodes and the edges between them."""
        subgraph = self.graph.subgraph(node_ids).copy()
        
        # Create a new MathReasoningGraph with the subgraph
        subgraph_tool = MathReasoningGraph(title=f"Subgraph of {self.title}")
        subgraph_tool.graph = subgraph
        
        # Copy node positions for the nodes in the subgraph
        if self.pos:
            subgraph_tool.pos = {node: self.pos[node] for node in node_ids if node in self.pos}
            
        return subgraph_tool
    
    def suggest_next_steps(self):
        """Suggest possible next steps in the reasoning process."""
        suggestions = []
        
        # Get sources and sinks
        sources_and_sinks = self.get_sources_and_sinks()
        
        # Suggest exploring sinks (endpoints of reasoning)
        for sink in sources_and_sinks["sinks"]:
            if sink["type"] != "conclusion":
                suggestions.append(f"Consider drawing conclusions from: '{sink['content']}'")
                
        # Suggest connecting disconnected components
        components = list(nx.weakly_connected_components(self.graph))
        if len(components) > 1:
            suggestions.append(f"There are {len(components)} disconnected reasoning paths. Consider finding relationships between them.")
            
        # Suggest exploring contradictions
        contradictions = [(u, v) for u, v, data in self.graph.edges(data=True) 
                         if data["type"] == "contradicts"]
        if contradictions:
            suggestions.append(f"Resolve the contradictions in your reasoning (found {len(contradictions)}).")
            
        # Suggest areas with few connections
        for node in self.graph.nodes:
            degree = self.graph.degree(node)
            if degree <= 1 and self.graph.nodes[node]["type"] not in ["conclusion", "axiom"]:
                suggestions.append(f"Explore more connections for: '{self.graph.nodes[node]['content']}'")
                
        return suggestions
    
    def display_html(self):
        """Generate an HTML representation of the graph for display in notebook."""
        html = f"""
        <div style="border:1px solid #ddd; padding:15px; border-radius:5px; margin:10px 0;">
            <h2 style="text-align:center; color:#333;">{self.title}</h2>
            <div style="margin:15px 0;">
                <h3>Nodes:</h3>
                <ul>
        """
        
        # Add nodes
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            html += f"""
                <li style="margin-bottom:8px;">
                    <span style="display:inline-block; width:15px; height:15px; background-color:{node['color']}; 
                    border-radius:{5 if node['shape']=='ellipse' else 0}px; margin-right:5px;"></span>
                    <strong>{node_id}</strong>: {node['content']} <em>({node['type']})</em>
                </li>
            """
            
        html += """
                </ul>
            </div>
            <div>
                <h3>Relationships:</h3>
                <ul>
        """
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            source_content = self.graph.nodes[u]["content"]
            target_content = self.graph.nodes[v]["content"]
            edge_label = data.get("label", "")
            html += f"""
                <li style="margin-bottom:8px;">
                    <span style="color:{data['color']}; font-weight:bold;">{u} → {v}</span>: 
                    "{source_content}" <span style="color:{data['color']}; font-style:italic;">{data['type']}</span> "{target_content}"
                    {f'<br><em>Note: {edge_label}</em>' if edge_label else ''}
                </li>
            """
            
        html += """
                </ul>
            </div>
        </div>
        """
        
        return HTML(html)


# Additional utility: Rich text formatting for mathematical notation
import textwrap

def format_math_text(text):
    """Format text with LaTeX math expressions for display."""
    # Replace $...$ with LaTeX math formatting
    import re
    text = re.sub(r'\$(.+?)\$', r'$\1$', text)
    return text

# Example usage
def demo_math_reasoning():
    # Create a reasoning graph for a simple math problem
    reasoning = MathReasoningGraph("Pythagorean Theorem Proof")
    
    # Start with basic definitions and axioms
    axiom1 = reasoning.add_node("In a right triangle, one angle is 90 degrees", node_type="axiom")
    
    # Add some relevant theorems
    theorem1 = reasoning.add_node("The sum of angles in a triangle is 180 degrees", node_type="theorem")
    theorem2 = reasoning.add_node("Similar triangles have proportional sides", node_type="theorem")
    
    # Add the Pythagorean statement
    pythag = reasoning.add_node("In a right triangle, a² + b² = c², where c is the hypotenuse", node_type="theorem")
    
    # Start the reasoning
    reasoning.add_reasoning_step("I'll prove the Pythagorean theorem using similar triangles")
    
    # Construction steps
    step1 = reasoning.add_node("Draw a right triangle with sides a, b and hypotenuse c", node_type="step")
    step2 = reasoning.add_node("Draw an altitude h from the right angle to the hypotenuse", node_type="step")
    
    # Observations
    obs1 = reasoning.add_node("The altitude h creates two right triangles", node_type="observation")
    obs2 = reasoning.add_node("These triangles are similar to the original triangle", node_type="observation")
    
    # Properties derived from similarity
    prop1 = reasoning.add_node("The three triangles are similar to each other", node_type="property")
    prop2 = reasoning.add_node("Due to similar triangles, a/c = h/b", node_type="property")
    prop3 = reasoning.add_node("Due to similar triangles, b/c = h/a", node_type="property")
    
    # Algebraic manipulations
    step3 = reasoning.add_node("From a/c = h/b, we get ah = bc", node_type="step")
    step4 = reasoning.add_node("From b/c = h/a, we get bh = ac", node_type="step")
    step5 = reasoning.add_node("Combining, we have ah + bh = bc + ac", node_type="step")
    step6 = reasoning.add_node("Factor out h: h(a + b) = c(a + b)", node_type="step")
    step7 = reasoning.add_node("Divide both sides by (a + b): h = c", node_type="step")
    
    # Final conclusion
    conclusion = reasoning.add_node("Therefore, a² + b² = c²", node_type="conclusion")
    
    # Connect the nodes with appropriate relationships
    reasoning.add_edge(axiom1, step1, edge_type="leads_to")
    reasoning.add_edge(theorem1, obs1, edge_type="supports")
    reasoning.add_edge(theorem2, prop1, edge_type="implies")
    
    reasoning.add_edge(step1, step2, edge_type="leads_to")
    reasoning.add_edge(step2, obs1, edge_type="leads_to")
    reasoning.add_edge(obs1, obs2, edge_type="leads_to")
    reasoning.add_edge(obs2, prop1, edge_type="implies")
    
    reasoning.add_edge(prop1, prop2, edge_type="implies")
    reasoning.add_edge(prop1, prop3, edge_type="implies")
    
    reasoning.add_edge(prop2, step3, edge_type="leads_to")
    reasoning.add_edge(prop3, step4, edge_type="leads_to")
    reasoning.add_edge(step3, step5, edge_type="leads_to")
    reasoning.add_edge(step4, step5, edge_type="leads_to")
    reasoning.add_edge(step5, step6, edge_type="leads_to")
    reasoning.add_edge(step6, step7, edge_type="leads_to")
    reasoning.add_edge(step7, conclusion, edge_type="implies")
    
    reasoning.add_edge(axiom1, pythag, edge_type="part_of")
    reasoning.add_edge(conclusion, pythag, edge_type="supports")
    
    # Plot the reasoning graph
    reasoning.plot(figsize=(15, 10), node_size=3000, font_size=8)
    reasoning.print_reasoning_steps()
    reasoning.print_conclusions()
    
    return reasoning

if __name__ == "__main__":
    demo_math_reasoning()