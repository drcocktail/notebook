import json
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional, Any
import ollama
from IPython.display import display, clear_output
import time

# Import the GeometryNotebook class (assumes it's in a file called geometry_notebook.py)
from playground import GeometryNotebook

class OllamaGeometryAgent:
    def __init__(self, model_name: str = "llama3.2:latest", temperature: float = 0.7):
        """
        Initialize an Ollama-powered geometry reasoning agent.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature parameter for the LLM (higher = more creative reasoning)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.notebook = GeometryNotebook()
        self.conversation_history = []
        self.system_prompt = self._create_system_prompt()
        self.reasoning_stages = []
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt that defines the agent's reasoning capabilities."""
        return """You are a geometry reasoning agent that can solve complex geometric problems by thinking visually.
You have access to a GeometryNotebook that functions as an external memory and visualization system for your thinking.

The notebook lets you:
1. Create points, lines, and circles
2. Find intersections between geometric objects
3. Perform constructions like perpendicular/angle bisectors and parallel/perpendicular lines
4. Most importantly: document your reasoning through notebook.add_reasoning()

Your task is NOT just to construct geometric objects, but to SOLVE COMPLEX GEOMETRIC PROBLEMS by:
1. Breaking down the problem into smaller steps
2. Exploring different approaches
3. Testing conjectures
4. Building solutions incrementally
5. Using visualizations to guide your thinking

IMPORTANT: For each significant reasoning step, use notebook.add_reasoning() to document your thought process.
The notebook is your "thinking out loud" space - use it to capture your reasoning as you work.

Think of the geometric constructions not as the goal, but as a tool to help you solve problems.
The notebook is your spatial memory that etches your thoughts into a visual, persistent form.

Here are the available notebook methods:
- notebook.add_point(name, x, y, source="direct")
- notebook.add_line(point1_name, point2_name)
- notebook.add_circle(center_name, point_on_circle_name)
- notebook.find_intersection_line_line(line1, line2, name=None)
- notebook.find_intersection_circle_circle(circle1, circle2, name1=None, name2=None)
- notebook.find_intersection_line_circle(line, circle, name1=None, name2=None)
- notebook.perpendicular_bisector(point1_name, point2_name)
- notebook.angle_bisector(vertex_name, point1_name, point2_name)
- notebook.parallel_line(line, point_name)
- notebook.perpendicular_line(line, point_name)
- notebook.add_reasoning(text) <- USE THIS EXTENSIVELY
- notebook.plot() <- Call this frequently to visualize your reasoning
- notebook.print_reasoning()
- notebook.print_construction()
- notebook.save_figure(filename)

When solving a problem:
1. First, analyze the problem and explore its properties
2. Document your reasoning for each approach you consider
3. Test your ideas by creating geometric constructions
4. Visualize frequently to guide your thinking
5. When you discover insights, document them
6. Refine your approach based on what you learn
7. Build toward a solution incrementally

Your code should contain many instances of notebook.add_reasoning() that show your evolving understanding.
Return your reasoning and solution in a code block that can be executed.

Example response format:
To solve this problem, I need to explore the relationship between the points and determine if there's a pattern...

```python
# Initial analysis
notebook.add_reasoning("I'll start by establishing the key elements of the problem...")
notebook.add_point("A", -3, 0)
notebook.add_point("B", 3, 0)
notebook.add_line("A", "B")
notebook.plot()

# Explore a possible approach
notebook.add_reasoning("One approach is to consider the locus of points equidistant from points A and B...")
# Perpendicular bisector represents points equidistant from A and B
bisector = notebook.perpendicular_bisector("A", "B")
notebook.add_reasoning("The perpendicular bisector represents all points equidistant from A and B.")
notebook.plot()

# Test a conjecture
notebook.add_reasoning("Let me test if point C satisfies our conditions...")
notebook.add_point("C", 0, 4)
notebook.add_line("A", "C")
notebook.add_line("B", "C")
notebook.plot()

# Insight and refinement
notebook.add_reasoning("I notice that if we measure the distances AC and BC, they appear equal. This suggests...")
# (more reasoning and construction)

# Solution summary
notebook.add_reasoning("In conclusion, the solution to this problem is...")
notebook.plot()
notebook.print_reasoning()
notebook.print_construction()
notebook.save_figure("problem_solution.png")
```
"""

    def update_conversation(self, role: str, content: str) -> None:
        """
        Update conversation history with a new message.
        
        Args:
            role: Either "user" or "assistant"
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def query_ollama(self, user_query: str, max_tokens: int = 4096) -> str:
        """
        Query the Ollama model with the geometric problem.
        
        Args:
            user_query: User's problem or question
            max_tokens: Maximum tokens in response
            
        Returns:
            The model's response
        """
        # Update conversation with user query
        self.update_conversation("user", user_query)
        
        # Create messages list with system prompt and conversation history
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        
        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                }
            )
            
            # Extract the assistant's response
            assistant_response = response["message"]["content"]
            
            # Update conversation with assistant response
            self.update_conversation("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error querying Ollama: {str(e)}"
            print(error_msg)
            return error_msg
    
    def extract_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from response that is enclosed in code blocks.
        
        Args:
            response: The full text response from the LLM
            
        Returns:
            Extracted Python code or None if no code found
        """
        # Look for Python code blocks
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, response)
        
        if matches:
            # Join all found code blocks
            return "\n".join(matches)
        return None
    
    def extract_reasoning_steps(self, code: str) -> List[str]:
        """
        Extract reasoning steps from the executed code.
        
        Args:
            code: The Python code containing notebook.add_reasoning() calls
            
        Returns:
            List of reasoning steps
        """
        reasoning_steps = []
        pattern = r'notebook\.add_reasoning\(["\'](.+?)["\']\)'
        matches = re.findall(pattern, code)
        
        return matches
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a geometric problem using the notebook as a reasoning tool.
        
        Args:
            problem: The geometric problem to solve
            
        Returns:
            Dictionary with results information
        """
        print(f"🧠 Thinking about how to solve: {problem}")
        
        # Query the LLM
        response = self.query_ollama(problem)
        
        # Extract code from response
        code = self.extract_code(response)
        
        results = {
            "problem": problem,
            "response": response,
            "execution_result": None,
            "error": None,
            "filename": None,
            "reasoning_steps": []
        }
        
        if code:
            print("📝 Found reasoning approach, executing...")
            try:
                # Create a namespace with the notebook for execution
                namespace = {"notebook": self.notebook, "np": np}
                
                # Extract reasoning steps first
                results["reasoning_steps"] = self.extract_reasoning_steps(code)
                
                # Execute the code
                exec(code, namespace)
                
                results["execution_result"] = "Success"
                
                # Find the filename from save_figure call if it exists
                save_pattern = r"notebook\.save_figure\(['\"]([^'\"]+)['\"]\)"
                save_match = re.search(save_pattern, code)
                
                if save_match:
                    results["filename"] = save_match.group(1)
                else:
                    # Save with default name as fallback
                    results["filename"] = "geometry_reasoning.png"
                    self.notebook.save_figure(results["filename"])
                
                print(f"✅ Reasoning process completed and saved as {results['filename']}")
                
            except Exception as e:
                error_msg = f"Error executing reasoning code: {str(e)}"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                results["execution_result"] = "Error"
        else:
            print("❌ No executable reasoning found in response")
            results["error"] = "No executable reasoning code found in response"
            results["execution_result"] = "Error"
            
        return results
    
    def multi_step_reasoning(self, problem: str, steps: int = 3) -> Dict[str, Any]:
        """
        Perform multi-step reasoning on a complex geometric problem.
        
        Args:
            problem: The geometric problem to solve
            steps: Number of reasoning iterations to perform
            
        Returns:
            Dictionary with final results information
        """
        print(f"🔄 Starting multi-step reasoning process for: {problem}")
        print(f"Will perform up to {steps} reasoning iterations")
        
        current_problem = problem
        all_results = []
        final_result = None
        
        for i in range(steps):
            print(f"\n📊 Reasoning Iteration {i+1}/{steps}")
            
            # Solve current step
            result = self.solve_problem(current_problem)
            all_results.append(result)
            
            # Check if successful
            if result["execution_result"] != "Success":
                print(f"❌ Reasoning iteration {i+1} failed, stopping process")
                break
                
            # Extract reasoning steps from the notebook
            reasoning_text = "\n".join([
                f"- {step}" for step in self.notebook.reasoning_steps[-5:]
            ])
                
            # Prepare for next iteration with enhanced prompt
            if i < steps - 1:
                # Create a new notebook to continue reasoning
                old_notebook = self.notebook
                self.notebook = GeometryNotebook()
                
                # Create prompt for next iteration
                current_problem = f"""
Continue solving this geometric problem: {problem}

Here are the insights from the previous reasoning iteration:
{reasoning_text}

Based on these insights, explore further to reach a complete solution.
Can you refine the approach or explore alternative methods?
Examine any patterns or properties you've discovered so far.

Important: Build on the previous reasoning but feel free to try new approaches if needed.
"""
            else:
                final_result = result
        
        # If we completed all steps, return the final result
        if final_result is None and all_results:
            final_result = all_results[-1]
            
        # Return the combined reasoning from all steps
        final_result["all_iterations"] = all_results
        return final_result
    
    def interactive_session(self):
        """Start an interactive session where the user can input multiple geometry problems."""
        print(f"🧠 Starting interactive Ollama Geometry Reasoning Agent session with model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'multi:N problem' to use multi-step reasoning with N iterations.")
        
        while True:
            user_input = input("\n📐 Enter your geometric problem: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Ending session")
                break
            
            # Check for multi-step reasoning request
            if user_input.lower().startswith('multi:'):
                try:
                    parts = user_input.split(' ', 1)
                    steps = int(parts[0].split(':')[1])
                    problem = parts[1]
                    results = self.multi_step_reasoning(problem, steps)
                except Exception as e:
                    print(f"Error parsing multi-step request: {str(e)}")
                    print("Format should be: multi:3 your problem here")
                    continue
            else:
                results = self.solve_problem(user_input)
            
            # Display the reasoning and construction steps
            print("\n🧠 Reasoning process:")
            try:
                self.notebook.print_reasoning()
                self.notebook.print_construction()
            except:
                pass
            
            # If we're in a Jupyter notebook, display the image
            try:
                from IPython.display import Image, display
                if results["filename"] and os.path.exists(results["filename"]):
                    print("\n🖼️ Visualization of reasoning:")
                    display(Image(results["filename"]))
            except ImportError:
                pass
            
            # Reset for next problem
            self.reset_notebook()
            
    def reset_notebook(self):
        """Reset the geometry notebook for a new problem."""
        self.notebook = GeometryNotebook()
        print("🔄 Geometry notebook has been reset")

    def visualize_reasoning_flow(self, results):
        """
        Create a visualization of the reasoning flow from the results.
        
        Args:
            results: Results dictionary from solve_problem or multi_step_reasoning
            
        Returns:
            Path to the generated visualization file
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create a directed graph to represent reasoning flow
        G = nx.DiGraph()
        
        # Extract reasoning steps
        reasoning_steps = []
        if "all_iterations" in results:
            for i, iteration in enumerate(results["all_iterations"]):
                if "reasoning_steps" in iteration:
                    for j, step in enumerate(iteration["reasoning_steps"]):
                        node_id = f"I{i+1}S{j+1}"
                        G.add_node(node_id, label=step[:30] + "..." if len(step) > 30 else step)
                        reasoning_steps.append((node_id, step))
                        
                        # Connect with previous step
                        if j > 0:
                            G.add_edge(f"I{i+1}S{j}", node_id)
                        
                        # Connect iterations
                        if i > 0 and j == 0:
                            G.add_edge(f"I{i}S{len(results['all_iterations'][i-1]['reasoning_steps'])}", node_id)
        else:
            # Single iteration reasoning
            for j, step in enumerate(results.get("reasoning_steps", [])):
                node_id = f"S{j+1}"
                G.add_node(node_id, label=step[:30] + "..." if len(step) > 30 else step)
                reasoning_steps.append((node_id, step))
                
                # Connect with previous step
                if j > 0:
                    G.add_edge(f"S{j}", node_id)
        
        # Create the visualization if we have steps
        if reasoning_steps:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=False, node_size=1500, node_color="skyblue", 
                    font_size=10, arrows=True, edge_color="gray")
            
            # Add node labels
            node_labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=node_labels)
            
            plt.title("Geometry Reasoning Flow")
            plt.tight_layout()
            
            # Save the visualization
            flow_filename = "reasoning_flow.png"
            plt.savefig(flow_filename)
            plt.close()
            
            print(f"📊 Created reasoning flow visualization: {flow_filename}")
            return flow_filename
        
        return None