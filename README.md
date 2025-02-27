# Ollama Geometry Reasoning Agent

This project transforms any Ollama LLM into a geometric reasoning agent that uses visual thinking to solve complex geometry problems.

## Core Concept: Geometric Reasoning as Etched Thought

The fundamental insight behind this agent is that **thoughts are not temporal but etched** - when we work through complex problems, creating spatial, visual representations of our thinking process helps us reason more effectively.

The GeometryNotebook serves as an "external memory" for the LLM, allowing it to:
1. Explore geometric properties visually
2. Document its reasoning process step by step
3. Test conjectures with concrete constructions
4. Build on previous insights to make discoveries

## Key Features

- **Visual Reasoning**: The agent uses geometric constructions to think through problems spatially
- **Reasoning Documentation**: Each significant thinking step is captured using `add_reasoning()`
- **Multi-step Reasoning**: Complex problems can be solved through multiple iterations, with each building on insights from previous steps
- **Reasoning Flow Visualization**: Creates a graph visualization showing how reasoning steps connect
- **Interactive Exploration**: Users can have conversations with the agent to explore geometric concepts

## Prerequisites

- Python 3.8+
- Ollama installed locally (https://ollama.ai/)
- LLM models downloaded via Ollama (e.g., `ollama pull llama3`)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install numpy matplotlib ollama ipython networkx
```

## Usage

### Standard Reasoning

```python
from ollama_geometry_agent import OllamaGeometryAgent

# Create a reasoning agent (higher temperature encourages creative thinking)
agent = OllamaGeometryAgent(model_name="llama3", temperature=0.7)

# Solve a geometric problem
results = agent.solve_problem(
    "Find the locus of points equidistant from two given points A(-2,0) and B(2,0)"
)

# View the reasoning steps
for step in agent.notebook.reasoning_steps:
    print(f"- {step}")
```

### Multi-step Reasoning for Complex Problems

```python
# For more complex problems, use multi-step reasoning
results = agent.multi_step_reasoning(
    """
    Given triangle ABC with vertices A(0,0), B(6,0), and C(3,4), 
    explore the Euler line connecting its orthocenter, 
    centroid, and circumcenter.
    """,
    steps=3  # Number of reasoning iterations
)

# Visualize the reasoning flow
flow_file = agent.visualize_reasoning_flow(results)
```

### Interactive Reasoning Session

```python
# Start an interactive session
agent.interactive_session()
```

In the interactive session, you can use the special command `multi:N your problem here` to trigger multi-step reasoning with N iterations.

## Example Problems to Explore

The agent excels at reasoning through problems like:

1. **Locus Problems**:
   - "Find the locus of points P such that |PA|^2 - |PB|^2 = k for constant k"
   - "Describe the locus of points equidistant from a point and a line"

2. **Triangle Centers**:
   - "Explore the relationship between the centroid, orthocenter, and circumcenter of a triangle"
   - "Discover the properties of the nine-point circle"

3. **Geometric Relationships**:
   - "Prove that the angle bisector of angle A in triangle ABC divides the opposite side BC in the ratio AB:AC"
   - "Investigate the power of a point with respect to a circle"

4. **Constructions with Verification**:
   - "Construct the center of a circle passing through three non-collinear points and verify the construction"
   - "Construct the common tangents to two circles and explain your reasoning"

## How It Works

1. The agent receives a geometric problem from the user
2. It analyzes the problem by breaking it down and identifying key relationships
3. For each reasoning step, it:
   - Documents its thinking using `add_reasoning()`
   - Creates geometric constructions to test ideas
   - Observes patterns and relationships in the visualization
4. As it discovers insights, it refines its approach
5. The agent continues this process until it reaches a solution

The agent's reasoning process is fully transparent - you can see both its visual constructions and its documented thought process.

## Customizing the Agent

To adapt the agent for your specific needs:

1. **Adjust Temperature**: Lower for more deterministic reasoning, higher for more creative exploration
2. **Modify System Prompt**: Edit the `_create_system_prompt()` method to emphasize different reasoning aspects
3. **Add Custom Geometric Functions**: Extend the GeometryNotebook with additional methods

## Limitations

- The quality of reasoning depends on the capabilities of the underlying LLM
- Very abstract geometric problems may require multiple iterations to solve
- The agent works best when problems can be approached through constructive geometry