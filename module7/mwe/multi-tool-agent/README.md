# Multi-Tool Agent MWE

This minimal working example demonstrates an agent controller that selects between multiple tools based on query intent, with observable decision traces.

## What This Demonstrates

1. **Tool abstraction** - Wrapping capabilities as callable tools with descriptions
2. **Heuristic routing** - Query-based tool selection using keyword matching
3. **Context passing** - Storing retrieval output for subsequent summarization
4. **Execution tracing** - Capturing each step (thought, action, observation) for debugging

## Files

| File | Purpose |
|------|---------|
| `main.py` | Complete agent implementation with mock tools |
| `requirements.txt` | Dependencies (none for mock version) |
| `expected_output/sample_output.txt` | Expected console output |

## Running the Example

```bash
# No dependencies required - uses only Python standard library
python main.py
```

## Key Components

### Tool Interface

```python
@dataclass
class Tool:
    name: ToolName
    description: str
    function: Callable[[str], Any]
```

### Routing Logic

The agent uses keyword-based routing:
- **Summarize** keywords → `SummarizeTool` (if context exists)
- **Question** keywords → `RetrievalTool`
- **Fallback** → `ClarifyTool`

### Trace Structure

Each execution produces a trace with:
- Query and final answer
- Step-by-step execution (thought → routing → action → observation → answer)
- Tool sequence and timing information

## Extending This Example

To use with real RAG and LLM:

1. Replace `MockRetrievalTool` with your RAG pipeline:
   ```python
   from your_rag_module import RAGPipeline
   
   class RealRetrievalTool:
       def __init__(self, rag: RAGPipeline):
           self.rag = rag
       
       def __call__(self, query: str) -> dict:
           result = self.rag.query(query)
           return {"answer": result["response"], ...}
   ```

2. Replace heuristic routing with LLM-based routing:
   ```python
   def _route_query(self, query: str, trace: AgentTrace) -> ToolName:
       prompt = f"Select tool for: {query}\nTools: {self.tool_descriptions}"
       decision = self.llm(prompt)
       return parse_tool_selection(decision)
   ```

## Milestone 6 Alignment

This example covers Milestone 6 Part 2 requirements:
-  At least two tools (retrieval + summarization)
-  Tool selection policy (heuristic-based, documented)
-  Observable routing decisions in traces
-  Evaluation tasks with traces

## Related Materials

- Retrieval Agent Integration
- Multi-Tool Coordination
- Agent Evaluation Tracing