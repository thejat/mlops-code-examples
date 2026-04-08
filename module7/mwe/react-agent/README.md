# ReAct Agent MWE

**Explicit Reasoning + Acting loop for agentic AI — pure Python, no external dependencies.**

## Overview

This MWE demonstrates the ReAct (Reasoning + Acting) pattern where an agent alternates between thinking and taking actions:

1. **Single-step ReAct** — question → thought → action → observation → answer
2. **Multi-step ReAct** — comparison query requiring 2 tool calls
3. **Max-steps termination** — agent hits step limit, returns partial answer
4. **Tool error recovery** — agent handles tool failures
5. **Trace export** — JSON traces for all scenarios

## Prerequisites

- Python 3.10+
- No external packages required (uses mock tools and mock LLM reasoning)

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/react-agent

# 2. Run the demo
python main.py
```

## How ReAct Differs from Multi-Tool Agent

| Aspect | `react-agent` (this MWE) | `multi-tool-agent` (existing) |
|--------|-------------------------|-------------------------------|
| Focus | The **reasoning loop** pattern | **Tool routing** and evaluation |
| Loop | Explicit thought → action → observation | Single route → execute |
| Multi-step | Agent reasons about intermediate results | One-shot tool selection |
| Error handling | Agent observes errors in reasoning | Tool-level error handling |
| Tracing | JSON traces per reasoning step | Evaluation report |

## Expected Output

| Demo | Scenario | Key Lesson |
|------|----------|------------|
| 1. Single-step | Direct factual lookup | Basic ReAct cycle |
| 2. Multi-step | Comparison needing 2 lookups | Agent chains tool calls |
| 3. Max-steps | Step limit hit | Prevents runaway loops |
| 4. Error recovery | Tool failure observed | Agent sees errors in reasoning |
| 5. Trace export | JSON output for 4 scenarios | Structured evaluation data |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| ReAct loop | `ReActAgent.run()` — thought/action/observation cycle |
| Mock reasoning | `_mock_reason()` — keyword heuristics simulating LLM |
| Tool interface | `ToolResult` dataclass with success/error |
| Max-steps guard | `max_steps` parameter prevents infinite loops |
| Structured traces | `ReActTrace.to_dict()` → JSON-serializable output |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add LLM-based reasoning by calling an API in `_mock_reason()`
2. Implement retry logic when a tool fails (try alternative tool)
3. Add a memory/scratchpad that persists across queries
4. Build a planning step before the ReAct loop (Plan-then-Act)

## Related Reading

- Agentic AI Patterns (`notes7/instructional-materials/agentic-ai-patterns.md`)
- Agent Evaluation & Tracing (`notes7/instructional-materials/agent-evaluation-tracing.md`)
- Multi-Tool Coordination (`notes7/instructional-materials/multi-tool-coordination.md`)
