#!/usr/bin/env python3
"""
ReAct Agent MWE - Explicit Reasoning + Acting loop for agentic AI.

This script demonstrates the ReAct (Reasoning + Acting) pattern where
an agent alternates between thinking about the problem and taking actions:

  1. Single-step ReAct — question → thought → action → observation → answer
  2. Multi-step ReAct — question requiring 2+ tool calls
  3. Max-steps termination — agent hits step limit
  4. Tool error recovery — agent handles tool failures gracefully
  5. Full trace export — JSON traces for all scenarios

Uses mock tools and mock LLM reasoning (keyword-based) — no external deps.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# =============================================================================
# Tool Definitions
# =============================================================================

class ToolName(Enum):
    CALCULATOR = "calculator"
    LOOKUP = "lookup"
    SEARCH = "search"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: str
    success: bool
    output: Any
    error: Optional[str] = None


# Mock tools
def calculator_tool(expression: str) -> ToolResult:
    """Evaluate a simple arithmetic expression."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return ToolResult("calculator", False, None,
                              f"Invalid expression: {expression}")
        result = eval(expression)  # Safe: only digits and operators
        return ToolResult("calculator", True, result)
    except Exception as e:
        return ToolResult("calculator", False, None, str(e))


def lookup_tool(topic: str) -> ToolResult:
    """Look up a fact from a mock knowledge base."""
    knowledge = {
        "rag": "RAG (Retrieval-Augmented Generation) grounds LLM outputs in "
               "retrieved documents. It has three stages: indexing, retrieval, "
               "and generation.",
        "chunking": "Document chunking splits text into 256-512 character "
                    "segments. Common strategies: fixed-size, sentence-boundary, "
                    "and recursive.",
        "faiss": "FAISS is a library for efficient similarity search. "
                 "IndexFlatL2 provides exact search; IndexIVF provides "
                 "approximate search.",
        "embedding": "Embedding models convert text to dense vectors. "
                     "all-MiniLM-L6-v2 produces 384-dim vectors. "
                     "Cosine similarity is the standard metric.",
        "population paris": "The population of Paris is approximately 2.1 million "
                            "in the city proper and 12.4 million in the metro area.",
        "population tokyo": "The population of Tokyo is approximately 13.9 million "
                            "in the city proper and 37.4 million in the metro area.",
    }

    topic_lower = topic.lower().strip()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return ToolResult("lookup", True, value)

    return ToolResult("lookup", False, None,
                      f"No information found for: {topic}")


def search_tool(query: str) -> ToolResult:
    """Search for documents (simulated)."""
    # Simulate search with pre-defined results
    results = {
        "rag pipeline": ["Doc1: RAG Architecture Overview",
                         "Doc2: Building RAG Systems"],
        "vector database": ["Doc1: FAISS Tutorial",
                            "Doc2: Chroma Getting Started"],
        "agent patterns": ["Doc1: ReAct Pattern Guide",
                           "Doc2: Tool-Augmented LLMs"],
    }

    query_lower = query.lower()
    for key, docs in results.items():
        if key in query_lower or any(w in query_lower for w in key.split()):
            return ToolResult("search", True, docs)

    return ToolResult("search", True, [f"No relevant results for: {query}"])


def failing_tool(query: str) -> ToolResult:
    """A tool that always fails (for error recovery demo)."""
    return ToolResult("failing_tool", False, None,
                      "Service unavailable: connection timeout after 30s")


# =============================================================================
# ReAct Agent
# =============================================================================

@dataclass
class ReActStep:
    """One step in the ReAct loop."""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class ReActTrace:
    """Complete trace of a ReAct execution."""
    query: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    status: str = "incomplete"  # "success", "max_steps", "error", "incomplete"
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "status": self.status,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "num_steps": len(self.steps),
            "final_answer": self.final_answer,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "is_final": s.is_final,
                }
                for s in self.steps
            ],
        }


class ReActAgent:
    """
    Agent that implements the ReAct (Reasoning + Acting) pattern.

    The loop:
      1. THOUGHT: Reason about the current state
      2. ACTION: Select and execute a tool
      3. OBSERVATION: Process the tool's output
      4. Repeat or produce FINAL ANSWER
    """

    def __init__(
        self,
        tools: dict[str, callable],
        max_steps: int = 5,
    ):
        self.tools = tools
        self.max_steps = max_steps

    def _mock_reason(self, query: str, history: list[ReActStep]) -> dict:
        """
        Simulate LLM reasoning to decide next action.

        In production, this would be an LLM call. Here we use
        keyword-based heuristics to demonstrate the pattern.
        """
        query_lower = query.lower()

        # If we have enough observations, formulate final answer
        if len(history) >= 1 and history[-1].observation:
            last_obs = history[-1].observation

            # Check if we need another step
            needs_more = False
            if "compare" in query_lower and len(history) < 2:
                needs_more = True
            if "and" in query_lower and len(history) < 2:
                needs_more = True

            if not needs_more:
                return {
                    "thought": f"I have the information needed to answer.",
                    "action": None,
                    "action_input": None,
                    "is_final": True,
                    "answer": f"Based on my research: {last_obs}",
                }

        # Determine which tool to use
        if any(w in query_lower for w in ["calculate", "compute", "math", "sum", "add"]):
            # Extract expression from query
            import re
            nums = re.findall(r'[\d+\-*/().]+', query)
            expr = nums[0] if nums else "0"
            return {
                "thought": f"This requires arithmetic. I'll use the calculator.",
                "action": "calculator",
                "action_input": expr,
                "is_final": False,
            }

        if any(w in query_lower for w in ["search", "find documents", "find articles"]):
            search_terms = query_lower.replace("search for", "").replace("find", "").strip()
            return {
                "thought": f"I need to search for relevant documents.",
                "action": "search",
                "action_input": search_terms,
                "is_final": False,
            }

        if any(w in query_lower for w in ["population", "capital"]):
            return {
                "thought": f"I need to look up factual information.",
                "action": "lookup",
                "action_input": query,
                "is_final": False,
            }

        # Check for multi-step: comparison queries
        if "compare" in query_lower:
            if not history:
                # First step: look up first topic
                topics = query_lower.replace("compare", "").strip().split(" and ")
                return {
                    "thought": f"Comparison query — I'll look up the first topic: {topics[0].strip()}",
                    "action": "lookup",
                    "action_input": topics[0].strip(),
                    "is_final": False,
                }
            elif len(history) == 1:
                topics = query_lower.replace("compare", "").strip().split(" and ")
                if len(topics) > 1:
                    return {
                        "thought": f"Now I'll look up the second topic: {topics[1].strip()}",
                        "action": "lookup",
                        "action_input": topics[1].strip(),
                        "is_final": False,
                    }

        # Default: lookup
        return {
            "thought": f"I'll look up information about this topic.",
            "action": "lookup",
            "action_input": query,
            "is_final": False,
        }

    def run(self, query: str) -> ReActTrace:
        """Execute the ReAct loop for a query."""
        start = time.time()
        trace = ReActTrace(query=query)

        for step_num in range(1, self.max_steps + 1):
            # THOUGHT: Reason about what to do
            reasoning = self._mock_reason(query, trace.steps)

            step = ReActStep(
                step_number=step_num,
                thought=reasoning["thought"],
                is_final=reasoning.get("is_final", False),
            )

            # Check if agent wants to give final answer
            if step.is_final:
                step.observation = "Final answer formulated."
                trace.steps.append(step)
                trace.final_answer = reasoning.get("answer", "No answer available.")
                trace.status = "success"
                break

            # ACTION: Execute the selected tool
            action = reasoning.get("action")
            action_input = reasoning.get("action_input", "")
            step.action = action
            step.action_input = action_input

            if action and action in self.tools:
                result = self.tools[action](action_input)
                if result.success:
                    step.observation = str(result.output)
                else:
                    step.observation = f"ERROR: {result.error}"
            elif action:
                step.observation = f"ERROR: Unknown tool '{action}'"
            else:
                step.observation = "No action taken."

            trace.steps.append(step)
        else:
            # Exhausted max_steps
            trace.status = "max_steps"
            last_obs = trace.steps[-1].observation if trace.steps else "No observations."
            trace.final_answer = (
                f"I reached the maximum number of steps ({self.max_steps}). "
                f"Partial result: {last_obs}"
            )

        trace.total_duration_ms = (time.time() - start) * 1000
        return trace


# =============================================================================
# Demonstrations
# =============================================================================

def print_trace(trace: ReActTrace, title: str = ""):
    """Pretty-print a ReAct trace."""
    if title:
        print(f"\n  {title}")
    print(f"  Query: \"{trace.query}\"")
    print(f"  Status: {trace.status} | Steps: {len(trace.steps)} | "
          f"Duration: {trace.total_duration_ms:.1f}ms")
    print(f"  {'-'*54}")

    for step in trace.steps:
        print(f"  Step {step.step_number}:")
        print(f"    💭 Thought:     {step.thought}")
        if step.action:
            print(f"    🔧 Action:      {step.action}(\"{step.action_input}\")")
        if step.observation:
            obs_short = step.observation[:80] + ("..." if len(step.observation) > 80 else "")
            print(f"    👁 Observation: {obs_short}")
        if step.is_final:
            print(f"    ✅ Final step")

    print(f"  {'-'*54}")
    answer = trace.final_answer or "No answer"
    print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")


def demo_single_step():
    """Demo 1: Single-step ReAct."""
    print("\n" + "=" * 60)
    print("Demo 1: Single-Step ReAct")
    print("=" * 60)
    print("Question answered in one thought → action → observation cycle.\n")

    agent = ReActAgent(
        tools={"lookup": lookup_tool, "calculator": calculator_tool,
               "search": search_tool},
    )

    trace = agent.run("What is RAG?")
    print_trace(trace)


def demo_multi_step():
    """Demo 2: Multi-step ReAct requiring 2+ tool calls."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Step ReAct")
    print("=" * 60)
    print("Comparison query requiring two lookups before answering.\n")

    agent = ReActAgent(
        tools={"lookup": lookup_tool, "calculator": calculator_tool,
               "search": search_tool},
    )

    trace = agent.run("Compare RAG and chunking")
    print_trace(trace)


def demo_max_steps():
    """Demo 3: Agent hits max-steps limit."""
    print("\n" + "=" * 60)
    print("Demo 3: Max-Steps Termination")
    print("=" * 60)
    print("Agent with max_steps=2 on a query needing 3 steps.\n")

    agent = ReActAgent(
        tools={"lookup": lookup_tool, "calculator": calculator_tool,
               "search": search_tool},
        max_steps=2,
    )

    trace = agent.run("Compare RAG and chunking")
    print_trace(trace)

    print(f"\n  ⚠  Agent terminated early — only completed {len(trace.steps)} of")
    print(f"     the needed steps. In production, max_steps prevents runaway loops.")


def demo_error_recovery():
    """Demo 4: Agent handles tool errors gracefully."""
    print("\n" + "=" * 60)
    print("Demo 4: Tool Error Recovery")
    print("=" * 60)
    print("Primary tool fails; agent observes the error.\n")

    # Register a failing tool as the lookup
    agent = ReActAgent(
        tools={"lookup": failing_tool, "calculator": calculator_tool,
               "search": search_tool},
    )

    trace = agent.run("What is FAISS?")
    print_trace(trace)

    print(f"\n  The agent received an error observation from the tool.")
    print(f"  In a production system, the reasoning step would detect the")
    print(f"  error and try an alternative tool or return a helpful message.")


def demo_trace_export():
    """Demo 5: Export all traces as JSON."""
    print("\n" + "=" * 60)
    print("Demo 5: Trace Export (JSON)")
    print("=" * 60)
    print("Run multiple scenarios and export traces for evaluation.\n")

    agent = ReActAgent(
        tools={"lookup": lookup_tool, "calculator": calculator_tool,
               "search": search_tool},
    )

    scenarios = [
        "What is RAG?",
        "What is embedding?",
        "Search for rag pipeline articles",
        "Calculate 256 * 4",
    ]

    all_traces = []
    for query in scenarios:
        trace = agent.run(query)
        all_traces.append(trace.to_dict())

    # Print summary table
    print(f"  {'#':>3}  {'Query':<40} {'Status':>10} {'Steps':>6}")
    print(f"  {'—'*3}  {'—'*40} {'—'*10} {'—'*6}")
    for i, t in enumerate(all_traces, 1):
        print(f"  {i:>3}  {t['query']:<40} {t['status']:>10} {t['num_steps']:>6}")

    # Print one trace as JSON
    print(f"\n  Sample JSON trace (scenario 1):")
    trace_json = json.dumps(all_traces[0], indent=4)
    for line in trace_json.split("\n")[:20]:
        print(f"    {line}")
    if len(trace_json.split("\n")) > 20:
        print(f"    ... ({len(trace_json.split(chr(10)))} total lines)")

    print(f"\n  Total traces: {len(all_traces)}")
    print(f"  Export format matches agent-evaluation-tracing.md structure.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("ReAct Agent MWE — Reasoning + Acting Loop")
    print("=" * 60)

    demo_single_step()
    demo_multi_step()
    demo_max_steps()
    demo_error_recovery()
    demo_trace_export()

    print("\n" + "=" * 60)
    print("MWE Complete — ReAct agent pattern demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
