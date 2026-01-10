"""
Multi-Tool Agent MWE - Demonstrates agent with retrieval and summarization tools.

This minimal working example shows how to build an agent controller that:
1. Selects between multiple tools based on query intent
2. Integrates RAG as a callable tool
3. Produces observable decision traces

Usage:
    python main.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import time


# =============================================================================
# Tool Definitions
# =============================================================================

class ToolName(Enum):
    """Available tools for the agent."""
    RETRIEVAL = "retrieval"
    SUMMARIZE = "summarize"
    CLARIFY = "clarify"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool: ToolName
    success: bool
    output: Any
    duration_ms: float
    error: Optional[str] = None


@dataclass
class Tool:
    """Tool definition with description for routing."""
    name: ToolName
    description: str
    function: Callable[[str], Any]
    
    def execute(self, input_text: str) -> ToolResult:
        """Execute tool with timing and error handling."""
        start = time.time()
        try:
            output = self.function(input_text)
            duration = (time.time() - start) * 1000
            return ToolResult(self.name, True, output, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolResult(self.name, False, None, duration, str(e))


# =============================================================================
# Mock Tools (Replace with real implementations in production)
# =============================================================================

class MockRetrievalTool:
    """Simulated RAG retrieval tool."""
    
    def __init__(self):
        # Mock knowledge base
        self.knowledge_base = {
            "rag": "RAG (Retrieval-Augmented Generation) combines retrieval with "
                   "generation to produce grounded responses. It reduces hallucination "
                   "by accessing external knowledge at inference time.",
            "vector": "Vector databases store embeddings and enable fast similarity "
                      "search using algorithms like HNSW and IVF.",
            "agent": "Agentic AI systems can reason about tasks, select tools, and "
                     "execute multi-step workflows autonomously.",
            "chunking": "Document chunking splits text into segments suitable for "
                        "embedding. Common approaches include fixed-size and semantic.",
        }
    
    def __call__(self, query: str) -> dict:
        """Search the mock knowledge base."""
        query_lower = query.lower()
        
        # Simple keyword matching (replace with real vector search)
        results = []
        for key, content in self.knowledge_base.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                results.append({"topic": key, "content": content})
        
        if not results:
            results = [{"topic": "general", "content": "No specific information found."}]
        
        return {
            "answer": results[0]["content"],
            "sources": [r["topic"] for r in results],
            "chunks": [r["content"][:100] + "..." for r in results],
        }
    
    @property
    def description(self) -> str:
        return ("Search the knowledge base for information. Use when the query "
                "asks about facts, definitions, or requires looking up information.")


class MockSummarizeTool:
    """Simulated summarization tool."""
    
    def __init__(self):
        self.last_context: Optional[str] = None
    
    def set_context(self, text: str):
        """Set text to summarize from previous retrieval."""
        self.last_context = text
    
    def __call__(self, instruction: str) -> dict:
        """Summarize based on instruction."""
        if not self.last_context:
            return {
                "summary": "No content available to summarize. Try retrieving first.",
                "success": False
            }
        
        # Mock summarization (replace with real LLM call)
        words = self.last_context.split()
        summary = " ".join(words[:20]) + "..." if len(words) > 20 else self.last_context
        
        return {
            "summary": f"Summary ({instruction}): {summary}",
            "input_length": len(self.last_context),
            "success": True
        }
    
    @property
    def description(self) -> str:
        return ("Summarize previously retrieved information. Use when asked to "
                "provide a brief overview or condense content.")


class MockClarifyTool:
    """Tool for requesting clarification."""
    
    def __call__(self, query: str) -> dict:
        return {
            "response": f"I need more information about: '{query}'. "
                        "Could you please clarify what you're looking for?",
            "needs_input": True
        }
    
    @property
    def description(self) -> str:
        return ("Request clarification from user. Use when the query is ambiguous "
                "or the agent cannot determine the appropriate action.")


# =============================================================================
# Agent Tracing
# =============================================================================

@dataclass
class AgentTrace:
    """Trace of agent execution for evaluation."""
    query: str
    steps: list = field(default_factory=list)
    final_answer: Optional[str] = None
    total_duration_ms: float = 0
    
    def add_step(self, step_type: str, content: dict):
        """Add a step to the trace."""
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "type": step_type,
            "content": content
        })
    
    def to_dict(self) -> dict:
        """Convert trace to dictionary."""
        return {
            "query": self.query,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_duration_ms": self.total_duration_ms,
            "tool_sequence": [
                s["content"].get("tool") for s in self.steps 
                if s["type"] == "action"
            ]
        }
    
    def format_for_display(self) -> str:
        """Format trace for human-readable output."""
        lines = [
            f"Query: {self.query}",
            f"Duration: {self.total_duration_ms:.1f}ms",
            "-" * 40,
        ]
        
        for step in self.steps:
            step_type = step["type"]
            content = step["content"]
            
            if step_type == "thought":
                lines.append(f"💭 Thought: {content.get('reasoning', '')}")
            elif step_type == "routing":
                lines.append(f"🔀 Route: {content.get('decision', '')} - {content.get('reasoning', '')}")
            elif step_type == "action":
                lines.append(f"🔧 Action: {content.get('tool', '')} <- {content.get('input', '')[:50]}")
            elif step_type == "observation":
                output = str(content.get('output', ''))[:100]
                lines.append(f"👁 Observation: {output}...")
            elif step_type == "answer":
                answer = str(content.get('final_answer', ''))[:100]
                lines.append(f"✅ Answer: {answer}...")
        
        return "\n".join(lines)


# =============================================================================
# Agent Controller
# =============================================================================

class MultiToolAgent:
    """Agent controller with tool selection and tracing."""
    
    def __init__(self, tools: dict[ToolName, Tool]):
        """
        Initialize agent with tools.
        
        Args:
            tools: Dict mapping ToolName to Tool instances
        """
        self.tools = tools
        self.context_buffer = ""  # Stores retrieval output for summarization
    
    def _route_query(self, query: str, trace: AgentTrace) -> ToolName:
        """
        Decide which tool to use based on query content.
        
        This uses simple heuristics. In production, replace with
        an LLM-based router for more sophisticated routing.
        """
        query_lower = query.lower()
        
        # Routing rules (priority order)
        summarize_keywords = ["summarize", "summary", "tldr", "brief", "condense", 
                             "key points", "main ideas", "overview"]
        retrieval_keywords = ["what", "how", "why", "when", "where", "who",
                             "explain", "describe", "tell me", "find", "search"]
        
        # Check for summarization (requires prior context)
        for keyword in summarize_keywords:
            if keyword in query_lower:
                if self.context_buffer:
                    selected = ToolName.SUMMARIZE
                    reasoning = f"Query contains '{keyword}' and context available"
                else:
                    selected = ToolName.RETRIEVAL
                    reasoning = f"Summarization requested but no context; retrieving first"
                break
        else:
            # Default to retrieval for knowledge queries
            selected = ToolName.RETRIEVAL
            reasoning = "Query appears to seek information from knowledge base"
        
        # Log routing decision
        trace.add_step("routing", {
            "decision": selected.value,
            "reasoning": reasoning,
            "context_available": bool(self.context_buffer)
        })
        
        return selected
    
    def run(self, query: str) -> tuple[str, AgentTrace]:
        """
        Execute agent for query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (final_answer, trace)
        """
        start = time.time()
        trace = AgentTrace(query=query)
        
        # Step 1: Initial thought
        trace.add_step("thought", {
            "reasoning": f"Analyzing query: '{query}'"
        })
        
        # Step 2: Route to appropriate tool
        selected_tool = self._route_query(query, trace)
        
        # Step 3: Execute selected tool
        trace.add_step("action", {
            "tool": selected_tool.value,
            "input": query
        })
        
        result = self.tools[selected_tool].execute(query)
        
        trace.add_step("observation", {
            "success": result.success,
            "output": result.output,
            "duration_ms": result.duration_ms
        })
        
        # Step 4: Update context buffer if retrieval
        if selected_tool == ToolName.RETRIEVAL and result.success:
            chunks = result.output.get("chunks", [])
            self.context_buffer = "\n".join(chunks)
            
            # Update summarizer context
            if ToolName.SUMMARIZE in self.tools:
                summarizer = self.tools[ToolName.SUMMARIZE].function
                if hasattr(summarizer, "set_context"):
                    summarizer.set_context(self.context_buffer)
        
        # Step 5: Formulate final answer
        if result.success:
            if selected_tool == ToolName.RETRIEVAL:
                final_answer = result.output.get("answer", str(result.output))
            elif selected_tool == ToolName.SUMMARIZE:
                final_answer = result.output.get("summary", str(result.output))
            else:
                final_answer = result.output.get("response", str(result.output))
        else:
            final_answer = f"Error executing {selected_tool.value}: {result.error}"
        
        trace.final_answer = final_answer
        trace.total_duration_ms = (time.time() - start) * 1000
        
        trace.add_step("answer", {
            "final_answer": final_answer
        })
        
        return final_answer, trace


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation(agent: MultiToolAgent, tasks: list[str]) -> list[dict]:
    """Run evaluation tasks and collect traces."""
    results = []
    
    for i, task in enumerate(tasks, 1):
        answer, trace = agent.run(task)
        results.append(trace.to_dict())
        
        print(f"\n{'='*60}")
        print(f"Task {i}: {task}")
        print("="*60)
        print(trace.format_for_display())
    
    return results


def generate_report(traces: list[dict]) -> dict:
    """Generate evaluation report from traces."""
    # Count tool usage
    tool_counts = {}
    for trace in traces:
        for tool in trace["tool_sequence"]:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    # Calculate latency stats
    latencies = [t["total_duration_ms"] for t in traces]
    
    return {
        "summary": {
            "total_tasks": len(traces),
            "unique_tools_used": len(tool_counts),
            "tool_distribution": tool_counts
        },
        "latency": {
            "avg_ms": sum(latencies) / len(latencies),
            "max_ms": max(latencies),
            "min_ms": min(latencies)
        }
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Demonstrate multi-tool agent with evaluation."""
    
    print("=" * 60)
    print("Multi-Tool Agent MWE")
    print("=" * 60)
    
    # Create tools
    retrieval_tool = MockRetrievalTool()
    summarize_tool = MockSummarizeTool()
    clarify_tool = MockClarifyTool()
    
    tools = {
        ToolName.RETRIEVAL: Tool(
            ToolName.RETRIEVAL,
            retrieval_tool.description,
            retrieval_tool
        ),
        ToolName.SUMMARIZE: Tool(
            ToolName.SUMMARIZE,
            summarize_tool.description,
            summarize_tool
        ),
        ToolName.CLARIFY: Tool(
            ToolName.CLARIFY,
            clarify_tool.description,
            clarify_tool
        )
    }
    
    # Create agent
    agent = MultiToolAgent(tools)
    
    # Define evaluation tasks
    evaluation_tasks = [
        # Retrieval tasks
        "What is RAG?",
        "Explain vector databases",
        "How does document chunking work?",
        
        # Multi-step: retrieval then summarization
        "Tell me about agentic AI systems",
        "Summarize the key points",  # Should use prior context
        
        # Edge case: summarization without context (agent should adapt)
        "Give me a brief overview",
    ]
    
    # Run evaluation
    traces = run_evaluation(agent, evaluation_tasks)
    
    # Generate report
    report = generate_report(traces)
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total tasks: {report['summary']['total_tasks']}")
    print(f"Tool distribution: {report['summary']['tool_distribution']}")
    print(f"Average latency: {report['latency']['avg_ms']:.2f}ms")
    print(f"Min latency: {report['latency']['min_ms']:.2f}ms")
    print(f"Max latency: {report['latency']['max_ms']:.2f}ms")
    
    return traces, report


if __name__ == "__main__":
    main()