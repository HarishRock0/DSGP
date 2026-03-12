"""
LFS Agentic AI — autonomous question-answering over LFS-2023 data.

Architecture
------------
  User Query
      │
      ▼
  LFSAgent.chat(query)
      │
      ├─ [ReActAgent available] ──► Groq LLM reads @tool descriptions
      │                              selects the right tool(s) autonomously
      │                              chains tool calls if needed
      │                              returns synthesized answer
      │
      └─ [Fallback mode] ──────────► BART (NLPC) detects intent
                                      routes to correct LLMQ method
                                      returns answer

No hardcoded routing in the happy path — the LLM decides which tools to call.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Make Engines importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .LLMQ import LLMQueryEngine
from .NLPC import NLPClusterQueryEngine
from .tools import init_tools, build_function_tools, list_tools


class LFSAgent:
    """
    Agentic AI interface over the Sri Lanka Labour Force Survey 2023 dataset.

    The ReActAgent autonomously selects from registered @tool functions based
    on the user's natural language query — no manual routing needed.

    Fallback: if ReActAgent or Groq LLM is unavailable, BART-based intent
    detection (NLPC) is used instead, preserving original behaviour.

    Parameters
    ----------
    model_path : str
        Path to the pretrained ``skilldev_model.pkl`` file.
    verbose : bool
        If True, the ReActAgent prints each reasoning step (Thought/Action/Observation).
    """

    def __init__(self, model_path: str, verbose: bool = True):
        print("\n" + "=" * 70)
        print("🤖  LFS Agentic AI — Initializing")
        print("=" * 70)

        # ── Load data engines ──────────────────────────────────────────────────
        self.llm_engine = LLMQueryEngine(model_path=model_path)
        self.nlpc_engine = NLPClusterQueryEngine(model_path=model_path)

        # ── Bind engines to @tool functions ───────────────────────────────────
        init_tools(self.llm_engine, self.nlpc_engine)
        self.tools = build_function_tools()

        tool_names = []
        for t in self.tools:
            if hasattr(t, 'metadata'):
                tool_names.append(t.metadata.name)       # FunctionTool
            elif hasattr(t, '__name__'):
                tool_names.append(t.__name__)             # raw callable fallback

        print(f"\n✅ {len(self.tools)} tools registered:")
        print(list_tools())

        # ── Build ReActAgent ───────────────────────────────────────────────────
        self.agent = None
        self.llm = self.llm_engine.llm      # None if GROQ_API_KEY is missing

        if self.llm is not None and self.tools:
            try:
                from llama_index.core.agent import ReActAgent

                self.agent = ReActAgent.from_tools(
                    self.tools,
                    llm=self.llm,
                    verbose=verbose,
                    max_iterations=10,
                )
                print(f"\n✅ ReActAgent ready — autonomous tool selection active")
            except ImportError as exc:
                print(f"\n⚠️  ReActAgent import failed ({exc})")
                print("   Falling back to NLPC keyword/BART routing")
            except Exception as exc:
                print(f"\n⚠️  ReActAgent setup error: {exc}")
                print("   Falling back to NLPC keyword/BART routing")
        else:
            if self.llm is None:
                print("\n⚠️  LLM not available — set GROQ_API_KEY for agentic mode")
                print("   Get a free key → https://console.groq.com/keys")
            print("   Running in NLPC fallback mode (BART intent routing)")

        self._mode = "agentic" if self.agent else "fallback"
        print(f"\n🧭 Mode: {'🤖 Agentic (ReActAgent)' if self._mode == 'agentic' else '🔀 Fallback (BART routing)'}")
        print("=" * 70 + "\n")

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Return the active routing mode: 'agentic' or 'fallback'."""
        return self._mode

    def chat(self, query: str) -> str:
        """
        Process a user query end-to-end.

        In agentic mode the ReActAgent reads available @tool descriptions,
        selects the correct tool(s) for the query, executes them, and
        synthesizes a final answer — fully autonomously.

        In fallback mode BART detects intent and routes to the matching
        LLMQueryEngine method (original behaviour).

        Parameters
        ----------
        query : str
            Natural language question from the user.

        Returns
        -------
        str
            The formatted answer.
        """
        query = query.strip()
        if not query:
            return ""

        # ── Agentic path ───────────────────────────────────────────────────────
        if self.agent is not None:
            try:
                response = self.agent.chat(query)
                return str(response)
            except Exception as exc:
                print(f"⚠️  Agent error: {exc} — switching to fallback router")

        # ── Fallback path (BART + NLPC) ────────────────────────────────────────
        return self._fallback_route(query)

    def reset(self) -> None:
        """Clear the agent's conversation memory for a fresh session."""
        if self.agent is not None:
            try:
                self.agent.reset()
            except Exception:
                pass

    # ── Fallback router (preserves original NLPC → LLMQ behaviour) ────────────

    def _fallback_route(self, query: str) -> str:
        """
        BART intent detection → route to correct LLMQ method.
        Used when ReActAgent is unavailable (no GROQ_API_KEY, or import error).
        """
        result = self.nlpc_engine.understand_query(query)
        route = result['route']

        if route == "resource_allocation":
            return self.llm_engine.handle_allocation(query)
        elif route == "compare_clusters":
            return self.llm_engine.compare_clusters()
        elif route == "cluster_query":
            return self.llm_engine.ask_about_clusters(query)
        elif route == "insights":
            return self.llm_engine.get_insights()
        else:
            return self.llm_engine.analyze_data(query)
