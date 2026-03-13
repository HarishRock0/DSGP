"""
LFS Agentic AI — autonomous question-answering over LFS-2023 data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE: LLAMA GATE -> TOOL-ALIGNED -> EXECUTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER QUERY
   │
   ├─-> [CONVERSATIONAL GUARD]
   │   └─ Is it a greeting/help request?
   │      [EMOJI] YES -> Direct response (no tools needed)
   │      ❌ NO  -> Continue to LLAMA gate
   │
   ├─-> [LLAMA GATE] (llama3.2:3b local LLM)
   │   └─ "Does this query fit any of our 8 tools?"
   │      • Classifies intent (7 categories)
   │      • Calculates tool ALIGNMENT strength (0-1)
   │      • Scores: >0.4 = alignable, ≤0.4 = out-of-scope
   │
   ├─-> IF ALIGNABLE (LLAMA says >0.4)
   │   └─ [TOOL SELECTION]
   │      • LLAMA selects best-fit tool
   │      • Route directly to tool function
   │      • FAST, FREE, OFFLINE-CAPABLE
   │
   ├─-> IF OUT-OF-SCOPE (LLAMA says ≤0.4)
   │   └─ [GROQ/REACTAGENT FALLBACK]
   │      • Route to Groq advanced LLM reasoning
   │      • Autonomous tool selection via ReActAgent
   │      • More powerful but requires API key
   │
   └─-> RESPONSE


TOOLS (8 registered):
────────────────────
1. allocate_resources      -> Resource allocation (give items to vulnerable people)
2. compare_clusters        -> Compare population segments
3. query_cluster           -> Query specific cluster data
4. get_insights           -> Analyze trends/patterns
5. analyze_demographics   -> Statistical breakdowns
6. find_outliers          -> Anomaly detection
7. get_cluster_stats      -> Cluster-level statistics
8. get_data_schema        -> Dataset schema & encodings

"""

import os
import sys
import json
import warnings
import datetime
from typing import Optional

warnings.filterwarnings('ignore')

# ── Query logger ──────────────────────────────────────────────────────────────
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'query_log.jsonl')

def _log_query(query: str, route: str, confidence: float, response_preview: str) -> None:
    """Append a single query record to query_log.jsonl for auditing."""
    try:
        record = {
            'ts':         datetime.datetime.now().isoformat(timespec='seconds'),
            'query':      query[:200],
            'route':      route,
            'confidence': round(confidence, 3),
            'response':   response_preview[:300],
        }
        with open(_LOG_FILE, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception:
        pass   # logging must never crash the agent

# Make Engines importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .LLMQ import LLMQueryEngine
from .NLPC import NLPClusterQueryEngine
from .tools import init_tools, build_function_tools, list_tools


class LFSAgent:
    """
    Agentic AI with LLAMA GATE architecture for LFS-2023 dataset queries.

    ARCHITECTURE PRINCIPLE: "LLAMA determines if tool-alignable, then route"
    ────────────────────────────────────────────────────────────────────

    The agent uses a 2-tier inference strategy:

    TIER 1 - LLAMA GATE (llama3.2:3b, local):
      • Fast, free, offline-capable LLM
      • Classifies query intent into 7 categories
      • Calculates tool ALIGNMENT strength (0-1)
      • Decision: Is this query tool-alignable? (>0.4 confidence = YES)
      • If YES -> route to appropriate tool directly (fast path)
      • If NO  -> route to Tier 2 (advanced reasoning needed)

    TIER 2 - GROQ/REACTAGENT FALLBACK (cloud-based):
      • Advanced LLM for complex/out-of-scope queries
      • Autonomous tool selection via ReActAgent
      • More powerful reasoning but requires GROQ_API_KEY

    Result: Fast response for tool-aligned queries, smart fallback for edge cases.

    Parameters
    ----------
    model_path : str
        Path to the pretrained ``skilldev_model.pkl`` file.
    verbose : bool
        If True, the ReActAgent (if used) prints reasoning steps.
    """

    def __init__(self, model_path: str, verbose: bool = True):
        print("\n" + "=" * 70)
        print("[LFS AGENTIC AI] Initializing")
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

        print(f"\n[OK] {len(self.tools)} tools registered:")
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
                    # FIX: cap at 3 loops — prevents compounding hallucination errors
                    max_iterations=3,
                )
                print(f"\n[OK] ReActAgent ready — autonomous tool selection available as fallback")
            except ImportError as exc:
                print(f"\n[EMOJI][EMOJI]  ReActAgent import failed ({exc})")
                print("   Will use Ollama as primary engine")
            except Exception as exc:
                print(f"\n[EMOJI][EMOJI]  ReActAgent setup error: {exc}")
                print("   Will use Ollama as primary engine")
        else:
            if self.llm is None:
                print("\n[WARN] Groq API key not set — Ollama will be primary engine")
            print("   Running in Ollama-first mode (ReActAgent as fallback)")

        self._mode = "ollama_first"  # Always ollama-first now
        print(f"\n[MODE] Ollama Primary (llama3.2:3b) -> Groq Fallback (if available)")
        print("=" * 70 + "\n")

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Return the active routing mode: 'ollama_first' or 'groq_only'."""
        return self._mode

    def chat(self, query: str) -> str:
        """
        Agentic query processing with LLAMA gating.

        ARCHITECTURE FLOW:
        ──────────────────
        1. CONVERSATIONAL GUARD
           Check if query is greeting/meta (hi, help, etc.)
           -> Return direct response, no tools needed
           
        2. LLAMA GATE (llama3.2:3b)
           Ask: "Does this query align with any of our 8 tools?"
           -> Determine tool relevance
           
        3. IF TOOL-ALIGNABLE (LLAMA says yes)
           LLAMA selects the appropriate tool
           -> Route DIRECTLY to tool via LLMQ
           -> Fast, deterministic, no external APIs
           
        4. IF OUT-OF-SCOPE (LLAMA says no)
           Route to GROQ/ReActAgent for advanced reasoning
           -> Autonomous tool selection via LLM reasoning
           -> More powerful but higher cost/latency

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

        # ──── STEP 1: Conversational Guard ──────────────────────────────────
        conversational_response = self._try_conversational(query)
        if conversational_response is not None:
            return conversational_response

        # ──── STEP 2-3: LLAMA Gate -> Try Tool Routing ──────────────────────
        route_used = 'fast_path'
        confidence_used = 0.0
        try:
            result = self._llama_gate_and_route(query)
            _log_query(query, route_used, confidence_used, result)
            return result
        except Exception as exc:
            print(f"⚠️  LLAMA gate failed: {str(exc)[:80]} → Falling back to ReActAgent…")
            route_used = 'react_fallback'

        # ──── STEP 4: Fallback to GROQ/ReActAgent ────────────────────────────
        if self.agent is not None:
            try:
                print("[ROUTE] Routing to Groq/ReActAgent (max 3 reasoning loops)…")
                response = self.agent.chat(query)
                answer = str(response)

                # FIX: validate tool output — reject suspiciously short or empty answers
                if len(answer.strip()) < 20:
                    print("⚠️  ReActAgent returned a suspiciously short answer — flagging.")
                    answer = (
                        f"{answer}\n\n"
                        "⚠️ *Note: This answer may be incomplete. "
                        "Try rephrasing your question more specifically.*"
                    )

                _log_query(query, route_used, 0.0, answer)
                return answer
            except Exception as exc:
                print(f"[WARN] ReActAgent also failed: {exc}")
                _log_query(query, 'error', 0.0, str(exc))
                return (
                    "❌ Both inference engines failed. Please check:\n"
                    "  • Ollama is running: ollama serve\n"
                    "  • Groq API key is set: GROQ_API_KEY"
                )

        return "❌ No inference engine available. Please run: ollama serve"

    def reset(self) -> None:
        """Clear the agent's conversation memory for a fresh session."""
        if self.agent is not None:
            try:
                self.agent.reset()
            except Exception:
                pass

    # ── Conversational guard: handle greetings without tool calls ──────────────

    def _try_conversational(self, query: str) -> Optional[str]:
        """
        Handle non-data queries (greetings, meta-questions, help) without
        invoking expensive @tool functions. Returns None if query needs tools.

        Parameters
        ----------
        query : str
            Normalized user query.

        Returns
        -------
        str or None
            Response if query is conversational, else None (route to agent).
        """
        lower_q = query.lower().strip()

        # Greetings
        if lower_q in ['hi', 'hello', 'hey', 'greetings', 'g\'day', 'namaste']:
            responses = [
                "👋 Hello! I'm an AI assistant for the Sri Lanka Labour Force Survey 2023. I can help you analyze workforce data, identify vulnerable populations, allocate resources, and discover insights. What would you like to know?",
                "Hi there! 👋 I'm here to help you explore LFS-2023 data. Ask me about employment, wages, disabilities, skills gaps, or any other labour market insights.",
                "Hello! 🙂 I can help you understand the Sri Lanka Labour Force Survey 2023. Whether you want to find beneficiaries, compare clusters, or explore workforce trends, just ask!",
            ]
            import random
            return random.choice(responses)

        # How are you
        if lower_q in ['how are you', 'how are you?', 'how\'s it going', 'what\'s up']:
            return "I'm doing great, thanks for asking! 😊 Ready to help you analyze the Labour Force Survey data. What would you like to explore?"

        # Help request
        if any(lower_q.startswith(h) for h in ['help', 'what can you do', 'what tools', 'available commands', 'show tools', 'list tools']):
            return (
                "🛠[EMOJI] **Available Tools:**\n"
                "1. **allocate_resources** - Find vulnerable people for aid programs\n"
                "   *'Give 50 laptops to the most vulnerable workers'*\n\n"
                "2. **get_insights** - Understand key labour market trends\n"
                "   *'What are the main employment challenges?'*\n\n"
                "3. **analyze_demographics** - Explore workforce demographics\n"
                "   *'Show me age and gender distribution'*\n\n"
                "4. **find_outliers** - Identify people with unusual patterns\n"
                "   *'Who are the highest earners?'*\n\n"
                "5. **compare_clusters** - Compare vulnerable groups\n"
                "   *'How do clusters differ in skill gaps?'*\n\n"
                "6. **get_cluster_stats** - View cluster statistics\n"
                "   *'Tell me about cluster 0'*\n\n"
                "Type your question naturally, and I'll select the right tool(s)!\n"
            )

        # What is/who are (meta)
        if lower_q.startswith('what is this') or lower_q.startswith('who are you'):
            return (
                "I'm an **Agentic AI** powered by Groq LLM, built to autonomously analyze the "
                "Sri Lanka Labour Force Survey 2023 (18,937 respondents across 134 features).\n\n"
                "I use **ReActAgent** to smartly route your queries to specialized tools:\n"
                "• Identify vulnerable populations\n"
                "• Allocate resources fairly\n"
                "• Uncover labour market insights\n"
                "• Analyze employment patterns\n\n"
                "Ask me anything about the survey data!"
            )

        # Thank you
        if lower_q in ['thanks', 'thank you', 'thanks!', 'thankyou']:
            return "You're welcome! 😊 Feel free to ask more questions anytime."

        # No match — route to agent
        return None

    # ──── LLAMA GATE: Check tool alignment & route ────────────────────────────

    def _llama_gate_and_route(self, query: str) -> str:
        """
        LLAMA gate: Determine if query aligns with registered tools.

        FLOW:
        1. Ask LLAMA: "Does this query fit any of our 8 tools?"
        2. If YES -> Determine which tool -> Route directly
        3. If NO -> Raise exception (caught by chat(), routes to ReActAgent)

        Returns
        -------
        str
            Answer from tool OR exception if out-of-scope
        """
        # Step 1: Check if query is tool-alignable using LLAMA
        is_alignable, best_tool = self._check_tool_alignment(query)

        if not is_alignable:
            raise ValueError(f"Query out-of-scope for registered tools.\nQuery: '{query}'")

        # Step 2: Route to the determined tool
        print(f"[OK] Tool-alignable: {best_tool}")
        return self._execute_tool_route(query, best_tool)

    def _check_tool_alignment(self, query: str) -> tuple:
        """
        Use LLAMA to determine:
        1. Is this query tool-alignable? (yes/no)
        2. Which tool best fits this query?

        Returns
        -------
        tuple: (is_alignable: bool, best_tool: str)
        """
        # Get the intent from NLPC (which uses LLAMA/Ollama)
        intent_result = self.nlpc_engine.understand_query(query)
        route = intent_result['route']
        confidence = intent_result['confidence']

        # FIX: raised threshold 0.4 → 0.6 to reduce ReActAgent fallback frequency.
        # Lower threshold caused ambiguous queries to be misrouted to fast-path tools.
        is_alignable = confidence > 0.6

        tool_map = {
            "resource_allocation": "allocate_resources",
            "compare_clusters": "compare_clusters",
            "cluster_query": "query_cluster",
            "insights": "get_insights",
            "general_analysis": "analyze_demographics",  # Default fallback tool
        }

        best_tool = tool_map.get(route, "analyze_demographics")

        print(f"\n[LLAMA GATE] Query alignment analysis:")
        print(f"   Detected intent: {intent_result['intent']}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Is alignable: {'Yes' if is_alignable else 'No (out-of-scope)'}")
        print(f"   Best tool: {best_tool}")

        return is_alignable, best_tool

    def _execute_tool_route(self, query: str, tool_name: str) -> str:
        """
        Execute the tool determined by LLAMA gate.

        Parameters
        ----------
        query : str
            User query
        tool_name : str
            Name of tool to execute (from LLAMA routing)

        Returns
        -------
        str
            Result from the tool
        """
        print(f"\n[EXEC] Executing tool: {tool_name}")

        if tool_name == "allocate_resources":
            return self.llm_engine.handle_allocation(query)
        elif tool_name == "compare_clusters":
            return self.llm_engine.compare_clusters()
        elif tool_name == "query_cluster":
            return self.llm_engine.ask_about_clusters(query)
        elif tool_name == "get_insights":
            return self.llm_engine.get_insights()
        elif tool_name in ["analyze_demographics", "find_outliers", "get_cluster_stats"]:
            return self.llm_engine.analyze_data(query)
        else:
            return self.llm_engine.analyze_data(query)  # Fallback
