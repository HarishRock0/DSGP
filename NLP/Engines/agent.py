import os
import sys
import json
import warnings
import datetime
from typing import Optional

warnings.filterwarnings('ignore')

# ── Query logger ──────────────────────────────────────────────────────────────
_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'query_log.jsonl'
)


def _log_query(query: str, route: str, confidence: float, response_preview: str) -> None:
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
        pass


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .LLMQ import LLMQueryEngine
    from .NLPC import NLPClusterQueryEngine
    from .tools import init_tools, build_function_tools, list_tools
except ImportError:
    from LLMQ import LLMQueryEngine
    from NLPC import NLPClusterQueryEngine
    from tools import init_tools, build_function_tools, list_tools


class LFSAgent:
    # ── Tool routing map ───────────────────────────────────────────────────────
    _ROUTE_TO_TOOL = {
        "resource_allocation": "allocate_resources",
        "compare_clusters":    "compare_clusters",
        "cluster_query":       "query_cluster",
        "insights":            "get_insights",
        "general_analysis":    "analyze_demographics",
    }

    def __init__(self, model_path: str, verbose: bool = True):
        print("\n" + "=" * 70)
        print("[LFS AGENTIC AI] Initializing")
        print("=" * 70)

        # Load data engines
        self.llm_engine  = LLMQueryEngine(model_path=model_path)
        self.nlpc_engine = NLPClusterQueryEngine(model_path=model_path)

        # Bind engines to @tool functions
        init_tools(self.llm_engine, self.nlpc_engine)
        self.tools = build_function_tools()

        print(f"\n {len(self.tools)} tools registered:")
        print(list_tools())

        # Build ReActAgent (Groq fallback for complex queries)
        self.agent = None
        self.llm   = self.llm_engine.llm

        if self.llm is not None and self.tools:
            try:
                from llama_index.core.agent import ReActAgent
                self.agent = ReActAgent.from_tools(
                    self.tools,
                    llm=self.llm,
                    verbose=verbose,
                    max_iterations=3,
                )
                print(" ReActAgent ready (Groq fallback for edge cases)")
            except Exception as exc:
                print(f"  ReActAgent setup failed: {exc}")
        else:
            print("  Groq API key not set — set GROQ_API_KEY in .env")

        self._mode = "keyword_router+groq"
        print(f"\n[MODE] Keyword Router → Tool → Groq LLM")
        print("=" * 70 + "\n")

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def chat(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Please enter a question."

        # Step 1: conversational guard
        conv = self._try_conversational(query)
        if conv:
            _log_query(query, 'conversational', 1.0, conv)
            return conv

        # Step 2+3+4: keyword route → tool → Groq format
        try:
            intent_result = self.nlpc_engine.understand_query(query)
            route      = intent_result['route']
            confidence = intent_result['confidence']
            tool_name  = self._ROUTE_TO_TOOL.get(route, "analyze_demographics")

            print(f"\n[ROUTE] {route} → {tool_name} (conf={confidence:.0%})")
            answer = self._execute_tool_route(query, tool_name)

            _log_query(query, route, confidence, answer[:300])
            return answer

        except Exception as exc:
            print(f"[WARN] Tool route failed: {exc}")

        # Step 5: ReActAgent fallback
        if self.agent is not None:
            try:
                print("[FALLBACK] Routing to ReActAgent (Groq)")
                response = self.agent.chat(query)
                answer   = str(response)

                if len(answer.strip()) < 20:
                    answer += (
                        "\n\n *Answer may be incomplete. "
                        "Try rephrasing your question.*"
                    )

                _log_query(query, 'reactagent', 0.0, answer)
                return answer

            except Exception as exc2:
                print(f"[WARN] ReActAgent also failed: {exc2}")
                _log_query(query, 'error', 0.0, str(exc2))
                return (
                    " Could not process your query. Please check:\n"
                    " • GROQ_API_KEY is set in your .env file\n"
                    "  • Your question is about LFS-2023 Sri Lanka data"
                )

        return " No inference engine available. Please set GROQ_API_KEY in .env"

    def reset(self) -> None:
        """Clear conversation memory."""
        if self.agent is not None:
            try:
                self.agent.reset()
            except Exception:
                pass

    # ── Conversational guard ───────────────────────────────────────────────────

    def _try_conversational(self, query: str) -> Optional[str]:
        """Handle greetings and meta-queries without tool calls."""
        import random
        q = query.lower().strip()

        if q in ['hi', 'hello', 'hey', 'greetings', 'namaste']:
            return random.choice([
                " Hello! I'm an AI assistant for the Sri Lanka Labour Force Survey 2023. "
                "Ask me about workforce data, vulnerable populations, resource allocation, or insights.",
                "Hi there! I can help you explore LFS-2023 data — employment, wages, "
                "disabilities, skills gaps, and more. What would you like to know?",
            ])

        if q in ['how are you', 'how are you?', "how's it going", "what's up"]:
            return "Doing great, thanks! Ready to help you analyze the Labour Force Survey data."

        if any(q.startswith(h) for h in [
            'help', 'what can you do', 'what tools', 'commands', 'show tools', 'list tools'
        ]):
            return (
                "🛠️ **Available capabilities:**\n\n"
                "1. **allocate_resources** — Find the most vulnerable people for aid\n"
                "   *'Give 50 laptops to the most vulnerable workers'*\n\n"
                "2. **get_insights** — Key labour market trends\n"
                "   *'What are the main employment challenges?'*\n\n"
                "3. **analyze_demographics** — Statistical breakdowns\n"
                "   *'Show age and gender distribution by district'*\n\n"
                "4. **compare_clusters** — Compare population segments\n"
                "   *'How do the clusters differ?'*\n\n"
                "5. **query_cluster** — Deep-dive into one cluster\n"
                "   *'Tell me about cluster 2'*\n\n"
                "6. **find_outliers** — Anomalous records\n"
                "   *'Find unusual cases in the data'*\n\n"
                "Just type your question naturally — I'll route it automatically."
            )

        if q.startswith('what is this') or q.startswith('who are you'):
            return (
                "I'm an **Agentic AI** built to analyse the Sri Lanka "
                "Labour Force Survey 2023 (18,937 respondents, 128 features).\n\n"
                "I can identify vulnerable populations, allocate resources fairly, "
                "and uncover labour market insights. Ask me anything about the data."
            )

        if q in ['thanks', 'thank you', 'thanks!', 'thankyou']:
            return "You're welcome! Feel free to ask more questions anytime."

        return None  # route to tools

    # ── Tool executor ──────────────────────────────────────────────────────────

    def _execute_tool_route(self, query: str, tool_name: str) -> str:
        """Call the appropriate engine method for the selected tool."""
        print(f"[EXEC] {tool_name}")

        if tool_name == "allocate_resources":
            return self.llm_engine.handle_allocation(query)
        elif tool_name == "compare_clusters":
            return self.llm_engine.compare_clusters()
        elif tool_name == "query_cluster":
            return self.llm_engine.ask_about_clusters(query)
        elif tool_name == "get_insights":
            return self.llm_engine.get_insights()
        else:
            return self.llm_engine.analyze_data(query)