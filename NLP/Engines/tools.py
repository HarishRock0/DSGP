
import functools
from typing import Callable, List, Optional

# ── Engine references (set once at agent init) ─────────────────────────────────
_llm_engine = None
_nlpc_engine = None


def init_tools(llm_engine, nlpc_engine=None) -> None:
    """Bind engine instances so all @tool functions can access them."""
    global _llm_engine, _nlpc_engine
    _llm_engine = llm_engine
    _nlpc_engine = nlpc_engine


# ── @tool decorator ─────────────────────────────────────────────────────────────
_TOOL_REGISTRY: List[Callable] = []


def tool(fn: Callable) -> Callable:

    fn._is_tool = True
    _TOOL_REGISTRY.append(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_tool = True
    return wrapper


# ── Tool implementations ────────────────────────────────────────────────────────

@tool
def allocate_resources(question: str, num_items: int = 10, item_type: str = "items") -> str:
    if _llm_engine is None:
        return " Engine not initialized. Call init_tools() first."
    return _llm_engine.handle_allocation(question, num_items=num_items, item_type=item_type)


@tool
def compare_clusters(question: str = "") -> str:

    if _llm_engine is None:
        return " Engine not initialized."
    return _llm_engine.compare_clusters()


@tool
def query_cluster(question: str) -> str:

    if _llm_engine is None:
        return " Engine not initialized."
    return _llm_engine.ask_about_clusters(question)


@tool
def get_insights(topic: str = "") -> str:

    if _llm_engine is None:
        return " Engine not initialized."
    return _llm_engine.get_insights(topic if topic else None)


@tool
def analyze_demographics(question: str) -> str:

    if _llm_engine is None:
        return " Engine not initialized."
    return _llm_engine.analyze_data(question)


@tool
def find_outliers() -> str:

    if _llm_engine is None and _nlpc_engine is None:
        return " Engine not initialized."
    if _nlpc_engine is not None:
        result = _nlpc_engine._find_outliers()
        count = len(result) if hasattr(result, '__len__') else '?'
        return (
            f" Outlier detection complete.\n"
            f"Found {count} statistical outlier records "
            f"(features > 3σ from cluster centroid).\n"
            f"See above output for the first 10 records."
        )
    return _llm_engine.analyze_data("identify outliers or unusual records in the dataset")


@tool
def get_cluster_stats() -> str:

    if _llm_engine is None and _nlpc_engine is None:
        return " Engine not initialized."
    if _nlpc_engine is not None:
        stats = _nlpc_engine._get_cluster_stats()
        return str(stats)
    return _llm_engine.analyze_data("cluster statistics and distribution summary")


@tool
def get_data_schema() -> str:

    if _llm_engine is None:
        return " Engine not initialized."
    return _llm_engine.analyze_data("what columns and fields are in the dataset?")


# ── FunctionTool builder ────────────────────────────────────────────────────────

def build_function_tools() -> list:

    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        print("  llama_index.core.tools not installed — returning raw tool functions")
        return list(_TOOL_REGISTRY)

    return [
        FunctionTool.from_defaults(fn=fn)
        for fn in _TOOL_REGISTRY
    ]

def list_tools() -> str:
    """Return a formatted list of all registered @tool names and one-line descriptions."""
    lines = []
    for fn in _TOOL_REGISTRY:
        doc_first_line = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        lines.append(f"  • {fn.__name__}: {doc_first_line}")
    return "\n".join(lines)
