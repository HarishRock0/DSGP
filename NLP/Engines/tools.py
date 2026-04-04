"""
Agentic tool registry for the LFS-2023 AI system.

Every function decorated with @tool is a self-describing capability.
The ReActAgent reads each tool's docstring to decide when and how to call it —
no hardcoded intent routing required.

Usage
-----
    from Engines.tools import init_tools, build_function_tools

    init_tools(llm_engine, nlpc_engine)   # bind engines once
    tools = build_function_tools()         # get LlamaIndex FunctionTool list
"""

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
    """
    Marks a function as an agentic LFS tool.

    The ReActAgent reads the wrapped function's docstring to decide when to
    invoke it.  Register new capabilities simply by adding @tool above any
    function in this module.
    """
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
    """
    Allocate or distribute resources/items to the most needy or vulnerable people
    identified in the Sri Lanka Labour Force Survey 2023 (LFS-2023).

    Use this tool when the user wants to:
    - Give, distribute, allocate, deliver, or send items to people
    - Identify beneficiaries for aid, scholarships, laptops, phones, food rations, clothing, etc.
    - Find who should receive a specific resource or government intervention
    - Target a specific number of people (e.g. "the top 50 most vulnerable workers")
    - Prioritise people by need (income, disability, informality, sector deprivation)

    Examples of triggering queries:
      "Give 50 laptops to the most vulnerable workers"
      "Allocate 100 food rations to people in need"
      "Who should receive the 30 sewing machines?"
      "Distribute scholarships to 20 needy students"

    Args:
        question  : The full, unmodified user question about resource allocation.
        num_items : How many items or people to select (default 10).
        item_type : Type of item being allocated (e.g. 'laptops', 'rations', 'scholarships').

    Returns:
        A ranked beneficiary list with demographic details, need scores, and
        an allocation rationale.
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized. Call init_tools() first."
    return _llm_engine.handle_allocation(question, num_items=num_items, item_type=item_type)


@tool
def compare_clusters(question: str = "") -> str:
    """
    Compare all demographic clusters in the LFS-2023 dataset side-by-side.

    Shows for every cluster: population size, average age, average income,
    dominant employment type, sector distribution (Urban/Rural/Estate),
    education level, and the key distinguishing characteristics of each group.

    Use this when the user asks to:
    - Compare groups, segments, or clusters against each other
    - See how worker segments differ from one another
    - Understand the makeup of each cluster or population group
    - Get a side-by-side breakdown of all clusters

    Examples of triggering queries:
      "Compare all clusters"
      "What are the differences between the groups?"
      "Show me how the segments differ"
      "Compare cluster 0 and cluster 1"

    Args:
        question : Optional follow-up or narrowing question about the comparison.

    Returns:
        A comparative summary of all clusters with key demographic statistics.
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized."
    return _llm_engine.compare_clusters()


@tool
def query_cluster(question: str) -> str:
    """
    Retrieve detailed information about a specific cluster or population segment
    in the LFS-2023 dataset.

    Use this when the user asks about a named or numbered cluster — e.g. 'cluster 0',
    'the vulnerable cluster', 'the digitally excluded group' — or wants to know
    who belongs to a specific population segment.

    Examples of triggering queries:
      "Tell me about cluster 2"
      "What is the high skill gap cluster?"
      "Show stats for the economically vulnerable group"
      "How many people are in cluster 1?"

    Args:
        question : The user's question mentioning the cluster number or label.

    Returns:
        Detailed demographics, statistics, and characteristics of the queried cluster.
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized."
    return _llm_engine.ask_about_clusters(question)


@tool
def get_insights(topic: str = "") -> str:
    """
    Generate key insights and trends from the Sri Lanka LFS-2023 dataset.

    Use this when the user wants a narrative summary, trends, or observations
    about any aspect of the workforce data.  Topics can include:
    - Income inequality and wage distribution
    - Employment and unemployment patterns
    - Education levels and digital literacy rates
    - Disability prevalence and impact
    - Regional or district-level disparities
    - Gender gaps in employment and income
    - Informal vs formal sector breakdown

    Examples of triggering queries:
      "What are the key insights from the data?"
      "Tell me about income trends"
      "Insights on education levels"
      "What patterns exist around disability?"
      "Summarise the employment situation"

    Args:
        topic : Specific topic to focus on (e.g. 'income', 'education', 'disability',
                'gender').  Leave empty for a broad five-insight dataset overview.

    Returns:
        Narrative insights with exact statistics and policy-relevant observations.
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized."
    return _llm_engine.get_insights(topic if topic else None)


@tool
def analyze_demographics(question: str) -> str:
    """
    Analyze demographic or statistical patterns in the Sri Lanka LFS-2023 data.

    This is the general-purpose data analysis tool.  Use it for any question that
    asks for counts, averages, percentages, distributions, or cross-tabulations
    across the workforce population.

    Examples of triggering queries:
      "How many people work in each sector?"
      "What is the average income by district?"
      "Show education level distribution by gender"
      "Employment statistics in the Western Province"
      "How many people have hearing difficulties?"
      "What percentage of workers are informal?"
      "Male vs female workforce breakdown"
      "Internet usage rate by age group"

    Args:
        question : The demographic or statistical question to analyze.

    Returns:
        Exact pre-computed statistics with LLM-formatted narrative analysis.
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized."
    return _llm_engine.analyze_data(question)


@tool
def find_outliers() -> str:
    """
    Identify statistical outliers or unusual records in the LFS-2023 dataset.

    Detects individuals whose feature values deviate more than 3 standard deviations
    from the cluster centroid — i.e. people who do not fit neatly into any cluster.
    These records may represent:
    - Data entry errors or anomalies worth reviewing
    - Genuinely extreme circumstances (very high income, severe multi-disability, etc.)
    - Edge-case individuals who sit at the boundary between clusters

    Use this when the user asks about:
      "Find outliers in the data"
      "Show unusual or anomalous records"
      "Which people don't fit any cluster?"
      "Identify extreme cases"

    Returns:
        A report of outlier records with their key demographic attributes.
    """
    if _llm_engine is None and _nlpc_engine is None:
        return "⚠️ Engine not initialized."
    if _nlpc_engine is not None:
        result = _nlpc_engine._find_outliers()
        count = len(result) if hasattr(result, '__len__') else '?'
        return (
            f"✅ Outlier detection complete.\n"
            f"Found {count} statistical outlier records "
            f"(features > 3σ from cluster centroid).\n"
            f"See above output for the first 10 records."
        )
    return _llm_engine.analyze_data("identify outliers or unusual records in the dataset")


@tool
def get_cluster_stats() -> str:
    """
    Get a comprehensive summary of cluster statistics from the LFS-2023 dataset.

    Returns an overview including:
    - Total number of records in the dataset
    - Number of clusters identified by the K-Means model
    - Population distribution across all clusters
    - Key aggregated metrics for each cluster

    Use this for a quick high-level snapshot of how the data is segmented.

    Examples of triggering queries:
      "Cluster statistics"
      "How many clusters are there?"
      "Overview of all clusters"
      "Cluster distribution summary"
      "How is the data split across clusters?"

    Returns:
        A summary table of cluster counts and key statistics.
    """
    if _llm_engine is None and _nlpc_engine is None:
        return "⚠️ Engine not initialized."
    if _nlpc_engine is not None:
        stats = _nlpc_engine._get_cluster_stats()
        return str(stats)
    return _llm_engine.analyze_data("cluster statistics and distribution summary")


@tool
def get_data_schema() -> str:
    """
    Return the dataset schema — every column name, its human-readable description,
    and the encoded value mappings used in the LFS-2023 dataset.

    Use this when the user asks about:
      "What columns are in the dataset?"
      "What does column Q16 mean?"
      "What are the possible values for SECTOR?"
      "Show me the data structure"
      "What fields are available?"

    Returns:
        Full column list with descriptions and value encodings (e.g. 1=Male, 2=Female).
    """
    if _llm_engine is None:
        return "⚠️ Engine not initialized."
    return _llm_engine.analyze_data("what columns and fields are in the dataset?")


# ── FunctionTool builder ────────────────────────────────────────────────────────

def build_function_tools() -> list:
    """
    Wrap every @tool-registered function as a LlamaIndex FunctionTool object.

    Must be called AFTER init_tools() so engine references are bound.
    Falls back to returning the raw callables if llama_index is not installed.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        print("⚠️  llama_index.core.tools not installed — returning raw tool functions")
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
