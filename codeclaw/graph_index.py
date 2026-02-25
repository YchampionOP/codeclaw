"""Graph-structured tool-call indexing for the CodeClaw MCP server.

Builds a lightweight in-memory directed graph (using networkx when available,
falling back to a pure-Python adjacency dict otherwise) where:

- Nodes represent concepts: file names, error types, function names, tool names
- Edges represent relationships: "co-occurs", "led_to_success", "caused_by", "was_fixed_by"

The graph is indexed from a list of session dicts (the same format produced by
``codeclaw.parser``) and can be queried to retrieve structurally similar past
sessions given a set of current context nodes.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Graph abstraction — uses networkx if available, pure-Python dict otherwise
# ---------------------------------------------------------------------------

try:
    import networkx as nx  # type: ignore

    def _make_graph():
        return nx.DiGraph()

    def _add_edge(graph, src: str, dst: str, **attrs) -> None:
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] = graph[src][dst].get("weight", 1) + 1
        else:
            graph.add_edge(src, dst, weight=1, **attrs)

    def _neighbors(graph, node: str) -> list[str]:
        if node not in graph:
            return []
        return list(graph.successors(node)) + list(graph.predecessors(node))

    def _has_node(graph, node: str) -> bool:
        return node in graph

    def _node_count(graph) -> int:
        return graph.number_of_nodes()

    def _edge_count(graph) -> int:
        return graph.number_of_edges()

    _NETWORKX_AVAILABLE = True

except ImportError:  # pragma: no cover — networkx optional
    class _PureGraph:
        """Minimal directed graph backed by adjacency dicts."""

        def __init__(self):
            self._out: dict[str, dict[str, dict]] = defaultdict(dict)
            self._in: dict[str, set] = defaultdict(set)

        def add_edge(self, src: str, dst: str, **attrs) -> None:
            if dst in self._out[src]:
                self._out[src][dst]["weight"] = self._out[src][dst].get("weight", 1) + 1
            else:
                self._out[src][dst] = {"weight": 1, **attrs}
            self._in[dst].add(src)
            # ensure nodes exist
            if src not in self._in:
                self._in[src]  # noqa: B018
            if dst not in self._out:
                self._out[dst]  # noqa: B018

        def has_edge(self, src: str, dst: str) -> bool:
            return dst in self._out.get(src, {})

        def successors(self, node: str):
            return list(self._out.get(node, {}).keys())

        def predecessors(self, node: str):
            return list(self._in.get(node, set()))

        def __contains__(self, node: str) -> bool:
            return node in self._out or node in self._in

        def number_of_nodes(self) -> int:
            nodes = set(self._out.keys()) | set(self._in.keys())
            return len(nodes)

        def number_of_edges(self) -> int:
            return sum(len(v) for v in self._out.values())

        def __getitem__(self, src: str):
            return self._out[src]

    def _make_graph():
        return _PureGraph()

    def _add_edge(graph, src: str, dst: str, **attrs) -> None:
        graph.add_edge(src, dst, **attrs)

    def _neighbors(graph, node: str) -> list[str]:
        if node not in graph:
            return []
        return graph.successors(node) + graph.predecessors(node)

    def _has_node(graph, node: str) -> bool:
        return node in graph

    def _node_count(graph) -> int:
        return graph.number_of_nodes()

    def _edge_count(graph) -> int:
        return graph.number_of_edges()

    _NETWORKX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

_TOOL_NODE_PREFIX = "tool:"
_FILE_NODE_PREFIX = "file:"
_ERROR_NODE_PREFIX = "error:"


def _normalize_node(label: str) -> str:
    """Lowercase + strip whitespace for consistent node IDs."""
    return label.strip().lower()


def _tool_node(name: str) -> str:
    return _TOOL_NODE_PREFIX + _normalize_node(name)


def _file_node(name: str) -> str:
    return _FILE_NODE_PREFIX + _normalize_node(name)


def _error_node(msg: str) -> str:
    return _ERROR_NODE_PREFIX + _normalize_node(msg[:60])


_ERROR_RE = re.compile(r"(error|exception|traceback|failed)", re.IGNORECASE)
_FILE_RE = re.compile(r"[\w./\-]+\.\w{1,6}")


def _extract_file_refs(text: str) -> list[str]:
    return [m for m in _FILE_RE.findall(text) if "/" in m or "." in m][:5]


def _extract_error_refs(text: str) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        if _ERROR_RE.search(line):
            hits.append(line.strip()[:60])
    return hits[:3]


def _index_session(graph: Any, session: dict) -> None:
    """Add edges to *graph* from a single session."""
    messages = session.get("messages", [])
    is_successful = session.get("trajectory_type") not in ("correction_loop",)

    tool_seq: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))

        # Index file references from content
        file_refs = _extract_file_refs(content)
        for fref in file_refs:
            fnode = _file_node(fref)
            _add_edge(graph, fnode, fnode, rel="self")  # ensure node exists

        # Index errors from content
        if role == "user" or role == "tool_result":
            for err in _extract_error_refs(content):
                enode = _error_node(err)
                _add_edge(graph, enode, enode, rel="self")

        # Tool uses
        for tu in msg.get("tool_uses", []):
            tool_name = str(tu.get("tool", "")).strip()
            if not tool_name:
                continue
            tnode = _tool_node(tool_name)
            tool_seq.append(tnode)

            # file → tool edges
            tool_input = str(tu.get("input", ""))
            for fref in _extract_file_refs(tool_input):
                fnode = _file_node(fref)
                _add_edge(graph, fnode, tnode, rel="accessed_by")

    # Sequential tool → tool edges
    for i in range(len(tool_seq) - 1):
        rel = "led_to_success" if is_successful else "co-occurs"
        _add_edge(graph, tool_seq[i], tool_seq[i + 1], rel=rel)


class GraphIndex:
    """In-memory graph index built from a list of sessions.

    Usage::

        index = GraphIndex()
        index.build(sessions)
        similar = index.query(["tool:bash", "tool:read"])
    """

    def __init__(self) -> None:
        self._graph = _make_graph()
        # Map node → list of session_ids that contain this node
        self._node_to_sessions: dict[str, list[str]] = defaultdict(list)
        self._sessions: dict[str, dict] = {}

    def build(self, sessions: list[dict]) -> None:
        """Index all *sessions*, replacing any previously indexed data."""
        self._graph = _make_graph()
        self._node_to_sessions = defaultdict(list)
        self._sessions = {}
        for session in sessions:
            self.add_session(session)

    def add_session(self, session: dict) -> None:
        """Incrementally add a single session to the index."""
        _index_session(self._graph, session)
        session_id = str(session.get("session_id", id(session)))
        self._sessions[session_id] = session

        # Record which nodes this session contributes to
        for msg in session.get("messages", []):
            for tu in msg.get("tool_uses", []):
                tool_name = str(tu.get("tool", "")).strip()
                if tool_name:
                    tnode = _tool_node(tool_name)
                    if session_id not in self._node_to_sessions[tnode]:
                        self._node_to_sessions[tnode].append(session_id)

    def query(self, context_nodes: list[str], max_results: int = 5) -> list[dict]:
        """Return up to *max_results* sessions structurally similar to *context_nodes*.

        Similarity is measured by the number of shared graph neighbors.
        """
        if not context_nodes:
            return []

        candidate_scores: dict[str, int] = defaultdict(int)
        for node in context_nodes:
            norm = _normalize_node(node)
            # Direct match
            for sid in self._node_to_sessions.get(norm, []):
                candidate_scores[sid] += 2
            # Neighbor traversal
            for neighbor in _neighbors(self._graph, norm):
                for sid in self._node_to_sessions.get(neighbor, []):
                    candidate_scores[sid] += 1

        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        results = []
        for sid, _score in ranked[:max_results]:
            session = self._sessions.get(sid)
            if session:
                results.append(session)
        return results

    def stats(self) -> dict:
        return {
            "nodes": _node_count(self._graph),
            "edges": _edge_count(self._graph),
            "sessions": len(self._sessions),
            "networkx_available": _NETWORKX_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Convenience: build from JSONL files
# ---------------------------------------------------------------------------

def build_index_from_jsonl(paths: list[Path]) -> GraphIndex:
    """Build a :class:`GraphIndex` from a list of JSONL file paths."""
    index = GraphIndex()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    session = json.loads(line)
                    index.add_session(session)
                except json.JSONDecodeError:
                    continue
    return index


def build_index_from_archive() -> GraphIndex:
    """Build a :class:`GraphIndex` from the default CodeClaw archive directory."""
    try:
        from codeclaw.config import CODECLAW_DIR
    except ImportError:
        from pathlib import Path
        CODECLAW_DIR = Path.home() / ".codeclaw"

    archive_dir = CODECLAW_DIR / "archive"
    pending = CODECLAW_DIR / "pending.jsonl"

    paths: list[Path] = []
    if archive_dir.exists():
        paths.extend(sorted(archive_dir.glob("*.jsonl")))
    if pending.exists():
        paths.append(pending)

    return build_index_from_jsonl(paths)
