"""CodeClaw MCP server exposing searchable session memory tools.

Run with: codeclaw serve
Install into Claude: codeclaw install-mcp

Tools:
- search_past_solutions
- get_project_patterns
- get_trajectory_stats
- get_session
- find_similar_sessions
- refresh_index
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def _get_mcp_or_exit():
    if not _MCP_AVAILABLE:
        print(
            "Error: 'mcp' package not installed. Run: pip install 'codeclaw[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)
    return FastMCP


def _ok_payload(results: Any, **meta: Any) -> str:
    return json.dumps({"ok": True, "results": results, "meta": meta}, indent=2)


def _error_payload(code: str, message: str, **meta: Any) -> str:
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message}, "meta": meta},
        indent=2,
    )


def _session_summary(session: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
    tool_uses: list[str] = []
    for message in session.get("messages", []):
        for tool_use in message.get("tool_uses", []):
            tool_name = str(tool_use.get("tool", "")).strip()
            if tool_name:
                tool_uses.append(tool_name)

    summary: dict[str, Any] = {
        "session_id": session.get("session_id"),
        "project": session.get("project"),
        "trajectory_type": session.get("trajectory_type"),
        "model": session.get("model"),
        "start_time": session.get("start_time"),
        "message_count": len(session.get("messages", [])),
        "tool_sequence": tool_uses[:20],
    }
    if rank is not None:
        summary["rank"] = rank
    return summary


def _parse_context_nodes(context: str) -> tuple[list[str], list[str]]:
    nodes: list[str] = []
    invalid_tokens: list[str] = []
    seen: set[str] = set()
    allowed_prefixes = {"tool", "file", "error"}

    for raw in context.split(","):
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if ":" not in lowered:
            invalid_tokens.append(token)
            continue
        prefix, value = lowered.split(":", 1)
        value = value.strip()
        if prefix not in allowed_prefixes or not value:
            invalid_tokens.append(token)
            continue
        normalized = f"{prefix}:{value}"
        if normalized in seen:
            continue
        seen.add(normalized)
        nodes.append(normalized)

    return nodes, invalid_tokens


class SessionIndexService:
    """Load sessions once, build a graph index, and serve cached access."""

    def __init__(
        self,
        discover_projects_fn: Callable[[], list[dict[str, Any]]] | None = None,
        parse_project_sessions_fn: Callable[..., list[dict[str, Any]]] | None = None,
        anonymizer_factory: Callable[[list[str]], Any] | None = None,
        graph_index_factory: Callable[[], Any] | None = None,
    ) -> None:
        if discover_projects_fn is None or parse_project_sessions_fn is None:
            from .parser import discover_projects, parse_project_sessions

            discover_projects_fn = discover_projects
            parse_project_sessions_fn = parse_project_sessions
        if anonymizer_factory is None:
            from .anonymizer import Anonymizer

            anonymizer_factory = Anonymizer
        if graph_index_factory is None:
            from .graph_index import GraphIndex

            graph_index_factory = GraphIndex

        self._discover_projects = discover_projects_fn
        self._parse_project_sessions = parse_project_sessions_fn
        self._anonymizer_factory = anonymizer_factory
        self._graph_index_factory = graph_index_factory

        self._sessions: list[dict[str, Any]] | None = None
        self._index: Any | None = None
        self._project_count = 0
        self._refresh_count = 0
        self._last_refresh_ms = 0.0
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        if self._sessions is None or self._index is None:
            with self._lock:
                if self._sessions is None or self._index is None:
                    self.refresh()

    def refresh(self) -> dict[str, Any]:
        start = time.perf_counter()
        anonymizer = self._anonymizer_factory([])
        projects = self._discover_projects()
        sessions: list[dict[str, Any]] = []
        for project in projects:
            sessions.extend(
                self._parse_project_sessions(
                    project.get("dir_name", ""),
                    anonymizer=anonymizer,
                    include_thinking=False,
                    source=project.get("source", "claude"),
                )
            )

        index = self._graph_index_factory()
        index.build(sessions)

        with self._lock:
            self._sessions = sessions
            self._index = index
            self._project_count = len(projects)
            self._refresh_count += 1
            self._last_refresh_ms = round((time.perf_counter() - start) * 1000, 2)
        return self.meta()

    def sessions(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return self._sessions or []

    def index(self) -> Any:
        self._ensure_loaded()
        return self._index

    def meta(self) -> dict[str, Any]:
        with self._lock:
            count = len(self._sessions or [])
            project_count = self._project_count
            refresh_count = self._refresh_count
            last_refresh_ms = self._last_refresh_ms
            index = self._index

        index_stats: dict[str, Any] = {}
        if index is not None and hasattr(index, "stats"):
            try:
                maybe_stats = index.stats()
                if isinstance(maybe_stats, dict):
                    index_stats = maybe_stats
            except Exception:
                index_stats = {}

        return {
            "session_count": count,
            "project_count": project_count,
            "refresh_count": refresh_count,
            "last_refresh_ms": last_refresh_ms,
            "index_stats": index_stats,
        }


def create_mcp_server(
    session_service: SessionIndexService | None = None,
    classify_fn: Callable[[dict[str, Any]], str] | None = None,
):
    """Create and return the FastMCP server instance with memory tools."""
    FastMCP = _get_mcp_or_exit()
    if classify_fn is None:
        from .classifier import classify_trajectory as classify_fn

    service = session_service or SessionIndexService()
    mcp = FastMCP("codeclaw")

    @mcp.tool()
    def refresh_index() -> str:
        """Rebuild session cache and graph index from local project logs."""
        meta = service.refresh()
        return _ok_payload([], action="index_refreshed", **meta)

    @mcp.tool()
    def search_past_solutions(query: str, max_results: int = 5) -> str:
        """Find past sessions matching a free-text query."""
        needle = query.lower().strip()
        if not needle:
            return _error_payload(
                "invalid_query",
                "query must be non-empty text.",
                provided_query=query,
                **service.meta(),
            )
        if max_results <= 0:
            return _error_payload(
                "invalid_max_results",
                "max_results must be greater than 0.",
                max_results=max_results,
                **service.meta(),
            )

        summaries: list[dict[str, Any]] = []
        for session in service.sessions():
            content_match = any(
                needle in str(message.get("content", "")).lower()
                for message in session.get("messages", [])
            )
            if not content_match and needle not in str(session.get("project", "")).lower():
                continue
            summaries.append(_session_summary(session, rank=len(summaries) + 1))
            if len(summaries) >= max_results:
                break

        return _ok_payload(
            summaries,
            query=needle,
            returned=len(summaries),
            max_results=max_results,
            **service.meta(),
        )

    @mcp.tool()
    def find_similar_sessions(context: str, max_results: int = 5) -> str:
        """Find similar sessions from graph context nodes.

        Context must be comma-separated nodes prefixed with one of:
        tool:, file:, error:
        """
        if max_results <= 0:
            return _error_payload(
                "invalid_max_results",
                "max_results must be greater than 0.",
                max_results=max_results,
                **service.meta(),
            )

        nodes, invalid_tokens = _parse_context_nodes(context)
        if not nodes:
            return _error_payload(
                "invalid_context",
                "context must include at least one node (tool:, file:, or error:).",
                invalid_tokens=invalid_tokens,
                context=context,
                **service.meta(),
            )

        sessions = service.index().query(nodes, max_results=max_results)
        results = [_session_summary(session, rank=i + 1) for i, session in enumerate(sessions)]
        return _ok_payload(
            results,
            context_nodes=nodes,
            invalid_tokens=invalid_tokens,
            returned=len(results),
            max_results=max_results,
            ranking="graph_similarity",
            **service.meta(),
        )

    @mcp.tool()
    def get_project_patterns(project: str | None = None) -> str:
        """Return per-project tool usage patterns from cached sessions."""
        project_filter = None
        if project is not None:
            project_filter = project.strip()
            if not project_filter:
                return _error_payload(
                    "invalid_project",
                    "project filter must be non-empty when provided.",
                    project=project,
                    **service.meta(),
                )

        per_project: dict[str, dict[str, Any]] = {}
        for session in service.sessions():
            project_name = str(session.get("project") or "unknown")
            if project_filter and project_name != project_filter:
                continue
            stats = per_project.setdefault(
                project_name,
                {"session_count": 0, "tool_counts": {}},
            )
            stats["session_count"] += 1
            for message in session.get("messages", []):
                for tool_use in message.get("tool_uses", []):
                    tool_name = str(tool_use.get("tool", "")).strip()
                    if not tool_name:
                        continue
                    stats["tool_counts"][tool_name] = stats["tool_counts"].get(tool_name, 0) + 1

        return _ok_payload(
            per_project,
            project_filter=project_filter,
            matched_projects=len(per_project),
            **service.meta(),
        )

    @mcp.tool()
    def get_trajectory_stats() -> str:
        """Return trajectory classification counts for cached sessions."""
        counts: dict[str, int] = {}
        for session in service.sessions():
            label = classify_fn(session)
            counts[label] = counts.get(label, 0) + 1
        return _ok_payload(
            counts,
            unique_trajectories=len(counts),
            **service.meta(),
        )

    @mcp.tool()
    def get_session(session_id: str) -> str:
        """Return full details for a specific session ID."""
        lookup = session_id.strip()
        if not lookup:
            return _error_payload(
                "invalid_session_id",
                "session_id must be non-empty.",
                session_id=session_id,
                **service.meta(),
            )

        for session in service.sessions():
            if str(session.get("session_id")) == lookup:
                return _ok_payload(session, session_id=lookup, **service.meta())
        return _error_payload(
            "session_not_found",
            "Session ID was not found in the local cache.",
            session_id=lookup,
            **service.meta(),
        )

    return mcp


def serve() -> None:
    """Entry point for ``codeclaw serve`` to run MCP over stdio."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


def _backup_corrupt_mcp_config(path: Path, content: str) -> Path | None:
    backup_path = path.with_name(path.name + ".corrupt.bak")
    try:
        backup_path.write_text(content, encoding="utf-8")
        return backup_path
    except OSError:
        return None


def install_mcp() -> None:
    """Install the CodeClaw MCP server into Claude's mcp.json config."""
    mcp_config_path = Path.home() / ".claude" / "mcp.json"
    mcp_config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    backup_path: Path | None = None
    if mcp_config_path.exists():
        try:
            raw_text = mcp_config_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error reading {mcp_config_path}: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            backup_path = _backup_corrupt_mcp_config(mcp_config_path, raw_text)
            print(
                (
                    f"Warning: {mcp_config_path} contained invalid JSON; "
                    f"{'backed up to ' + str(backup_path) if backup_path else 'backup failed'}."
                ),
                file=sys.stderr,
            )
            parsed = {}

        if isinstance(parsed, dict):
            existing = parsed
        else:
            backup_path = _backup_corrupt_mcp_config(mcp_config_path, raw_text)
            print(
                (
                    f"Warning: {mcp_config_path} root value is not a JSON object; "
                    f"{'backed up to ' + str(backup_path) if backup_path else 'backup failed'}."
                ),
                file=sys.stderr,
            )
            existing = {}

    mcp_servers = existing.get("mcpServers")
    if mcp_servers is None:
        mcp_servers = {}
        existing["mcpServers"] = mcp_servers
    elif not isinstance(mcp_servers, dict):
        existing["_codeclaw_previous_mcpServers"] = mcp_servers
        mcp_servers = {}
        existing["mcpServers"] = mcp_servers

    mcp_servers["codeclaw"] = {
        "command": sys.executable,
        "args": ["-m", "codeclaw.mcp_server", "--serve"],
    }

    mcp_config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"MCP server registered in {mcp_config_path}")
    print(
        json.dumps(
            {
                "installed": True,
                "config": str(mcp_config_path),
                "backup": str(backup_path) if backup_path else None,
            },
            indent=2,
        )
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CodeClaw MCP server")
    parser.add_argument("--serve", action="store_true", help="Run MCP server over stdio")
    parser.add_argument("--install", action="store_true", help="Install into Claude mcp.json")
    args = parser.parse_args()
    if args.install:
        install_mcp()
    else:
        serve()


if __name__ == "__main__":
    main()
