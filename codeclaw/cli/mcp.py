"""Serve and install-mcp subcommands for the CodeClaw MCP server."""


def handle_serve() -> None:
    from codeclaw.mcp_server import serve as _serve
    _serve()


def handle_install_mcp() -> None:
    from codeclaw.mcp_server import install_mcp as _install_mcp
    _install_mcp()
