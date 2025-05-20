# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-05-20

### Added
- **SFTP Support for Remote Notebooks**: Added SFTP integration for accessing and managing notebooks on remote SSH servers (resolves [issue #2](https://github.com/jbeno/cursor-notebook-mcp/issues/2)).
    - New command-line arguments: `--sftp-root` (multiple allowed), `--sftp-password`, `--sftp-key`, `--sftp-port`, `--sftp-no-interactive`, `--sftp-no-agent`, `--sftp-no-password-prompt`, `--sftp-auth-mode`.
    - Supports various SSH authentication methods including password, public key (with passphrase), SSH agent, and interactive (2FA).
    - Transparently handles file operations on remote SFTP paths.
    - Automatic tilde (`~`) expansion for remote paths.
- **Enhanced Web Transport with FastMCP (v2.3.4+):**
    - Integrated `FastMCP`'s `run()` method for robust handling of all transport modes (`stdio`, `streamable-http`, `sse`).
    - `--transport streamable-http`: Now uses `FastMCP`'s built-in Streamable HTTP, becoming the **recommended web transport**.
    - `--transport sse`: Now uses `FastMCP`'s built-in (but deprecated by FastMCP) two-endpoint SSE for legacy compatibility.
- New tool: `notebook_edit_cell_output` to allow direct manipulation and setting of cell outputs.
- New tool: `notebook_bulk_add_cells` for adding multiple cells to a notebook in a single operation.
- New tool: `notebook_get_server_path_context` to provide detailed server path configuration for robust client path construction.
- Added PowerShell script `run_tests.ps1` for test execution on Windows.
- Added `examples/demo_tools_list.py` script, demonstrating client-side MCP handshake and `tools/list` request (part of resolving [issue #5](https://github.com/jbeno/cursor-notebook-mcp/issues/5)).

### Changed
- **Refactored Server Logic**: Server now leverages `FastMCP`'s internal `run()` method for all transport modes, simplifying logic and improving reliability.
- Improved path handling for Windows-style paths and URL-encoded components (related to [issue #4](https://github.com/jbeno/cursor-notebook-mcp/issues/4)).
- Updated `README.md` with detailed instructions for all transport modes, `mcp.json` configurations, and refined transport recommendations. Added known issues for [issue #1](https://github.com/jbeno/cursor-notebook-mcp/issues/1) and [issue #3](https://github.com/jbeno/cursor-notebook-mcp/issues/3).
- Updated `examples/demo_tools_list.py` script to demonstrate client-side MCP handshake and `tools/list` request.
- Refined `cursor_rules.md` for clarity and to reflect new tool capabilities.
- **Simplified Installation**: `uvicorn` and `starlette` are now core dependencies. Optional extras `[http]` and `[sse]` removed. All transports supported by default install.
- Command-line `--transport` choices are now `stdio`, `streamable-http`, and `sse`.
- Updated code coverage metrics: Overall 82%; `notebook_ops.py` 92%, `server.py` 93%, `tools.py` 82%, `sftp_manager.py` 74%.

### Removed
- Custom SSE transport implementation (`cursor_notebook_mcp/sse_transport.py`), now handled by `FastMCP`.
- Removed `validate_imports` tool, which, along with `tools/list` availability and updated documentation, resolves [issue #5](https://github.com/jbeno/cursor-notebook-mcp/issues/5).

### Fixed
- HTTP 405 errors and client fallback issues for web transports by adopting `FastMCP`'s implementations.
- Addressed issues with Windows path interpretation (resolves [issue #4](https://github.com/jbeno/cursor-notebook-mcp/issues/4)).

## [0.2.4] - 2025-04-26

### Added

- Added tools to get an outline and search a notebook, so specific cells can be targeted for read/edit.
- The `notebook_get_outline` method analyzes a Jupyter notebook's structure, extracting cell types, line counts, and outlines for code and markdown cells.
- The `notebook_search` method allows for case-insensitive searching within notebook cells, returning matches with context snippets.
- Added dedicated tests for error paths and edge cases in the NotebookTools module, focusing on improving code coverage.
- Added validation tests for notebooks addressing invalid JSON and non-notebook files.
- Added tests for outline extraction with invalid code syntax.
- Added tests for empty search queries and behavior of large file truncation.
- Added edge case tests for export functionality and cell transformations.

### Changed
- Improved overall code coverage to 84%.
- Improved `tools.py` coverage to 80%.
- Achieved 100% coverage for `notebook_ops.py`.

## [0.2.3] - 2025-04-20

### Added
- CI workflow using GitHub Actions (`.github/workflows/ci.yml`) to run tests on Python 3.9 and 3.12.
- Code coverage reporting via `pytest-cov` and Coveralls integration (>70% overall, >80% for core tools).
- Additional tests for `tools.py`, `server.py`, and `notebook_ops.py` targeting error conditions, edge cases, and validation logic.
- Test script `run_tests.sh` to simplify local test execution with necessary environment variables.
- Tests for SSE transport layer (`tests/test_sse_transport.py`).

### Changed
- Improved documentation in `README.md`:
  - Added Video Walkthrough section and badges (Downloads, Issues, Coverage, MCP).
  - Clarified `stdio` vs `sse` transport configuration in `mcp.json`, recommending SSE.
  - Added troubleshooting tips for `stdio` environment issues.
  - Refined "Suggested Cursor Rules" for clarity, tone, and promoting proactive tool use.
  - Removed invalid comments from JSON examples.
  - Explicitly documented external system requirements (Pandoc, LaTeX) for PDF export.
- Updated project metadata (`classifiers`, `keywords`, `urls`) in `pyproject.toml`.
- Configured `pytest` via `pyproject.toml` to set environment variables (`JUPYTER_PLATFORM_DIRS`