[![PyPI Version](https://img.shields.io/pypi/v/cursor-notebook-mcp)](https://pypi.org/project/cursor-notebook-mcp/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/cursor-notebook-mcp)](https://pypi.org/project/cursor-notebook-mcp/) [![Total Downloads](https://img.shields.io/pepy/dt/cursor-notebook-mcp)](https://pepy.tech/project/cursor-notebook-mcp) [![License](https://img.shields.io/github/license/jbeno/cursor-notebook-mcp)](https://github.com/jbeno/cursor-notebook-mcp/blob/main/LICENSE) [![Python Version](https://img.shields.io/pypi/pyversions/cursor-notebook-mcp)](https://pypi.org/project/cursor-notebook-mcp/) [![GitHub issues](https://img.shields.io/github/issues/jbeno/cursor-notebook-mcp)](https://github.com/jbeno/cursor-notebook-mcp/issues) [![Last Commit](https://img.shields.io/github/last-commit/jbeno/cursor-notebook-mcp)](https://github.com/jbeno/cursor-notebook-mcp) [![Coverage Status](https://coveralls.io/repos/github/jbeno/cursor-notebook-mcp/badge.svg?branch=main)](https://coveralls.io/github/jbeno/cursor-notebook-mcp?branch=main) ![](https://badge.mcpx.dev 'MCP') ![](https://badge.mcpx.dev?type=server&features=tools 'MCP server with features')

# Jupyter Notebook MCP Server (for Cursor)

This directory contains a Model Context Protocol (MCP) server designed to allow AI agents **within Cursor** to interact with Jupyter Notebook (`.ipynb`) files. It was created to overcome a limitation with Cursor. As of version 0.48.9, in Agent mode, the model could not edit notebooks or notebook cells in response to dialog in the AI chat pane. This provides the agent with a suite of MCP tools that allow direct notebook cell manipulation.

I'm sure at some point this will be handled natively by Cursor, but I have a data science background, and I live in Jupyter notebooks. I got tired of copy/paste-ing output of the chat manually into the notebook cells.

Although designed to overcome a limitation with Cursor, this MCP server does not have anything specific to Cursor other than the configuration instructions. You could easily configure this for use with Claude Code or any model/agent that can take advantage of MCP.

This MCP server uses the `nbformat` library to safely manipulate notebook structures and enforces security by restricting operations to user-defined directories. It also uses `nbconvert` to enable exporting notebooks to various formats like Python scripts, HTML, and more. The server handles all notebook operations through a clean API that maintains notebook integrity and prevents malformed changes.

## Video Walkthrough

[![Video Walkthrough Thumbnail](https://img.youtube.com/vi/VOVMH-tle14/maxresdefault.jpg)](https://youtu.be/VOVMH-tle14)

[Cursor Jupyter Notebook MCP Server](https://youtu.be/VOVMH-tle14) (YouTube) walks through:
  - The current **limitations** of editing notebooks directly in Cursor.
  - **Installing** and **configuring** the Notebook MCP Server.
  - **Creating a notebook** from scratch (example shown: Singular Value Decomposition tutorial in less than 2 minutes).
  - Demonstrating various **editing tools** (edit, split, duplicate cells).
  - Reading notebook **metadata**.
  - **Exporting** notebooks to python



## Latest Version

**Current Version:** `0.2.4` - See the [CHANGELOG.md](CHANGELOG.md) for details on recent changes. Two new tools were added that make it easier to work with large notebooks:

- `notebook_get_outline` analyzes a Jupyter notebook's structure, extracting cell types, line counts, and headings/functions/classes. It then returns an outline the agent can reference.
- `notebook_search` enables case-insensitive searching within notebook cells. The results show which cell the match was found, and include contextual snippets. This helps the agent know which cell to read/edit when asked to modify something.

## Features

Exposes the following MCP tools (registered under the `notebook_mcp` server):

*   `notebook_create`: Creates a new, empty notebook file.
*   `notebook_delete`: Deletes an existing notebook file.
*   `notebook_rename`: Renames/moves a notebook file from one path to another.
*   `notebook_read`: Reads an entire notebook and returns its structure as a dictionary.
*   `notebook_read_cell`: Reads the source content of a specific cell.
*   `notebook_add_cell`: Adds a new code or markdown cell after a specified index.
*   `notebook_edit_cell`: Replaces the source content of a specific cell.
*   `notebook_delete_cell`: Deletes a specific cell.
*   `notebook_change_cell_type`: Changes a cell's type (code, markdown, or raw).
*   `notebook_duplicate_cell`: Duplicates a cell multiple times (default: once).
*   `notebook_get_cell_count`: Returns the total number of cells.
*   `notebook_read_metadata`: Reads the top-level notebook metadata.
*   `notebook_edit_metadata`: Updates the top-level notebook metadata.
*   `notebook_read_cell_metadata`: Reads the metadata of a specific cell.
*   `notebook_read_cell_output`: Reads the output list of a specific code cell.
*   `notebook_edit_cell_metadata`: Updates the metadata of a specific cell.
*   `notebook_clear_cell_outputs`: Clears the outputs and execution count of a specific cell.
*   `notebook_clear_all_outputs`: Clears outputs and execution counts for all code cells.
*   `notebook_move_cell`: Moves a cell to a different position.
*   `notebook_split_cell`: Splits a cell into two at a specified line number.
*   `notebook_merge_cells`: Merges a cell with the cell immediately following it.
*   `notebook_validate`: Validates the notebook structure against the `nbformat` schema.
*   `notebook_get_info`: Retrieves general information (cell count, metadata, kernel, language info).
*   `notebook_export`: Exports the notebook to another format (e.g., python, html) using nbconvert. **Note:** See External Dependencies below for requirements needed for certain export formats like PDF.
* `notebook_get_outline`: Produces an outline showing cell numbers with major headings/functions and line counts to make it easier for the agent to navigate a large notebook.
* `notebook_search`: Searches cells for a keyword, showing which cell matches were found with contextual snippets. This helps the agent know which cell to read/edit when asked to modify something.

## Requirements

This project has both Python package dependencies and potentially external system dependencies for full functionality.

### Python Dependencies

*   **Python Version:** 3.10+
*   **Core:** `mcp>=0.1.0`, `nbformat>=5.0`, `nbconvert>=6.0`, `ipython`, `jupyter_core`. These are installed automatically when you install `cursor-notebook-mcp`.
*   **Optional - SSE Transport:** `uvicorn>=0.20.0`, `starlette>=0.25.0`. Needed only if using the SSE transport mode. Install via `pip install cursor-notebook-mcp[sse]`.
*   **Optional - Development/Testing:** `pytest>=7.0`, `pytest-asyncio>=0.18`, `pytest-cov`, `coveralls`. Install via `pip install -e ".[dev]"` from source checkout.

### External System Dependencies

These are **not** Python packages and must be installed separately on your system for certain features to work:

*   **Pandoc:** Required by `nbconvert` for many non-HTML export formats (including the intermediate steps for PDF). See [Pandoc installation instructions](https://pandoc.org/installing.html).
*   **LaTeX (XeLaTeX recommended):** Required by `nbconvert` for exporting notebooks directly to PDF (`--to pdf` option used by `notebook_export` with `export_format="pdf"`). See [Installing TeX](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex).

If these external dependencies are missing, the `notebook_export` tool may fail when attempting to export to formats that rely on them (like PDF).

## Installation

### From PyPI

```bash
# Basic installation (stdio transport only)
pip install cursor-notebook-mcp

# With SSE transport support
pip install "cursor-notebook-mcp[sse]"
```

### Development Installation (From Source)

1.  Clone this repository:
    ```bash
    git clone https://github.com/jbeno/cursor-notebook-mcp.git # Or your fork
    cd cursor-notebook-mcp
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    # Using Python's venv
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Or using uv (if installed)
    # uv venv
    # source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  Install in editable mode with all optional dependencies:
    ```bash
    # Includes SSE and Test dependencies
    pip install -e ".[dev]"
    
    # Or install just the base + SSE
    # pip install -e ".[sse]"
    
    # Or install just the base
    # pip install -e .
    ```

## Running the Server

There are two main ways to run the server:

### 1. Direct Execution (Recommended for Testing/Development)

Activate your virtual environment and run the main script directly:

```bash
# stdio transport (default)
python notebook_mcp_server.py --allow-root /path/to/notebooks

# sse transport
python notebook_mcp_server.py --transport sse --allow-root /path/to/notebooks --host 127.0.0.1 --port 8080
```

### 2. Using the Installed Script (After `pip install`)

If you have installed the package (even with `-e`), the `cursor-notebook-mcp` command should be available in your virtual environment:

```bash
# stdio transport
cursor-notebook-mcp --allow-root /path/to/notebooks

# sse transport
cursor-notebook-mcp --transport sse --allow-root /path/to/notebooks --host 127.0.0.1 --port 8080
```

## Cursor Integration (`mcp.json`)

To make Cursor aware of this server, configure it in `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project-specific).

**Recommendation: Use SSE Transport**

While both `stdio` and `sse` transport modes are supported, **using SSE is generally recommended** for integration with Cursor. 

*   **Simpler Configuration:** The `mcp.json` setup only requires the server's URL.
*   **Avoids Environment Issues:** Since you run the server process manually in its own terminal with its virtual environment activated, you avoid potential conflicts or complications related to Cursor launching the server process directly using `stdio` and ensuring the correct Python environment and packages are used.

See the sections below for configuring each transport type.

### For stdio Transport

If you choose to use `stdio` (where Cursor launches and manages the server process), you need to tell Cursor how to start the server using the `command` and `args` fields. **Care must be taken to ensure Cursor uses the correct Python environment containing this package and its dependencies.** There are two primary ways to configure the `command`:

1.  **Use the installed script (Simplest `stdio` method):** Point `command` to the `cursor-notebook-mcp` script located in your virtual environment's `bin` directory (e.g., `.venv/bin/cursor-notebook-mcp`). This requires installing the package first.
2.  **Use direct Python execution:** Set `command` to the path of the Python interpreter in your virtual environment (e.g., `.venv/bin/python`). Then, the *first* item in the `args` list must be the absolute path to the server script within the project (e.g., `/path/to/project/cursor_notebook_mcp/server.py`).

**Troubleshooting `stdio` Environment Issues:**

If the server fails to start or cannot find dependencies when using `stdio`, it likely means Cursor is not launching the process with the correct virtual environment activated. A common workaround is to:

1.  Activate your virtual environment manually in your terminal (`source .venv/bin/activate`).
2.  Launch Cursor *from that same terminal* by navigating to your project directory and running `cursor .`.

This ensures Cursor inherits the activated environment, which should then be passed down to the `stdio` server process it launches.

**Example (using installed script method):**

Make sure to replace `/absolute/path/to/venv/bin/cursor-notebook-mcp` and `/absolute/path/to/your/notebooks` with the correct paths for your system.

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "/absolute/path/to/venv/bin/cursor-notebook-mcp",
      "args": [
        "--allow-root", "/absolute/path/to/your/notebooks"
      ]
    }
  }
}
```

*(Note: If using direct Python execution, modify the `command` and `args` like this):*
```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "/absolute/path/to/project/cursor_notebook_mcp/server.py", 
        "--allow-root", "/absolute/path/to/your/notebooks"
      ]
    }
  }
}
```

### For SSE Transport

When using `sse`, you must run the server process manually first (see "Running the Server" section). Then, configure Cursor to connect to the running server's URL.

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "url": "http://127.0.0.1:8080/sse"
    }
  }
}
```

**Note**: When using SSE transport, make sure the server is started *before* attempting to use it in Cursor.

### Suggested Cursor Rules

For smooth collaboration with the AI agent on Jupyter Notebooks, you might want to add rules like these to your Cursor configuration. Go to Cursor Settings > Rules and add them in either User Roles or Project Rules. This ensures that Cursor's AI features will consistently follow these best practices when working with Jupyter notebooks.

```markdown 
### Jupyter Notebook Rules (Using notebook_mcp):

1.  **Tool Usage:**
    *   Always use the tools provided by the `notebook_mcp` server for operations on Jupyter Notebook (`.ipynb`) files.
    *   Avoid using the standard `edit_file` tool on `.ipynb` files, as this can corrupt the notebook structure.

2.  **Investigation Strategy:**
    *   A comprehensive suite of tools is available to inspect notebooks. If the user mentions an issue, a specific cell, or asks for a modification, first attempt to gather context independently.
    *   Use the available tools (`notebook_get_outline`, `notebook_search`, `notebook_read`, `notebook_read_cell`, `notebook_get_info`, `notebook_read_metadata`, `notebook_read_cell_output`, `notebook_validate`) to examine the notebook structure, content, metadata, and outputs to locate the relevant context or identify the problem.
    *   Ask the user for clarification only if the necessary information cannot be determined after using the investigation tools.

3.  **Available Tools:**
    *   Be aware of the different categories of tools: Situation awareness (`notebook_get_outline`), Search (`notebook_search`), File operations (`create`, `delete`, `rename`), Notebook/Cell Reading (`read`, `read_cell`, `get_cell_count`, `get_info`), Cell Manipulation (`add_cell`, `edit_cell`, `delete_cell`, `move_cell`, `change_cell_type`, `duplicate_cell`, `split_cell`, `merge_cells`), Metadata (`read/edit_metadata`, `read/edit_cell_metadata`), Outputs (`read_cell_output`, `clear_cell_outputs`, `clear_all_outputs`), and Utility (`validate`, `export`, `diagnose_imports`).

4.  **Math Notation:** For LaTeX in Markdown cells, use `$ ... $` for inline math and `$$ ... $$` for display math. Avoid `\( ... \)` and `\[ ... \]`.

5.  **Cell Magics:**
    *   Avoid unsupported cell magics like `%%bash`, `%%timeit`, and `%%writefile`.
    *   Use `!command` for shell commands instead of `%%bash`.
    *   Use `%timeit` (line magic) for timing single statements.
    *   `%%html` works for rendering HTML output.
    *   `%%javascript` can execute (e.g., `alert`), but avoid relying on it for manipulating cell output display.

6.  **Rich Outputs:** Matplotlib, Pandas DataFrames, Plotly, ipywidgets (`tqdm.notebook`), and embedded HTML in Markdown generally render correctly.

7.  **Mermaid:** Diagrams in ` ```mermaid ``` ` blocks are not rendered by default.

8.  **Character Escaping in `source` Parameter:**
    *   When providing the `source` string for `add_cell` or `edit_cell`, ensure that backslashes (`\`) are handled correctly. Newline characters **must** be represented as `\n` (not `\\n`), and LaTeX commands **must** use single backslashes (e.g., `\Sigma`, not `\\Sigma`).
    *   Incorrect escaping by the tool or its interpretation can break Markdown formatting (like paragraphs intended to be separated by `\n\n`) and LaTeX rendering.
    *   After adding or editing cells with complex strings (especially those involving newlines or LaTeX), consider using `read_cell` to verify the content was saved exactly as intended and correct if necessary.
```

## Command-Line Arguments

The server accepts the following command-line arguments:

*   `--allow-root`: (Required, can use multiple times) Absolute path to directory where notebooks are allowed.
*   `--log-dir`: Directory to store log files. Defaults to `~/.cursor_notebook_mcp`.
*   `--log-level`: Set the logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `INFO`.
*   `--max-cell-source-size`: Maximum allowed size in bytes for cell source content. Defaults to 10 MiB.
*   `--max-cell-output-size`: Maximum allowed size in bytes for cell output content. Defaults to 10 MiB.
*   `--transport`: Transport type to use: `stdio` or `sse`. Defaults to `stdio`.
*   `--host`: Host to bind the SSE server to. Only used with `--transport=sse`. Defaults to `127.0.0.1`.
*   `--port`: Port to bind the SSE server to. Only used with `--transport=sse`. Defaults to `8080`.

## Security

*   **Workspace Root Enforcement:** The server **requires** the `--allow-root` command-line argument during startup. It will refuse to operate on any notebook file located outside the directories specified by these arguments. This is a critical security boundary.
*   **Path Handling:** The server uses `os.path.realpath` to resolve paths and checks against the allowed roots before any read or write operation.
*   **Input Validation:** Basic checks for `.ipynb` extension are performed.
*   **Cell Source Size Limit:** The server enforces a maximum size limit (configurable via `--max-cell-source-size`, default 10 MiB) on the source content provided to `notebook_edit_cell` and `notebook_add_cell` to prevent excessive memory usage.
*   **Cell Output Size Limit:** The server enforces a maximum size limit (configurable via `--max-cell-output-size`, default 10 MiB) on the total serialized size of outputs returned by `notebook_read_cell_output`.

## Limitations

*   **No Cell Execution:** This server **cannot execute** notebook cells. It operates solely on the `.ipynb` file structure using the `nbformat` library and does not interact with Jupyter kernels. Cell execution must be performed manually by the user within the Cursor UI (selecting the desired kernel and running the cell). Implementing execution capabilities in this server would require kernel management and introduce significant complexity and security considerations.

## Known Issues

*   **UI Refresh Issues:** Occasionally, some notebook operations (like cell splitting or merging) may succeed at the file level, but the Cursor UI might not show the updated content correctly. In such situations, you can:
    * Close and re-open the notebook file
    * Save the file, which might prompt to "Revert" or "Overwrite" - select "Revert" to reload the actual file content

## Development & Testing

1. Setup virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Run tests:
   ```bash
   # Use the wrapper script to ensure environment variables are set
   ./run_tests.sh 
   # Or run specific tests
   # ./run_tests.sh tests/test_notebook_tools.py
   ```

## Issues

If you encounter any bugs or issues, please submit them to our GitHub issue tracker:

1. Visit [jbeno/cursor-notebook-mcp](https://github.com/jbeno/cursor-notebook-mcp/issues)
2. Click on "New Issue"
3. Provide:
   - A clear description of the problem
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Your environment details (OS, Python version, etc.)
   - Any relevant error messages or logs
   - Which model and client/version you're using

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please make sure your PR:
- Includes tests for new functionality
- Updates documentation as needed
- Follows the existing code style
- Includes a clear description of the changes

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Author

This project was created and is maintained by Jim Beno - jim@jimbeno.net