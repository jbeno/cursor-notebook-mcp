# Cursor Notebook MCP Examples

This directory contains examples demonstrating how to use the Cursor Notebook MCP server.
The main `README.md` in the parent directory contains more comprehensive setup and configuration instructions.

## 1. Streamable HTTP Transport (Recommended)

With Streamable HTTP, you run the server process manually, and Cursor connects to it over the network. This is the recommended method for most setups.

### A. Running the Server (Manual Start Required)

Ensure the package is installed (e.g., `pip install cursor-notebook-mcp`) or your virtual environment is active if running from source.

*   **Using the installed script:**
    ```bash
    cursor-notebook-mcp --transport streamable-http --allow-root /path/to/your/notebooks --host 127.0.0.1 --port 8080
    ```
*   **Running from a source checkout (ensure dependencies are installed and venv is active):**
    ```bash
    python -m cursor_notebook_mcp.server --transport streamable-http --allow-root /path/to/your/notebooks --host 127.0.0.1 --port 8080
    ```
    Replace `/path/to/your/notebooks` with the actual path.

### B. Cursor `mcp.json` Configuration

Create or update your `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project-specific) file:
```json
{
  "mcpServers": {
    "notebook_mcp": {
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```
*Note: The server must be running before Cursor attempts to connect. The `/mcp` path is the default for `FastMCP`'s streamable HTTP transport.*

## 2. SSE Transport (Legacy Web/Network)

SSE is a legacy transport but still supported. Streamable HTTP is preferred. You run the server process manually.

### A. Running the Server (Manual Start Required)

Ensure the package is installed or venv is active for source.

*   **Using the installed script:**
    ```bash
    cursor-notebook-mcp --transport sse --allow-root /path/to/your/notebooks --host 127.0.0.1 --port 8080
    ```
*   **Running from a source checkout (ensure venv is active):**
    ```bash
    python -m cursor_notebook_mcp.server --transport sse --allow-root /path/to/your/notebooks --host 127.0.0.1 --port 8080
    ```
    *Note: If running Streamable HTTP on port 8080 simultaneously, you'll need to choose a different port for SSE (e.g., `--port 8081`) and update the URL in `mcp.json` accordingly.*

### B. Cursor `mcp.json` Configuration

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "url": "http://127.0.0.1:8080/sse"
    }
  }
}
```
*Note: The `/sse` path is the default for `FastMCP`'s SSE transport. Adjust the port in the URL if you changed it when starting the server.*

## 3. stdio Transport

With `stdio` transport, Cursor launches and manages the server process directly.

### Basic `mcp.json` for stdio:

```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "cursor-notebook-mcp",
      "args": [
        "--allow-root", "/absolute/path/to/your/notebooks",
        "--log-level", "INFO"
        // Add other server arguments here if needed
      ]
    }
  }
}
```
Place this file in `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project-specific).

### Environment Management for stdio Transport

Ensure Cursor uses the correct Python environment.

**Option 1: Using absolute path to venv's installed script (Recommended for `venv`)**
```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "/absolute/path/to/venv/bin/cursor-notebook-mcp",
      "args": ["--allow-root", "/path/to/notebooks"]
    }
  }
}
```

**Option 2: Using venv's Python to run the module**
```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "-m", "cursor_notebook_mcp.server",
        "--allow-root", "/path/to/notebooks"
      ]
    }
  }
}
```

**Option 3: Using a wrapper script (like `launch-notebook-mcp.sh`)**
```json
{
  "mcpServers": {
    "notebook_mcp": {
      "command": "/absolute/path/to/your/launch-notebook-mcp.sh",
      "args": ["--allow-root", "/path/to/notebooks"]
    }
  }
}
```
*(The `launch-notebook-mcp.sh` script in this directory helps activate a venv.)*

**Option 4: Using `uv` to run from a specific project's venv (Recommended for `uv`)**

*   If `cursor-notebook-mcp` is an installed script in the `uv` environment:
    ```json
    {
      "mcpServers": {
        "notebook_mcp": {
          "command": "uv", 
          "args": [
            "--directory", "/absolute/path/to/your/project/with/uv/venv", 
            "run", "--", 
            "cursor-notebook-mcp", 
            "--allow-root", "/absolute/path/to/your/notebooks"
          ]
        }
      }
    }
    ```
*   Using `uv run` with `python -m ...`:
    ```json
    {
      "mcpServers": {
        "notebook_mcp": {
          "command": "uv", 
          "args": [
            "--directory", "/absolute/path/to/your/project/with/uv/venv", 
            "run", "--", 
            "python", "-m", "cursor_notebook_mcp.server",
            "--allow-root", "/absolute/path/to/your/notebooks"
          ]
        }
      }
    }
    ```
    *Note: For `uv` examples, provide the full path to `uv` if it's not on PATH. The `--directory` should be your project root where `uv` manages the environment.*

## Environment Management for Network Transports (User-managed)

For Streamable HTTP or SSE, always activate your virtual environment *before* manually launching the server:

```bash
# First activate your environment (example for bash/zsh)
source /path/to/venv/bin/activate 

# Then launch the server (example for Streamable HTTP)
cursor-notebook-mcp --transport streamable-http --host 127.0.0.1 --port 8080 --allow-root /path/to/notebooks
```

## Systemd Service (`cursor-notebook-mcp.service`)

If using the example `cursor-notebook-mcp.service` file (now configured for Streamable HTTP):

```ini
# In your systemd service file (examples/cursor-notebook-mcp.service):

# Option 1: Use direct path to the installed script in venv
ExecStart=/path/to/venv/bin/cursor-notebook-mcp --transport streamable-http --host 127.0.0.1 --port 8080 --allow-root /path/to/notebooks

# Option 2: Use Python from venv to run the module
# ExecStart=/path/to/venv/bin/python -m cursor_notebook_mcp.server --transport streamable-http --host 127.0.0.1 --port 8080 --allow-root /path/to/notebooks

# Option 3: Use bash to source the environment first
# ExecStart=/bin/bash -c 'source /path/to/venv/bin/activate && cursor-notebook-mcp --transport streamable-http --host 127.0.0.1 --port 8080 --allow-root /path/to/notebooks'
```
*Remember to replace placeholder paths and ensure correct user/group settings in the service file.*

## Verification

When configured correctly, you should see `notebook_mcp` listed in Cursor's MCP settings page under "Available Tools" after Cursor connects to or starts the server. 