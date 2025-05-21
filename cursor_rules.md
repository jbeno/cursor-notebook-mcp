### Jupyter Notebook Rules for Cursor (Using notebook_mcp):

1. **IMPORTANT - Markdown Formatting:**
   * When creating or editing markdown cells, use **actual newlines** for paragraph breaks, not literal `\n\n` strings.
   * CORRECT: `"## Title\n\nThis is a paragraph."`
   * INCORRECT: `"## Title\\n\\nThis is a paragraph."`
   * After editing cells, always use `notebook_read_cell` to verify proper formatting.

2. **Tool Usage:**
   * Always use `notebook_mcp` tools for `.ipynb` files, never `edit_file`.
   * Verify changes after making them with `notebook_read_cell` or `notebook_read`.

3. **Path Resolution for Notebooks:**
   * **Initial Step:** At the beginning of notebook operations, or if path ambiguity exists, call `notebook_get_server_path_context` (providing the current `project_directory`).
   * Use its output (`allowed_roots`, `server_path_style`, `project_directory_status`, `effective_notebook_base_path_for_project`, `path_construction_guidance`) to determine how to construct `notebook_path` arguments for all other notebook tools.
   * **Goal:** For unqualified notebook names (e.g., `my_notebook.ipynb`), the `notebook_path` sent to tools should correctly target the user's current project directory by leveraging the `effective_notebook_base_path_for_project` (e.g., `project_name/my_notebook.ipynb`).
   * If the `project_directory_status` is `outside_allowed_roots` or `resolution_error`, inform the user and proceed with caution, relying on their explicit path guidance or warning about potential issues.
   * For explicit user-provided paths (e.g., `../another_project/data.ipynb` or absolute paths), use them as given, but warn if they appear to be outside the server's `allowed_roots` based on the context tool's output.
   * The server ultimately resolves paths relative to its configured `allowed_root`(s). The context tool helps align client-side path construction with server-side expectations.

4. **Character Escaping:**
   * For LaTeX: Use single backslashes (e.g., `\alpha`, not `\\alpha`).
   * For newlines: Use actual newlines in the string, not escaped `\\n`.
   * For display math: Use `$$..$$` not `\\[..\]`.

5. **Investigation Before Editing:**
   * Use `notebook_get_outline` and `notebook_search` first to understand notebook structure.
   * Read existing cells with `notebook_read_cell` before modifying them.

6. **Available Tools by Category:**
   * **Path & Server Context**: `notebook_get_server_path_context`
   * **Navigation & Discovery**: `notebook_get_outline`, `notebook_search`, `notebook_get_info`, `notebook_get_cell_count`
   * **File Operations**: `notebook_create`, `notebook_delete`, `notebook_rename`, `notebook_read`, `notebook_export`
   * **Cell Operations**: `notebook_read_cell`, `notebook_add_cell`, `notebook_edit_cell`, `notebook_delete_cell`, `notebook_bulk_add_cells`
   * **Cell Transformations**: `notebook_change_cell_type`, `notebook_move_cell`, `notebook_split_cell`, `notebook_merge_cells`, `notebook_duplicate_cell`
   * **Metadata & Output**: `notebook_read_metadata`, `notebook_edit_metadata`, `notebook_read_cell_metadata`, `notebook_edit_cell_metadata`, `notebook_read_cell_output`, `notebook_edit_cell_output`, `notebook_clear_cell_outputs`, `notebook_clear_all_outputs`
   * **Validation**: `notebook_validate`

7. **Cell Magics & Rich Output:**
   * Use `!command` for shell commands (not `%%bash`).
   * Matplotlib, Pandas, and HTML in markdown render correctly.
   * Avoid `%%writefile` and similar unsupported magics.

8. **Visibility Mode (Less Waiting):**
   * Optimize for visibility and transparency during long edits
   * Create notebooks and make edits with incremental tool calls