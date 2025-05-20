# Comprehensive Test Plan for Jupyter Notebook MCP Tools (Model-Executable)

This document outlines a methodical approach for testing all `notebook_mcp` tools.

**Execution Context:**
- `{{PROJECT_DIR}}`: This placeholder represents the **current project's root directory** where the test is initiated. This directory itself **should NOT be created or deleted** by this test plan. 
- `{{TEST_SUBDIR}}`: A dedicated subdirectory, fixed as `"mcp_test_artifacts"`, will be created by this test plan inside `{{PROJECT_DIR}}` for all test files and operations. All relative paths in tool calls will be prefixed with `{{TEST_SUBDIR}}/`.

**Windows Note for PowerShell:** When running on Windows, use simpler PowerShell commands to avoid PSReadLine errors:
- Use `mkdir {{TEST_SUBDIR}} -Force` instead of complex conditionals
- Use `Remove-Item` without complex conditionals and quotes when possible
- Split operations into multiple commands for better reliability
- Path separators are `\` instead of `/` in PowerShell

## I. Prerequisites & Initial Setup

1.  **Environment:**
    *   Jupyter environment with `notebook_mcp` tools installed.
    *   **Local Testing:** MCP Server should be configured such that `{{PROJECT_DIR}}` is an allowed root and serves as the base for resolving relative paths like `{{TEST_SUBDIR}}/filename.ipynb`.
    *   **SFTP Testing:** SFTP connection established. The SFTP root for `SFTPManager` should correspond to `{{PROJECT_DIR}}/{{TEST_SUBDIR}}` on the remote server (e.g., if project is `/home/user/myproj` on remote, SFTP root is `/home/user/myproj/mcp_test_artifacts`).
    *   Permissions: Full read/write/delete/execute permissions for the user within `{{PROJECT_DIR}}/{{TEST_SUBDIR}}` (local and remote).
2.  **Initial Directory and File Setup:**
    *   **Instruction for Model:**
        1.  **Initial Cleanup (Important for reruns):**
            *   For Windows PowerShell: Use `mkdir {{TEST_SUBDIR}} -Force` which creates the directory if it doesn't exist or does nothing if it already exists.
            *   For complete cleanup when rerunning tests: Use `Remove-Item {{TEST_SUBDIR}} -Recurse -Force -ErrorAction SilentlyContinue` first. This requires splitting commands into separate steps to avoid complex conditionals.
        2.  Create directory `{{PROJECT_DIR}}/{{TEST_SUBDIR}}`. This operation should succeed.
            *   For Windows PowerShell: `mkdir {{TEST_SUBDIR}} -Force`
        3.  Create directory `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/subdir/`. This should also succeed.
            *   For Windows PowerShell: `mkdir {{TEST_SUBDIR}}\subdir -Force`

## II. Test Plan Execution

**Initial Server Context Check (Execute First):**
*   **Instruction for Model:** Call `notebook_get_server_path_context`, providing `{{PROJECT_DIR}}` as the `project_directory` argument. Log its full output (especially `allowed_roots`, `sftp_enabled`, `project_directory_status`, `effective_notebook_base_path_for_project`, and `path_construction_guidance`). This output is crucial for constructing all subsequent `notebook_path` arguments.

**Path Convention for Tool Calls:** Unless specified as an absolute path or a `~/` SFTP path, all relative notebook paths mentioned in this test plan (e.g., `empty_test_nb.ipynb` or `subdir/some_file.ipynb`) are assumed to be targets *within* the `{{TEST_SUBDIR}}` (i.e., `mcp_test_artifacts`). The model **must** construct the full `notebook_path` argument for tools based on the output of `notebook_get_server_path_context`:

1.  **Retrieve Context:** Use `sftp_enabled`, `project_directory_status`, and `effective_notebook_base_path_for_project` from the initial context check.
2.  **Determine Base Path for Tools:**
    *   **If `sftp_enabled` is `true` AND `project_directory_status` is `outside_allowed_roots` (common SFTP setup where `allowed_roots` are local temp paths for remote mounts):**
        The `base_path_for_tool` will be an empty string (`""`). Paths sent to tools will be relative to the SFTP root (e.g., `{{TEST_SUBDIR}}/filename.ipynb`).
    *   **Else (typically local operations or SFTP where project dir matches a direct remote root concept not yet fully captured by `effective_notebook_base_path_for_project` in this specific output):**
        Use the `effective_notebook_base_path_for_project` string. If it's `null` or not applicable from the guidance, and it's a local scenario, assume an empty string or the most logical prefix that targets `{{PROJECT_DIR}}` based on `allowed_roots`.
3.  **Construct Full Path:** The `notebook_path` for the tool is `base_path_for_tool` (if not empty, add a path separator) + `{{TEST_SUBDIR}}/` + `target_filename_or_subpath`.

    *Example (Local):* If `effective_notebook_base_path_for_project` is `my_project_folder` and test case is `some_test.ipynb`, `notebook_path` becomes `my_project_folder/mcp_test_artifacts/some_test.ipynb`.
    *Example (SFTP as per user provided output):* If `sftp_enabled` is `true`, `project_directory_status` is `outside_allowed_roots`, `effective_notebook_base_path_for_project` is `null`, and test case is `some_test.ipynb`, the `base_path_for_tool` is `""`, so `notebook_path` becomes `mcp_test_artifacts/some_test.ipynb`. The MCP server handles mapping this to the correct SFTP remote path.

**Verification Note for SFTP:** When `sftp_enabled` is true, verification of file existence and content should primarily use `notebook_mcp` tools (e.g., `notebook_read`, `notebook_get_cell_count`) as these operate through the server's SFTP layer. Direct local file system checks on `{{PROJECT_DIR}}` might not reflect the state on the remote SFTP server.
**Logging Note:** For handled errors (e.g., `FileNotFoundError`, `IndexError`), stack traces may only appear in server logs if the server log level is `DEBUG`. Standard `ERROR` level logs will show the error message without the full trace. Unexpected errors (e.g., from `logger.exception`) will typically include stack traces regardless of level.

### Section 0: Core Notebook Creation, Cell Management, and Output Tests

| Tool | Test Case Action (Model Instruction) | Expected Outcome (Verify After Action) |
|------|--------------------------------------|----------------------------------------|
| `notebook_create` | Create notebook `empty_test_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/empty_test_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. `notebook_read` shows empty cells array and default metadata. |
| `notebook_create` | Create notebook `simple_test_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/simple_test_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. `notebook_read` shows empty cells array and default metadata. |
| `notebook_add_cell` | To `simple_test_nb.ipynb`, add md cell "## Simple Title", `idx=-1` | Tool success. `notebook_get_cell_count` returns 1. `notebook_read_cell` (idx 0) returns "## Simple Title". `notebook_get_outline` for cell 0 shows H2. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `simple_test_nb.ipynb`, add code cell "print('Hello')", `idx=0` | Tool success. `notebook_get_cell_count` returns 2. `notebook_read_cell` (idx 1) returns "print('Hello')". `notebook_get_outline` for cell 1 shows context. `notebook_validate` reports valid. |
| `notebook_create` | Create notebook `complex_test_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/complex_test_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `complex_test_nb.ipynb`, add 5 cells (mix of md/code, e.g., C0: "# Title", C1: "print(1)", C2: "## Sub", C3: "import os", C4: "Final notes") | Tool success. `notebook_get_cell_count` returns 5. Verify content of each cell with `notebook_read_cell`. `notebook_validate` reports valid. |
| `notebook_create` | Create notebook `to_be_renamed_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_renamed_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `to_be_renamed_nb.ipynb`, add md cell "Rename Test Content", `idx=-1` | Tool success. `notebook_get_cell_count` returns 1. `notebook_read_cell` (idx 0) returns "Rename Test Content". `notebook_validate` reports valid. |
| `notebook_create` | Create notebook `to_be_deleted_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_deleted_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_create` | Create notebook `to_be_exported_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_exported_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `to_be_exported_nb.ipynb`, add code cell "a=1 # Export test content", `idx=-1` | Tool success. `notebook_get_cell_count` returns 1. `notebook_read_cell` (idx 0) returns "a=1 # Export test content". `notebook_validate` reports valid. |
| `notebook_create` | Create notebook `to_be_read_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_read_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `to_be_read_nb.ipynb`, add md cell "# Read Me Content", `idx=-1` | Tool success. `notebook_get_cell_count` returns 1. `notebook_read_cell` (idx 0) returns "# Read Me Content". `notebook_validate` reports valid. |
| **Output Test Notebook Setup** (Tests `notebook_edit_cell_output`) |
| `notebook_create` | Create `output_test_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/output_test_nb.ipynb` exists. `notebook_get_cell_count` returns 0. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `output_test_nb.ipynb`, add code cell (idx 0) `print("Placeholder for output1")` | Tool success. `notebook_get_cell_count` returns 1. `notebook_read_cell` (idx 0) returns "print(\"Placeholder for output1\")". `notebook_validate` reports valid. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 0, set outputs: `[{"output_type": "stream", "name": "stdout", "text": "output1\n"}]` | Tool success. `notebook_read_cell_output` (idx 0) returns the exact specified output (text is string, not list). `notebook_read` and check cell 0 `execution_count` is null/unchanged. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `output_test_nb.ipynb`, add code cell (idx 1) `1/0 # Placeholder for error` | Tool success. `notebook_get_cell_count` returns 2. `notebook_read_cell` (idx 1) returns "1/0 # Placeholder for error". `notebook_validate` reports valid. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 1, set outputs: `[{"output_type": "error", "ename": "SampleError", "evalue": "This is a sample error", "traceback": ["Traceback line 1"]}]` | Tool success. `notebook_read_cell_output` (idx 1) returns the exact specified error output. `notebook_read` and check cell 1 `execution_count` is null/unchanged. `notebook_validate` reports valid. |
| `notebook_add_cell` | To `output_test_nb.ipynb`, add code cell (idx 2) `pass # For adding output later` | Tool success. `notebook_get_cell_count` returns 3. `notebook_read_cell` (idx 2) returns "pass # For adding output later". `notebook_validate` reports valid. |
| `notebook_add_cell` | To `output_test_nb.ipynb`, add md cell (idx 3) `## Markdown Cell for Output Test` | Tool success. `notebook_get_cell_count` returns 4. `notebook_read_cell` (idx 3) returns "## Markdown Cell for Output Test". `notebook_validate` reports valid. |

**Bulk Cell Operations Test (`notebook_bulk_add_cells`)**
| Tool | Test Case Action (Model Instruction) | Expected Outcome (Verify After Action) |
|------|--------------------------------------|----------------------------------------|
| `notebook_create` | Create notebook `bulk_add_test_nb.ipynb` | Tool success. File exists. `notebook_get_cell_count` is 0. `notebook_validate` is valid. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=-1`, add cells: `[{"cell_type": "markdown", "source": "# Bulk Title"}, {"cell_type": "code", "source": "print('bulk code 1')"}, {"cell_type": "markdown", "source": "End of bulk."}]` | Tool success. `notebook_get_cell_count` is 3. `notebook_read_cell`(idx 0) is "# Bulk Title". `notebook_read_cell`(idx 1) is "print('bulk code 1')". `notebook_read_cell`(idx 2) is "End of bulk.". `notebook_validate` is valid. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=0`, add cells: `[{"cell_type": "code", "source": "inserted_code_cell"}]` | Tool success. `notebook_get_cell_count` is 4. Original cell 0 ("# Bulk Title") is still at idx 0. New cell "inserted_code_cell" is at idx 1. Original cell 1 ("print('bulk code 1')") is now at idx 2. Verify with `notebook_read_cell` for these indices. `notebook_validate` is valid. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=0`, add cells: `[]` (empty list) | Tool success (message should indicate no cells added or no change). `notebook_get_cell_count` remains 4. `notebook_validate` is valid. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=0`, add cells: `[{"cell_type": "invalid_type", "source": "test"}]` | Fails with `ValueError`. Verify error type. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=0`, add cells: `[{"cell_type": "code", "source": "test"}, {"source": "missing_type"}]` | Fails with `ValueError` (due to malformed cell data). Verify error type. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=99`, add cells: `[{"cell_type": "code", "source": "test"}]` | Fails with `IndexError`. Verify error type. |
| `notebook_bulk_add_cells` | To `bulk_add_test_nb.ipynb`, `insert_after_index=0`, add cells: `[{"cell_type": "code", "source": "[very_long_string_exceeding_max_source_size]"}]` (Model should construct such a string if feasible for testing, otherwise describe the intent) | Fails with `ValueError` (source size exceeded). Verify error type. |

### Section 1: Notebook Structure and Information

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_get_outline` | On `empty_test_nb.ipynb` | Returns structure indicating 0 cells or empty message (e.g., `[{"message": "Notebook is empty or has no cells"}]`). Verify with visual inspection of the returned structure. |
| `notebook_get_outline` | On `simple_test_nb.ipynb` | Returns structured outline (2 cells, e.g., Cell 0: H2: Simple Title, Cell 1: context: print('Hello')). Verify structure and content snippets. |
| `notebook_get_cell_count` | On `empty_test_nb.ipynb` | Returns 0. Verify value. |
| `notebook_get_cell_count` | On `simple_test_nb.ipynb` | Returns 2. Verify value. |
| `notebook_get_info` | On `empty_test_nb.ipynb` | Returns info, cell count 0. Verify cell count and default nbformat. |
| `notebook_get_info` | On `simple_test_nb.ipynb` | Returns info, cell count 2. Verify cell count and default nbformat. |
| `notebook_search` | On `simple_test_nb.ipynb`, query "NonExistentTerm" | Returns empty list `[]`. Verify. |
| `notebook_search` | On `simple_test_nb.ipynb`, query "Simple Title" | Returns list with 1 match (cell_index 0, cell_type markdown, snippet "## Simple Title"). Verify match details. |
| `notebook_search` | On `simple_test_nb.ipynb`, query "hello" (lowercase) | Returns list with 1 match (cell_index 1, cell_type code, snippet "print('Hello')"). Verify match details. |
| `notebook_search` | On `simple_test_nb.ipynb`, query "" (empty string) | Fails with `ValueError`. Verify error type. |
| `notebook_validate` | On `empty_test_nb.ipynb` | Returns "Notebook is valid." Verify message. |
| `notebook_validate` | On `simple_test_nb.ipynb` | Returns "Notebook is valid." Verify message. |
| `notebook_validate` | On `non_existent_notebook.ipynb` (Path becomes `{{TEST_SUBDIR}}/non_existent_notebook.ipynb`) | Fails with `FileNotFoundError`. Verify error type. |

### Section 2: Notebook Metadata Operations

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_read_metadata` | On `simple_test_nb.ipynb` | Returns default metadata (e.g., empty dict `{}` or dict with `kernelspec`, `language_info`). Verify content. |
| `notebook_edit_metadata` | On `simple_test_nb.ipynb`, add `{'author': 'AI Tester', 'version': '1.0'}` | Tool success. |
| `notebook_read_metadata` | On `simple_test_nb.ipynb` | Metadata includes *exactly* `{'author': 'AI Tester', 'version': '1.0'}` plus any other defaults. Verify all keys and values. |
| `notebook_edit_metadata` | On `simple_test_nb.ipynb`, add `{'author': 'AI Tester v2', 'status': 'testing'}` | Tool success. (This will merge/overwrite). |
| `notebook_read_metadata` | On `simple_test_nb.ipynb` | Metadata is *exactly* `{'author': 'AI Tester v2', 'version': '1.0', 'status': 'testing'}` plus any other defaults. Verify all keys/values carefully. |
| `notebook_edit_metadata` | On `simple_test_nb.ipynb`, set to an empty dict `{}` (to clear custom) | Tool success. |
| `notebook_read_metadata` | On `simple_test_nb.ipynb` | Metadata is reset to default notebook metadata (containing `kernelspec`, `language_info`, etc.); custom fields `author`, `version`, `status` are removed. Verify custom fields are gone and standard defaults are present. |

### Section 3: Cell Content Operations
*Notebook under test for this section: `complex_test_nb.ipynb` (starts with 5 cells from Section 0, e.g., C0: "# Title", C1: "print(1)", C2: "## Sub", C3: "import os", C4: "Final notes")*

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_read_cell` | On `complex_test_nb.ipynb`, cell_index 0 | Returns *exactly* "# Title". Verify. |
| `notebook_read_cell` | On `complex_test_nb.ipynb`, cell_index 4 | Returns *exactly* "Final notes". Verify. |
| `notebook_read_cell` | On `complex_test_nb.ipynb`, cell_index 99 | Fails with `IndexError`. Verify error type. |
| `notebook_edit_cell` | On `complex_test_nb.ipynb`, cell_index 1, source "New Content for Cell 1" | Tool success. |
| `notebook_read_cell` | On `complex_test_nb.ipynb`, cell_index 1 | Content is *exactly* "New Content for Cell 1". Verify. |
| `notebook_add_cell` | On `complex_test_nb.ipynb`, type 'markdown', source "### Added at Start", `insert_after_index=-1` | Tool success. `notebook_get_cell_count` is 6. `notebook_read_cell` (idx 0) is "### Added at Start". Original cell 0 ("# Title") is now at index 1. `notebook_validate` reports valid. |
| `notebook_add_cell` | On `complex_test_nb.ipynb`, type 'code', source "pass # Added after new cell 0", `insert_after_index=0` | Tool success. `notebook_get_cell_count` is 7. `notebook_read_cell` (idx 1) is "pass # Added after new cell 0". Original cell 0 ("### Added at Start") is still at index 0. Original cell 1 ("# Title") is now at index 2. `notebook_validate` reports valid. |
| `notebook_delete_cell` | On `complex_test_nb.ipynb`, cell_index 2 (this was "# Title") | Tool success. `notebook_get_cell_count` is 6. `notebook_read_cell` (idx 2) is now content of original C1 ("New Content for Cell 1"). `notebook_read_cell` with index 6 fails (`IndexError`). `notebook_validate` reports valid. |
| `notebook_delete_cell` | On `empty_test_nb.ipynb`, cell_index 0 | Fails with `IndexError`. Verify error type. |

### Section 4: Cell Metadata Operations
*Notebook under test: `complex_test_nb.ipynb` (currently 6 cells after Section 3 operations)*

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_read_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0 | Returns metadata (likely empty dict `{}` or default). Verify. |
| `notebook_edit_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0, updates `{'tags': ['header'], 'custom_id': 'cell0'}` | Tool success. |
| `notebook_read_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0 | Metadata includes *exactly* `{'tags': ['header'], 'custom_id': 'cell0'}` (plus any defaults). Verify all keys/values. |
| `notebook_edit_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0, updates `{'tags': ['updated'], 'new_key': 'new_val'}` | Tool success. |
| `notebook_read_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0 | Metadata includes `{'tags': ['updated'], 'custom_id': 'cell0', 'new_key': 'new_val'}`. Verify. |
| `notebook_edit_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0, updates `{}` (to clear custom) | Tool success. |
| `notebook_read_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 0 | Metadata is an empty dict `{}`. Custom fields are gone. Verify. |
| `notebook_edit_cell_metadata` | On `complex_test_nb.ipynb`, cell_index 99, updates `{}` | Fails with `IndexError`. Verify error type. |

### Section 5: Cell Type and Structure Operations
*Notebook under test: `complex_test_nb.ipynb` (currently 6 cells. State after Sec 4: C0="### Added at Start"(md), C1="pass # Added after new cell 0"(code), C2="New Content for Cell 1"(code), C3="## Sub"(md), C4="import os"(code), C5="Final notes"(md))*

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_change_cell_type` | On `complex_test_nb.ipynb`, cell_index 0 (md "### Added at Start"), new_type 'code'. (Read source before: S0) | Tool success. Use `notebook_read` to verify cell 0 type is 'code'. `notebook_read_cell` (idx 0) to verify source is S0. `notebook_validate` reports valid. |
| `notebook_change_cell_type` | On `complex_test_nb.ipynb`, cell_index 1 (code "pass # Added after new cell 0"), new_type 'raw'. (Read source before: S1) | Tool success. Use `notebook_read` to verify cell 1 type is 'raw'. `notebook_read_cell` (idx 1) to verify source is S1. `notebook_validate` reports valid. |
| `notebook_change_cell_type` | On `complex_test_nb.ipynb`, cell_index 0 (now code), new_type 'code' | Tool success. Message indicates no change needed. Verify type is still 'code' and source is S0. |
| `notebook_move_cell` | On `complex_test_nb.ipynb`, cell_index 0 (src S0, type code), new_index 1. (Cell 1 is S1, type raw) | Tool success. `notebook_read_cell` (idx 0) is S1. `notebook_read_cell` (idx 1) is S0. `notebook_read` to verify types also moved. `notebook_validate` reports valid. |
| `notebook_move_cell` | On `complex_test_nb.ipynb`, cell_index 5 (src "Final notes", type md), new_index 0. | Tool success. `notebook_read_cell` (idx 0) is "Final notes". Original cell 0 (S1) is now at index 1. Verify a few other cells shifted correctly. `notebook_validate` reports valid. |
| `notebook_duplicate_cell` | On `complex_test_nb.ipynb`, cell_index 0 (src "Final notes"), count 1 | Tool success. `notebook_get_cell_count` is 7. `notebook_read_cell` (idx 0) and (idx 1) are both "Final notes". `notebook_validate` reports valid. |
| `notebook_edit_cell` | Prerequisite for split: On `complex_test_nb.ipynb`, cell_index 2, new source: "Line1\nLine2\nLine3" | Tool success. `notebook_read_cell` (idx 2) is "Line1\nLine2\nLine3". |
| `notebook_split_cell` | On `complex_test_nb.ipynb`, cell_index 2 (source "Line1\nLine2\nLine3"), split_at_line 2 | Tool success. `notebook_get_cell_count` is 8. `notebook_read_cell` (idx 2) is "Line1\n". `notebook_read_cell` (idx 3) is "Line2\nLine3". `notebook_validate` reports valid. |
| `notebook_merge_cells` | On `complex_test_nb.ipynb`, first_cell_index 2 (merges current cell 2 "Line1\n" and cell 3 "Line2\nLine3") | Tool success. `notebook_get_cell_count` is 7. `notebook_read_cell` (idx 2) is "Line1\nLine2\nLine3". `notebook_validate` reports valid. |

### Section 6: Output Operations

**Note on Setup:** The following tests use `output_test_nb.ipynb`, prepared in Section 0. Its initial state after Section 0 is: 4 cells. Cell 0 (code) has stream output `output1\n`. Cell 1 (code) has error output `SampleError`. Cell 2 (code) `pass # For adding output later` has no output. Cell 3 is markdown `## Markdown Cell for Output Test`.

*Notebook under test: `output_test_nb.ipynb`*

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_read_cell_output` | On `output_test_nb.ipynb`, cell_idx 0 | Returns *exactly* `[{"output_type": "stream", "name": "stdout", "text": "output1\n"}]`. Verify structure and content. (May need fallback to `notebook_read` if tool unresponsive). |
| `notebook_read_cell_output` | On `output_test_nb.ipynb`, cell_idx 1 | Returns *exactly* `[{"output_type": "error", "ename": "SampleError", "evalue": "This is a sample error", "traceback": ["Traceback line 1"]}]`. Verify. (May need fallback to `notebook_read` if tool unresponsive). |
| `notebook_read_cell_output` | On `output_test_nb.ipynb`, cell_idx 2 (code cell, no output initially) | Returns empty list `[]`. Verify. (May need fallback to `notebook_read` if tool unresponsive). |
| `notebook_read_cell_output` | On `output_test_nb.ipynb`, cell_idx 3 (markdown cell) | Returns empty list `[]`. Verify. (May need fallback to `notebook_read` if tool unresponsive). |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 0, new outputs: `[{"output_type": "stream", "name": "stdout", "text": "new_output\n"}]` | Tool success. `notebook_read_cell_output` (idx 0) returns *exactly* the new stream output. `notebook_read` (for cell 0) and verify `execution_count` is null/unchanged. `notebook_validate` reports valid. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 1, new outputs: `[{"output_type": "stream", "name": "stdout", "text": "error replaced\n"}]` | Tool success. `notebook_read_cell_output` (idx 1) returns *exactly* the new stream output. `notebook_read` (for cell 1) and verify `execution_count` is null/unchanged. `notebook_validate` reports valid. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 2, new outputs: `[{"output_type": "display_data", "data": {"text/plain": "Hello"}, "metadata": {}}]` | Tool success. `notebook_read_cell_output` (idx 2) returns *exactly* the new display_data output (data value is string). `notebook_read` (for cell 2) and verify `execution_count` is null/unchanged. `notebook_validate` reports valid. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 3 (markdown), outputs: `[{"output_type": "stream", "name": "stdout", "text": "md output attempt"}]` | Fails with `ValueError` (cannot set output on non-code cell). Verify error type. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 0, outputs: `{"output_type": "stream"}` (not a list) | Fails with `ValueError` (outputs parameter must be a list). Verify error type. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 0, outputs: `[{"type": "stream"}]` (malformed dict) | Fails with `ValueError` (output item missing 'output_type' or other required fields). Verify error type. |
| `notebook_edit_cell_output` | On `output_test_nb.ipynb`, cell_idx 0, outputs with very large string to exceed `max_cell_output_size`. | Fails with `ValueError` (output data size exceeded). Verify error type. |
| `notebook_clear_cell_outputs` | On `output_test_nb.ipynb`, cell_idx 0 (has output from previous edit: "new_output\n") | Tool success. `notebook_read_cell_output` on cell 0 returns `[]`. `notebook_read` (for cell 0) and verify `execution_count` is null. `notebook_validate` reports valid. |
| `notebook_clear_all_outputs` | On `output_test_nb.ipynb` (cells 1, 2 still have outputs if previous tests passed) | Tool success. `notebook_read_cell_output` on cells 1 & 2 returns `[]`. `notebook_read` (for cells 1 & 2) and verify `execution_count` is null. `notebook_validate` reports valid. |

### Section 7: File Operations

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| **`notebook_create`** | Create `created_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/created_nb.ipynb` exists. `notebook_get_cell_count` is 0. `notebook_validate` reports valid. `notebook_read` shows empty notebook structure. |
|  | Create `created_nb.ipynb` again. | Fails with `FileExistsError`. Verify error type. |
|  | Create `subdir/created_in_subdir.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/subdir/created_in_subdir.ipynb` exists. `notebook_get_cell_count` is 0. `notebook_validate` reports valid. |
| **`notebook_read`** | Read `to_be_read_nb.ipynb` (contains md cell "# Read Me Content") | Tool success. Returns dict. Verify `cells[0].cell_type` is 'markdown' and `cells[0].source` is *exactly* "# Read Me Content". Verify `nbformat` and `metadata` presence. |
|  | Read `non_existent_notebook.ipynb` | Fails with `FileNotFoundError`. Verify error type. |
| **`notebook_rename`** | Rename `to_be_renamed_nb.ipynb` (contains md cell "Rename Test Content") to `renamed_nb.ipynb` | Tool success. `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_renamed_nb.ipynb` does not exist. `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/renamed_nb.ipynb` exists. `notebook_read` on `renamed_nb.ipynb` shows cell 0 content is "Rename Test Content". `notebook_validate` on `renamed_nb.ipynb` is valid. |
|  | Rename `non_existent_notebook.ipynb` to `new_name_fail.ipynb` | Fails with `FileNotFoundError`. Verify error type. |
|  | Create `temp_exists.ipynb`. Rename `renamed_nb.ipynb` to `temp_exists.ipynb` | Fails with `FileExistsError`. Verify error type. Delete `temp_exists.ipynb`. |
| **`notebook_delete`** | Delete `to_be_deleted_nb.ipynb` | Tool success. File `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_deleted_nb.ipynb` no longer exists. `notebook_read` on it fails with `FileNotFoundError`. |
|  | Delete `to_be_deleted_nb.ipynb` again. | Fails with `FileNotFoundError`. Verify error type. (Tool pre-checks existence and raises error if not found). |
| **`notebook_export`** | Export `to_be_exported_nb.ipynb` (contains code cell "a=1 # Export test content") to format `python`. Store result path. | Tool success. Result path (local temp on MCP server or `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/...` for SFTP) ends with `.py`. `read_file` on result path shows valid Python code, e.g., "a=1 # Export test content" (and potentially boilerplate). **If SFTP, verify `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_exported_nb.py` also exists on remote and its content is correct.** |
|  | Export `to_be_exported_nb.ipynb` to format `html`. Store result path. | Tool success. Result path ends with `.html`. `read_file` on result path shows valid HTML structure (e.g., starts with `<!DOCTYPE html>`). **If SFTP, verify `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/to_be_exported_nb.html` also exists on remote and its content is correct.** |
|  | Export `non_existent_notebook.ipynb` to `python`. | Fails with `FileNotFoundError`. Verify error type. |

### Section 8: Path Variation Specific Tests (Focus on relative paths within `{{TEST_SUBDIR}}`)
*These paths are relative to `{{PROJECT_DIR}}` when calling tools, e.g., `{{TEST_SUBDIR}}/new_relative.ipynb`. The server resolves these based on its configuration.* 

| Tool | Test Case Action | Expected Outcome |
|---|---|---|
| `notebook_create` | Create `new_relative.ipynb` | Tool success. `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/new_relative.ipynb` exists. `notebook_get_cell_count` is 0. `notebook_validate` is valid. |
| `notebook_read` | Read `new_relative.ipynb` | Tool success. Verify `notebook_read` shows empty notebook structure. |
| `notebook_add_cell` | To `new_relative.ipynb`, add code cell "print('Relative Path Test')" | Tool success. `notebook_get_cell_count` is 1. `notebook_read_cell` (idx 0) is "print('Relative Path Test')". |
| `notebook_rename` | Rename `new_relative.ipynb` to `new_relative_renamed.ipynb` | Tool success. `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/new_relative_renamed.ipynb` exists. `notebook_read` on new path shows cell 0 content "print('Relative Path Test')". Old path does not exist. |
| `notebook_delete` | Delete `new_relative_renamed.ipynb` | Tool success. File no longer exists. `notebook_read` on it fails. |
| `notebook_create` | Create `subdir/another_relative.ipynb` | Tool success. `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/subdir/another_relative.ipynb` exists. `notebook_get_cell_count` is 0. `notebook_validate` is valid. |
| `notebook_delete` | Delete `subdir/another_relative.ipynb` | Tool success. File no longer exists. |
| **SFTP Only (if applicable & `~` maps outside `{{PROJECT_DIR}}/{{TEST_SUBDIR}}`)** |
| `notebook_create` | Create `~/sftp_home_test_nb.ipynb` (Path is absolute `~` path, not prefixed by `{{TEST_SUBDIR}}`) | Fails with `PermissionError` if `~` resolves outside the configured SFTP root (e.g., `jim@kitsune:~/git/`). Verify error. Success only if `~` is allowed and writable. |
| `notebook_add_cell` | To `~/sftp_home_test_nb.ipynb`, add md cell "SFTP Home Test" | Skipped if create failed. If create succeeded, then: Tool success. `notebook_get_cell_count` is 1. `notebook_read_cell` (idx 0) is "SFTP Home Test". |
| `notebook_delete` | Delete `~/sftp_home_test_nb.ipynb` | Skipped if create failed. If create succeeded, then: Tool success. File no longer exists at resolved tilde path. |


## III. Test Execution Notes

*   **Verification:** For all test cases, the "Expected Outcome" column lists key verification points. The model executing this plan **must** use appropriate `notebook_read_*`, `notebook_get_*`, or `read_file` (for exported files) tool calls to actively confirm these outcomes. Do not rely solely on a "success" message from the tool being tested. Explicitly check cell contents, notebook structures, file contents, metadata, and error types as specified.
*   Execute sections/tests in order.
*   Record exact tool calls, parameters, and full error messages if any.
*   Log environment: local or SFTP. `{{PROJECT_DIR}}` is project root. `{{TEST_SUBDIR}}` is `mcp_test_artifacts` within it.
*   **For output tests in Section 6:** The `notebook_edit_cell_output` tool is used to set up initial states. This tool modifies the `outputs` array of cells but does not affect `execution_count`.

## IV. Results Documentation (Template)

'''
### Test Run Environment:
- Date: YYYY-MM-DD
- MCP Version: (if known)
- Execution Mode: Local / SFTP (details: SFTP root maps to `{{PROJECT_DIR}}/{{TEST_SUBDIR}}` on remote, e.g. `user@host:{{PROJECT_DIR_ON_REMOTE}}/{{TEST_SUBDIR}}`)
- `{{PROJECT_DIR}}` (Project Root on machine running model) used: [Actual Path]
- `{{TEST_SUBDIR}}` (Test Artifacts subdir name): `mcp_test_artifacts`

### Section X: Section Name
| Tool | Test Case Action | Parameters (actual path sent to tool) | Actual Outcome | Pass/Fail | Notes |
|------|------------------|---------------------------------------|----------------|-----------|-------|
|      |                  |                                       |                |           |       |
'''

## V. Environment Diagnostics (If Issues)

*   Check file system permissions and paths manually if discrepancies arise.
*   Review full MCP server logs.

## VI. Test Cleanup (Final)

*   **Instruction for Model:**
    1.  List contents of `{{PROJECT_DIR}}/{{TEST_SUBDIR}}`.
    2.  Delete each file individually:
        * For `.ipynb` files: Use `notebook_delete` for each file
        * For `.py` and `.html` files: Use `delete_file` for each file
*   **Instruction for Model:**
    1.  List contents of `{{PROJECT_DIR}}/{{TEST_SUBDIR}}/subdir/`.
    2.  Delete each file individually in the subdirectory.
*   **Instruction for Model:**
    1.  Check if subdir exists and is empty.
    2.  If confirmed empty, use `Remove-Item {{TEST_SUBDIR}}\subdir -Force` for Windows PowerShell.
*   **Instruction for Model:**
    1.  Check if `{{TEST_SUBDIR}}` exists and is empty.
    2.  If confirmed empty, use `Remove-Item {{TEST_SUBDIR}} -Force` for Windows PowerShell.
*   **Instruction for Model (Cleanup of SFTP home test file):**
    *   For Windows PowerShell: `Remove-Item $HOME\sftp_home_test_nb.ipynb -Force -ErrorAction SilentlyContinue`
*   **Note:** For SFTP exports, the local temporary files on the *MCP server machine* are cleaned up by the `notebook_export` tool itself. This cleanup focuses on artifacts created within the defined test directory (`{{PROJECT_DIR}}/{{TEST_SUBDIR}}`) on the local filesystem or remote SFTP server. 