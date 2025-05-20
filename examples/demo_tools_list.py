import asyncio
import httpx
import json
import uuid # For generating unique IDs

MCP_ENDPOINT_URL = "http://127.0.0.1:8080/mcp/" # Adjust if your host/port is different

async def demonstrate_mcp_tools_list():
    session_id = None
    initialize_request_id = str(uuid.uuid4())
    tools_list_request_id = str(uuid.uuid4())

    default_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Send initialize request
        initialize_request_payload = {
            "jsonrpc": "2.0",
            "id": initialize_request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26", # Use current spec version
                "capabilities": {},
                "clientInfo": {
                    "name": "TestScriptClient",
                    "version": "0.1.0"
                }
            }
        }
        print(f"\n--- 1. Sending Initialize Request ---")
        print(json.dumps(initialize_request_payload, indent=2))

        try:
            async with client.stream("POST", MCP_ENDPOINT_URL, json=initialize_request_payload, headers=default_headers) as response:
                print(f"Initialize Response Status: {response.status_code}")
                print(f"Initialize Response Headers: {response.headers}")
                session_id = response.headers.get("mcp-session-id")
                print(f"Extracted Mcp-Session-Id: {session_id}")

                if not session_id:
                    print("ERROR: Mcp-Session-Id not found in initialize response headers!")
                    # Try to read body for error details anyway
                    body_content = await response.aread()
                    print(f"Initialize Response Body: {body_content.decode(errors='ignore')}")
                    return

                # Process SSE stream for InitializeResult
                async for line in response.aiter_lines():
                    print(f"SSE (Init): {line}")
                    if line.startswith("data:"):
                        data_content = line[len("data:"):].strip()
                        if not data_content: continue
                        try:
                            rpc_response = json.loads(data_content)
                            if rpc_response.get("id") == initialize_request_id and "result" in rpc_response:
                                print(f"Initialize Succeeded: {json.dumps(rpc_response['result'], indent=2)}")
                                break # Found InitializeResult
                            elif rpc_response.get("id") == initialize_request_id and "error" in rpc_response:
                                print(f"Initialize Failed (RPC Error): {json.dumps(rpc_response['error'], indent=2)}")
                                return
                        except json.JSONDecodeError:
                            print(f"Could not decode JSON from init data line: {data_content}")
                print("--- Finished processing Initialize response stream ---")

        except httpx.RequestError as exc:
            print(f"Initialize Request Error: {exc}")
            return
        except Exception as e:
            print(f"Unexpected error during Initialize: {e}")
            return

        if not session_id:
            print("Critical Error: Session ID was not obtained. Cannot proceed.")
            return

        # 2. Send initialized notification
        initialized_notification_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        headers_with_session = {**default_headers, "Mcp-Session-Id": session_id}
        
        print(f"\n--- 2. Sending Initialized Notification ---")
        print(json.dumps(initialized_notification_payload, indent=2))
        print(f"Headers: {headers_with_session}")

        try:
            response = await client.post(MCP_ENDPOINT_URL, json=initialized_notification_payload, headers=headers_with_session)
            print(f"Initialized Notification Response Status: {response.status_code}") # Expect 202
            if response.status_code != 202:
                print(f"WARN: Expected 202 for initialized notification, got {response.status_code}")
                # Try to read body for error details if any
                body_content = await response.aread()
                print(f"Initialized Notification Response Body: {body_content.decode(errors='ignore')}")
        except httpx.RequestError as exc:
            print(f"Initialized Notification Request Error: {exc}")
            return
        except Exception as e:
            print(f"Unexpected error during Initialized Notification: {e}")
            return

        # 3. Send tools/list request
        tools_list_request_payload = {
            "jsonrpc": "2.0",
            "id": tools_list_request_id,
            "method": "tools/list",
            "params": {}
        }
        print(f"\n--- 3. Sending tools/list Request ---")
        print(json.dumps(tools_list_request_payload, indent=2))
        print(f"Headers: {headers_with_session}")

        try:
            async with client.stream("POST", MCP_ENDPOINT_URL, json=tools_list_request_payload, headers=headers_with_session) as response:
                print(f"tools/list Response Status: {response.status_code}")
                print(f"tools/list Response Headers: {response.headers}")
                print("Streaming tools/list response content:\n")

                if response.headers.get("content-type", "").startswith("text/event-stream"):
                    async for line in response.aiter_lines():
                        print(f"SSE (tools/list): {line}")
                        if line.startswith("data:"):
                            data_content = line[len("data:"):].strip()
                            if not data_content: continue
                            try:
                                rpc_response = json.loads(data_content)
                                if rpc_response.get("id") == tools_list_request_id and "result" in rpc_response:
                                    if "tools" in rpc_response["result"]:
                                        print("\n" + "="*10 + " Successfully extracted tools list: " + "="*10)
                                        print(json.dumps(rpc_response["result"]["tools"], indent=2))
                                        print("="*40)
                                        return # Success!
                                    else:
                                        print("tools/list Response Error: 'tools' key missing in result.")
                                elif rpc_response.get("id") == tools_list_request_id and "error" in rpc_response:
                                    print("\n" + "!"*10 + " tools/list JSON-RPC Error: " + "!"*10)
                                    print(json.dumps(rpc_response["error"], indent=2))
                                    print("!"*40)
                                    return
                            except json.JSONDecodeError:
                                print(f"Could not decode JSON from tools/list data line: {data_content}")
                            except Exception as e:
                                print(f"Error processing tools/list SSE data: {e}")
                    print("--- Finished processing tools/list response stream ---")
                elif response.headers.get("content-type", "").startswith("application/json"):
                    print("Received direct JSON response for tools/list:")
                    body_content = await response.aread()
                    try:
                        rpc_response_json = json.loads(body_content)
                        print(json.dumps(rpc_response_json, indent=2))
                        if rpc_response_json.get("id") == tools_list_request_id and "result" in rpc_response_json and "tools" in rpc_response_json["result"]:
                             print("\n" + "="*10 + " Successfully extracted tools list: " + "="*10)
                             print(json.dumps(rpc_response_json["result"]["tools"], indent=2))
                             print("="*40)
                        elif rpc_response_json.get("id") == tools_list_request_id and "error" in rpc_response_json:
                            print("\n" + "!"*10 + " tools/list JSON-RPC Error: " + "!"*10)
                            print(json.dumps(rpc_response_json["error"], indent=2))
                            print("!"*40)
                    except json.JSONDecodeError:
                        print(f"Could not decode JSON from tools/list response: {body_content.decode(errors='ignore')}")
                    except Exception as e:
                        print(f"Error processing tools/list JSON response: {e}")
                else:
                    print(f"tools/list: Unexpected Content-Type: {response.headers.get('content-type')}")
                    full_content = await response.aread()
                    print(f"tools/list: Full response content: {full_content.decode(errors='ignore')}")

        except httpx.RequestError as exc:
            print(f"tools/list Request Error: {exc}")
        except Exception as e:
            print(f"Unexpected error during tools/list: {e}")

if __name__ == "__main__":
    asyncio.run(demonstrate_mcp_tools_list())