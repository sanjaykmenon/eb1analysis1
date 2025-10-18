#!/usr/bin/env python3
"""
Quick test script to verify MCP server functionality
"""
import json
import subprocess
import sys

def test_mcp_server():
    """Send a test request to the MCP server"""
    
    # Initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    # List tools request
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    # Start the MCP server
    proc = subprocess.Popen(
        ["uv", "run", "aao", "mcp-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Send initialize
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()
        
        # Read response
        response = proc.stdout.readline()
        print("Initialize response:", response)
        
        # Send list tools
        proc.stdin.write(json.dumps(list_tools_request) + "\n")
        proc.stdin.flush()
        
        # Read response
        response = proc.stdout.readline()
        print("\nAvailable tools:")
        tools_data = json.loads(response)
        if "result" in tools_data and "tools" in tools_data["result"]:
            for tool in tools_data["result"]["tools"]:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        else:
            print("Tools response:", response)
            
    except Exception as e:
        print(f"Error: {e}")
        stderr = proc.stderr.read()
        if stderr:
            print(f"Server stderr: {stderr}")
    finally:
        proc.terminate()
        proc.wait(timeout=2)

if __name__ == "__main__":
    print("Testing AAO ETL MCP Server...\n")
    test_mcp_server()
