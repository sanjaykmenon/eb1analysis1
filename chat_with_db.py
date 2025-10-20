#!/usr/bin/env python3
"""
Simple MCP client to chat with your database using any LLM provider.

This script connects to the AAO ETL MCP server and lets you query your database
using natural language. Works with OpenAI, Together AI, Anthropic, or any other provider.

Usage:
    python chat_with_db.py
    
Environment variables:
    OPENAI_API_KEY - for OpenAI models
    TOGETHER_API_KEY - for Together AI models
    ANTHROPIC_API_KEY - for Anthropic models
"""

import asyncio
import json
import os
from typing import Optional

# Try to import MCP client libraries
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP client libraries not found. Installing...")
    import subprocess
    subprocess.check_call(["uv", "pip", "install", "mcp"])
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

# Try to import LLM libraries
try:
    from openai import OpenAI
except ImportError:
    print("Installing OpenAI library...")
    import subprocess
    subprocess.check_call(["uv", "pip", "install", "openai"])
    from openai import OpenAI


class DatabaseChat:
    """Chat interface to query AAO ETL database using MCP tools"""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize chat client
        
        Args:
            provider: "openai", "together", "anthropic"
            model: Model name (uses sensible defaults if not specified)
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.client = self._init_llm_client()
        self.conversation_history = []
        
    def _get_default_model(self) -> str:
        """Get default model for the provider"""
        defaults = {
            "openai": "gpt-4o-mini",
            "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "anthropic": "claude-3-5-sonnet-20241022"
        }
        return defaults.get(self.provider, "gpt-4o-mini")
    
    def _init_llm_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAI(api_key=api_key)
            
        elif self.provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            # Together AI uses OpenAI-compatible API
            return OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"
            )
            
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                import subprocess
                subprocess.check_call(["uv", "pip", "install", "anthropic"])
                from anthropic import Anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return Anthropic(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def connect_to_mcp(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "aao", "mcp-server"],
            env=dict(os.environ)  # Pass through DATABASE_URL
        )
        
        # Connect to MCP server
        self.stdio_context = stdio_client(server_params)
        self.read, self.write = await self.stdio_context.__aenter__()
        self.session_context = ClientSession(self.read, self.write)
        self.session = await self.session_context.__aenter__()
        await self.session.initialize()
        
        # Get available tools
        tools_result = await self.session.list_tools()
        self.tools = {tool.name: tool for tool in tools_result.tools}
        
        print(f"✅ Connected to MCP server with {len(self.tools)} tools:")
        for tool_name, tool in self.tools.items():
            print(f"   - {tool_name}")
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if hasattr(self, 'session_context'):
            await self.session_context.__aexit__(None, None, None)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(None, None, None)
    
    def _tools_to_openai_format(self):
        """Convert MCP tools to OpenAI function calling format"""
        functions = []
        for tool_name, tool in self.tools.items():
            # FastMCP wraps Pydantic models in an 'args' parameter
            # We need to unwrap this for OpenAI function calling
            schema = tool.inputSchema
            
            # If schema has an 'args' wrapper with a $ref, unwrap it
            if (schema and 
                'properties' in schema and 
                'args' in schema['properties'] and 
                '$ref' in schema['properties']['args']):
                
                # Get the actual parameter schema from $defs
                ref_path = schema['properties']['args']['$ref']  # e.g., '#/$defs/SearchDecisionsArgs'
                def_name = ref_path.split('/')[-1]
                
                if '$defs' in schema and def_name in schema['$defs']:
                    # Use the unwrapped schema
                    parameters = schema['$defs'][def_name]
                else:
                    parameters = schema
            else:
                parameters = schema
            
            func = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.description or f"Call {tool_name}",
                    "parameters": parameters
                }
            }
            functions.append(func)
        return functions
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the result"""
        try:
            # OpenAI sends flat arguments, but FastMCP expects them wrapped in 'args'
            # Check if the tool schema requires an 'args' wrapper
            tool_schema = self.tools[tool_name].inputSchema
            if (tool_schema and 
                'properties' in tool_schema and 
                'args' in tool_schema['properties']):
                # Wrap the arguments
                wrapped_args = {"args": arguments}
            else:
                # No wrapping needed
                wrapped_args = arguments
            
            result = await self.session.call_tool(tool_name, arguments=wrapped_args)
            # Extract text content from result
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list):
                    return "\n".join(str(item.text) if hasattr(item, 'text') else str(item) 
                                   for item in result.content)
                return str(result.content)
            return str(result)
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
    
    async def chat(self, user_message: str) -> str:
        """Send a message and get a response, using tools as needed"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that can query an AAO EB-1A immigration decisions database.
You have access to MCP tools to search and analyze the database.

Available tools:
- describe_schema: Show database structure
- search_decisions: Find decisions (returns case_number, decision_id, outcome, field, summary)
- get_decision_details: Get FULL details including criteria findings and evidence breakdown
- summarize_criterion_insights: Analyze denial/success patterns for a criterion

CRITICAL INSTRUCTIONS:
1. search_decisions returns overview information only (case numbers, dates, outcomes)
2. To get evidence items, criteria findings, or quotes, you MUST call get_decision_details with the decision_id or case_number
3. Always show case_number and decision_id in your responses so users can verify in the database
4. DO NOT make up or invent data - only use what the tools return
5. After calling tools, ALWAYS present the results in a clear, formatted way
6. If asked for "evidence items" or "detailed evidence", you MUST call get_decision_details for EACH case

WORKFLOW for evidence requests:
1. Use search_decisions to find relevant cases -> Get case numbers and IDs
2. For EACH case, call get_decision_details to get evidence breakdown
3. Present ALL the evidence data including case numbers for verification

IMPORTANT: Do not just say "I've retrieved the information" - actually show the data from the tool results!"""
            }
        ] + self.conversation_history
        
        # Call LLM with tools
        tools = self._tools_to_openai_format()
        
        # Together AI requires different parameters for function calling
        if self.provider == "together":
            # Together AI uses a simpler approach - no tool_choice parameter
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=1,
                max_completion_tokens=20000  # Updated for newer OpenAI models
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        
        message = response.choices[0].message
        
        # Check if LLM wants to call tools
        if message.tool_calls:
            # Execute tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content or "",  # Use empty string instead of None
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            tool_results = []
            # Call each tool
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Calling {func_name}({json.dumps(args, indent=2)})")
                
                result = await self.call_tool(func_name, args)
                tool_results.append(result)
                
                # DEBUG: Show actual tool result
                print(f"\n📊 Tool Result Preview (first 500 chars):\n{result[:3500]}...\n")
                
                # Add tool result to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            print(f"💭 Tool calls complete. Asking LLM to synthesize {len(tool_results)} result(s)...")
            
            # Get final response from LLM with tool results
            if self.provider == "together":
                # Together AI doesn't handle tool role well, so convert to user/assistant format
                clean_history = []
                for msg in self.conversation_history:
                    if msg["role"] == "tool":
                        # Convert tool results to assistant message
                        clean_history.append({
                            "role": "assistant",
                            "content": f"Tool result: {msg['content']}"
                        })
                    elif msg["role"] == "assistant" and "tool_calls" in msg:
                        # Skip the assistant message with tool_calls
                        continue
                    else:
                        clean_history.append(msg)
                
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[messages[0]] + clean_history + [{
                        "role": "user", 
                        "content": "Please present the tool results in a clear, formatted way with all the details."
                    }],
                    temperature=1.0,
                    max_completion_tokens=2000  # Updated for newer OpenAI models
                )
            else:
                # For OpenAI, add explicit request to present the data
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + self.conversation_history[len(self.conversation_history) - len(tool_results)*2:] + [{
                        "role": "user",
                        "content": "Please present the tool results above in a clear, formatted way showing all the data including case numbers."
                    }],
                    temperature=1.0,
                    max_completion_tokens=2000  # Updated for newer OpenAI models
                )
            
            final_message = final_response.choices[0].message
            # Handle None or empty content
            content = final_message.content or "I've retrieved the information from the database."
            
            # DEBUG: Check if content is actually present
            if not final_message.content:
                print(f"⚠️  Warning: LLM returned empty content after tool calls")
                print(f"   Tool results were: {[r[:100] for r in tool_results]}")
            
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            
            return content
        
        else:
            # No tool calls, just return the response
            content = message.content or "I understand."
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            return content


async def main():
    """Main interactive chat loop"""
    import sys
    
    print("=" * 60)
    print("🤖 AAO ETL Database Chat")
    print("=" * 60)
    
    # Check for API keys
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("TOGETHER_API_KEY"):
        providers.append("together")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    if not providers:
        print("\n❌ No API keys found!")
        print("Set one of these environment variables:")
        print("  - OPENAI_API_KEY for OpenAI models")
        print("  - TOGETHER_API_KEY for Together AI models")
        print("  - ANTHROPIC_API_KEY for Anthropic models")
        sys.exit(1)
    
    # Let user choose provider
    if len(providers) > 1:
        print(f"\nAvailable providers: {', '.join(providers)}")
        provider_input = input(f"Choose provider [{providers[0]}]: ").strip()
        # Validate that input is actually a provider name, not a number
        if provider_input in providers:
            provider = provider_input
        elif not provider_input:  # Empty input = use default
            provider = providers[0]
        else:
            print(f"⚠️  Invalid provider '{provider_input}', using default: {providers[0]}")
            provider = providers[0]
    else:
        provider = providers[0]
    
    # Let user optionally choose a different model
    model = None
    if os.getenv("MODEL"):
        model = os.getenv("MODEL")
        print(f"\n🎯 Using model from environment: {model}")
    else:
        # Show available models for the provider
        openai_models = [
            "gpt-4o-mini",  # Default - fast and cheap
            "gpt-4o",  # Balanced capability
            "gpt-4-turbo",  # Previous generation
            "o1",  # Reasoning model
            "o1-mini",  # Faster reasoning
            "o3-mini", # Latest reasoning (if available)
            "gpt-5-mini-2025-08-07"  
        ]
        
        together_models = [
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # Default
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Larger, better reasoning
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Newer version
            "Qwen/Qwen2.5-72B-Instruct-Turbo",  # Strong alternative
            "deepseek-ai/DeepSeek-V3",  # Excellent reasoning
        ]
        
        if provider == "openai":
            print(f"\n📋 Available OpenAI models:")
            for i, m in enumerate(openai_models, 1):
                default_marker = " (default)" if i == 1 else ""
                reasoning_marker = " [reasoning model]" if m.startswith("o") else ""
                print(f"  {i}. {m}{default_marker}{reasoning_marker}")
            choice = input(f"\nChoose model [1]: ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(openai_models):
                model = openai_models[int(choice) - 1]
        
        elif provider == "together":
            print(f"\n📋 Available Together AI models:")
            for i, m in enumerate(together_models, 1):
                default_marker = " (default)" if i == 1 else ""
                print(f"  {i}. {m}{default_marker}")
            choice = input(f"\nChoose model [1]: ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(together_models):
                model = together_models[int(choice) - 1]
    
    print(f"\n�🚀 Starting chat with {provider}...")
    if model:
        print(f"   Model: {model}")
    
    # Initialize chat
    chat = DatabaseChat(provider=provider, model=model)
    
    # Connect to MCP server
    print("📡 Connecting to AAO ETL MCP server...")
    await chat.connect_to_mcp()
    
    print("\n" + "=" * 60)
    print("💬 Chat started! Type your questions or 'exit' to quit.")
    print("=" * 60)
    print("\nExample queries:")
    print("  - What's in the database schema?")
    print("  - Find the last 5 denied cases")
    print("  - What are common ORIGINAL_CONTRIBUTION denial reasons?")
    print("  - Search for decisions about machine learning")
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            print("\n🤔 Thinking...")
            response = await chat.chat(user_input)
            
            print(f"\n🤖 Assistant:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Error: {error_msg}")
            
            # Give helpful hints for common errors
            if "400" in error_msg and "validation" in error_msg:
                print("\n💡 Tip: Together AI has some limitations with function calling.")
                print("   Try rephrasing your question or use a simpler query.")
            elif "rate limit" in error_msg.lower():
                print("\n💡 Tip: Rate limit exceeded. Wait a moment and try again.")
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                print("\n💡 Tip: Check that your API key is set correctly.")
                print(f"   Current provider: {chat.provider}")
            
            # Only show full traceback in debug mode
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()
    
    # Clean up
    print("\n🔌 Disconnecting from MCP server...")
    await chat.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
