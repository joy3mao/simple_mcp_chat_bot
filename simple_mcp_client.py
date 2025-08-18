# coding:utf-8
import asyncio
import json
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit.completion import WordCompleter
from rich.box import Box,ROUNDED
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any
import hashlib
import uuid
import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult,TextContent,PromptArgument,GetPromptResult,PromptMessage,TextContent
import traceback

class NoSlideBox(Box):
    def __init__(self,):
        super().__init__(
                "╭─┬╮\n"
                "    \n"
                "├─┼┤\n"
                "    \n"
                "├─┼┤\n"
                "├─┼┤\n"
                "    \n"
                "╰─┴╯\n"
        )

# 配置Console
console = Console(
    color_system="auto",
)
error_console = Console(
    stderr=True,
    style="bold red",
)
server_console = Console(
    style="dim blue", 
)

client_models = {
    "1":{
        "ai_channel":"OpenAI",
        "ai_model":"gpt-3.5-turbo",
        "ai_api_url":"https://api.openai.com/v1/chat/completions",
        "ai_provider":"OpenAI"
    },
    "2":{
        "ai_channel":"OpenAI",
        "ai_model":"gpt-4o",
        "ai_api_url":"https://api.openai.com/v1/chat/completions",
        "ai_provider":"OpenAI"
    },
	"3": {
        "ai_channel":"OpenAI",
        "ai_model":"gpt-4",
        "ai_api_url":"https://api.openai.com/v1/chat/completions",
        "ai_provider":"OpenAI"
    },
    "4": {
        "ai_channel":"Deepseek",
        "ai_model":"deepseek-chat",
        "ai_api_url":"https://api.deepseek.com/v1/chat/completions",
        "ai_provider":"Deepseek"
    }
}

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self):
        """Initialize configuration with environment variables."""
        self.load_env()
        self.open_api_key = os.getenv("OPEN_API_KEY","sk-proj-*****")
        self.open_proxy = os.getenv("OPEN_PROXY",None) or None
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY","*****")
        self.deepseek_proxy = os.getenv("DEEPSEEK_PROXY",None) or None
        self.fosp_open_id= os.getenv("FOSP_OPEN_ID","*****")
        self.fosp_developer_secret = os.getenv("FOSP_DEVELOPER_SECRET","*****")
        self.fosp_encode_key = os.getenv("FOSP_ENCODE_KEY","*****")
        self.fosp_proxy = os.getenv("FOSP_PROXY",None) or None

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


# 定义全局的Configuration对象，以便在整个程序中共享配置
config = Configuration()


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any],server_name: str = None
    ) -> None:
        self.name: str = name
        self.server_name = server_name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

class MCPPrompt:
    """Represents a prompt with its properties and formatting."""

    def __init__(
        self, name: str, description: str, arguments: list[PromptArgument],server_name: str = None
    ) -> None:
        self.name: str = name
        self.server_name = server_name
        self.description: str = description
        self.arguments: list[PromptArgument] = arguments

    def get_prompt_dict(self) -> dict:
        """Format prompt information for LLM.

        Returns:
            A formatted string describing the prompt.
        """
        prompt={"ServerName":self.server_name,"PromptName": self.name}
        if self.description:
            prompt["Description"]= self.description
        args_desc = []
        if self.arguments:
            for argObj in self.arguments:
                arg_dict = {
                    "Argument": argObj.name,
                }
                if argObj.description:
                    arg_dict["Description"]= argObj.description
                if argObj.required:
                    arg_dict["Required"]=True
                args_desc.append(arg_dict)
        if args_desc:
            prompt["Arguments"]=args_desc
        return prompt
    @property
    def format_for_rich(self) -> str:
        """Format prompt information for rich terminal."""
        return f"+ [bold white]{self.server_name}[/bold white] - [bold yellow]{self.name}[/bold yellow]" + \
            (f"\n  > [magenta]arguments[/magenta]: {json.dumps([arg.name for arg in self.arguments],ensure_ascii=False)}" if self.arguments else "") + \
            (f"\n  > [magenta]description[/magenta]: {self.description}" if self.description else "")



class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config.get("command") == "npx"
            else self.config.get("command")
        )
        sseUrl = self.config.get("sseUrl") # 增加sse连接支持
        streamableHttpUrl = self.config.get("streamableHttpUrl") # 增加StreamableHttp连接支持
        cm = None
        if command:
            server_params = StdioServerParameters(
                command=command,
                args=self.config["args"],
                env={**os.environ, **self.config["env"]}
                if self.config.get("env")
                else None,
            )
            cm = stdio_client(server_params)
            read, write = await self.exit_stack.enter_async_context(cm)
        elif sseUrl:
            cm = sse_client(sseUrl)
            read, write = await self.exit_stack.enter_async_context(cm)
        elif streamableHttpUrl:
            cm = streamablehttp_client(streamableHttpUrl,headers={"accept": "text/event-stream"})
            read, write, getSessionIdCallback = await self.exit_stack.enter_async_context(cm)
        else:
            raise ValueError("The command or sseUrl or streamableHttpUrl must be a valid string and cannot be None.")
        try:      
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except:
            await self.cleanup()
            raise Exception("Initialize MCP Server Failed")
        
        
    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        try:
            if not self.session:
                raise RuntimeError(f"Server {self.name} not initialized")

            tools_response = await self.session.list_tools()
            tools = []

            for item in tools_response:
                if isinstance(item, tuple) and item[0] == "tools":
                    tools.extend(
                        Tool(tool.name, tool.description, tool.inputSchema,server_name=self.name)
                        for tool in item[1]
                    )

            return tools
        except:
            error_console.print(f"Error getting tools from server {self.name}.")
            await self.cleanup()
            return None

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except:
                error_console.print(f"Error during cleanup of server {self.name}.")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
        """
        try:
            if not self.session:
                raise RuntimeError(f"Server {self.name} not initialized")
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            return CallToolResult(content=[TextContent(type="text",text=f"Error executing tool {self.name} - {tool_name}. {str(e)}.")],
                                  isError=True)

    async def list_prompts(self) -> list[MCPPrompt]:
        """Get prompts from the server.

        Returns:
            A list of prompts.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        try:
            prompts_response = await self.session.list_prompts()
            prompts = []
            for item in prompts_response:
                if isinstance(item, tuple) and item[0] == "prompts":
                    prompts.extend(
                        MCPPrompt(
                            prompt.name,
                            prompt.description,
                            prompt.arguments,
                            server_name=self.name
                        )
                        for prompt in item[1]
                    )
            return prompts
        except Exception as e:
            error_console.print(f"Error getting prompts from server {self.name}. {str(e)}")
            return []

    async def get_prompt(self, prompt_name: str, arguments: dict[str, Any]) -> GetPromptResult | None:
        """Call a prompt with retry mechanism.

        Args:
            prompt_name: Name of the prompt to call.
            arguments: Prompt arguments.

        Returns:
            Prompt execution result.

        Raises:
            RuntimeError: If server is not initialized.
        """
        try:
            if not self.session: 
                raise RuntimeError(f"Server {self.name} not initialized")        
            result = await self.session.get_prompt(prompt_name, arguments) 
            return result   
        except Exception as e:
            return None





class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str,ai_channel: str = "OpenAI",ai_model: str = "gpt-4",
                 ai_api_url: str = "https://api.openai.com/v1/chat/completions",
                 ai_provider: str = "OpenAI",
                 http_proxy: str = None) -> None:
        self.api_key: str = api_key
        self.ai_channel = ai_channel
        self.ai_model = ai_model
        self.ai_api_url = ai_api_url
        self.ai_provider = ai_provider
        self.http_proxy = http_proxy

    async def get_response(self, messages: list[dict[str, str]]) -> tuple[str,dict|None]:
        """Get a response from the LLM.
        Args:
            messages: A list of message dictionaries.
        Returns:
            The LLM's response as a string.
        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = self.ai_api_url

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": self.ai_model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            async with httpx.AsyncClient(proxy=self.http_proxy) as client:
                response = await client.post(url, headers=headers, json=payload,timeout=60)
                response.raise_for_status()
                data = response.json()
                # console.print(f"LLM response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                usage = None
                if data.get("usage"):
                    usage={}
                    usage["prompt_tokens"] = data["usage"].get("prompt_tokens")
                    usage["completion_tokens"] = data["usage"].get("completion_tokens")
                    usage["total_tokens"] = data["usage"].get("total_tokens")

                return data["choices"][0]["message"]["content"],usage

        except httpx.HTTPError as e:
            error_message = f"Error getting LLM response. {str(e)}"

            if isinstance(e, httpx.HTTPStatusError):
                error_message = f"Error getting LLM response: {e.response.status_code} | {e.response.text}"
            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            ),None

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient ) -> None:
        self.servers: list[Server] = servers
        self.invalid_servers: set[Server] = set()
        self.llm_client: LLMClient = llm_client
        self.usage :dict | None = None

    async def initialize_servers(self) -> None:
        """Initialize all servers."""
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                error_console.print(f"Error for server {server.name}. {str(e)}")
                self.invalid_servers.add(server)
    
    async def reinitialize_servers(self,new_servers: list[Server]) -> None:
        """Reinitialize all servers."""
        await self.cleanup_servers()
        self.servers = new_servers
        for server in self.servers:
            try:
                server.exit_stack = AsyncExitStack()
                await server.initialize()
            except Exception as e:
                error_console.print(f"Error for server {server.name}. {str(e)}")
                self.invalid_servers.add(server)

    def showSysInfo(self,msg:str|Markdown,title:str,subtitle:str=None):
        """Show system info."""
        sys_info_pannel = Panel(
            msg,
            title=title,
            title_align="left",
            style="bright_blue", 
            border_style="white", 
            subtitle = f"[gray37]{subtitle}[/gray37]" if subtitle else None,
            subtitle_align="right",
            padding=(1, 2)
        )
        console.print(sys_info_pannel)

    def assistantResponse(self,msg:str|Markdown,subtitle:str=None):
        panel = Panel(
            msg,
            title="[Assistant 🤖]",
            title_align="left",
            style="white",
            border_style="green",
            subtitle = f"[gray37]{subtitle}[/gray37]" if subtitle else None,
            subtitle_align="left",
            box = NoSlideBox(),
            padding=(1, 2)
        )
        return panel

    
    async def showAndGetAssistantResponse(self,call_llm: callable,subtitle:str=None):
        """Show Assistant response.

        Args:call_llm (callable): A function that returns a tuple of (assistant_response, usage).
        Returns: opitimazied_assistant_response (str|list|dict),  assistant_response (str)
        """
        with Live(auto_refresh=False) as live:
            start_time = asyncio.get_running_loop().time()
            task = asyncio.create_task(call_llm())
            input_obj = create_input()
            while not task.done():
                key_press = input_obj.read_keys()
                for key in key_press:
                    if key.data.upper() == 'P':
                        if not task.done():
                            task.cancel()
                elapsed = asyncio.get_running_loop().time() - start_time
                process_info = f"Waiting: ⌛ Cost [bold red]{elapsed:.2f}[/bold red] Sec"
                assistant_panel = self.assistantResponse(process_info,"[Press P to Cancel]")
                live.update(assistant_panel)
                live.refresh()
                await asyncio.sleep(0.2)  # 降低 CPU 占用
            try:
                input_obj.close()
                result,usage = task.result()
            except (asyncio.CancelledError,Exception) as e:
                result,usage = f"⚠️ You Cancelled Or Exception Occurred. {str(e)}",None
            if usage: # 更新usage
                self.usage = usage
            try:
                opt_result=json.loads(result)
                show_result= f"""```json\n{json.dumps(opt_result, indent=2, ensure_ascii=False)}\n```"""
            except json.JSONDecodeError:
                show_result = opt_result = result
            cost_info = f"🚩 Cost {elapsed:.2f} Sec"
            assistant_panel = self.assistantResponse(Markdown(show_result),cost_info if not subtitle else f"{subtitle} | {cost_info}")
            live.update(assistant_panel)
            live.refresh()
        return opt_result,result
        
    
    def toolCalledPanel(self,toolName:str,args: None | dict,process_info:str=None,out_put:str=None,subtitle:str=None):
        """Show Tool called."""
        process_msg=(
            f"Tool Calling: [bold yellow]{toolName}[/bold yellow]\n"
            f"Arguments: [bold light_sea_green]{args}[/bold light_sea_green]"
            f"{('\n'+process_info) if process_info else ''}"
        )
        result_msg = Markdown(
f"""Tool Called: *{toolName}*  
Arguments: *{args}*  
{('\n'+out_put) if out_put else ''}"""
        )
        tool_panel = Panel(
                process_msg if not out_put else result_msg,
                title="[MCP Tool 🔧]",
                title_align="left",
                border_style="magenta",
                subtitle = f"[gray37]{subtitle}[/gray37]" if subtitle else None,
                subtitle_align="left",
                # box = NoSlideBox(),
                padding=(1, 2)
            )
        return tool_panel


    def switch_model(self, model_no: str):
        """Switch the model of the LLM client.
        Args:
            model_no: The model number to switch to.
        Returns:
            True if the model was switched successfully, False otherwise.
        """
        model_info = client_models.get(model_no)
        if not model_info:
            error_console.print(f"Invalid model number: {model_no}")
            return
        if model_info["ai_provider"].lower() == "openai":
            self.llm_client = LLMClient(api_key=config.open_api_key,
                ai_channel=model_info["ai_channel"],
                ai_model=model_info["ai_model"],
                ai_api_url=model_info["ai_api_url"],
                ai_provider=model_info["ai_provider"],
                http_proxy=config.open_proxy)
        elif model_info["ai_provider"].lower() == "deepseek":
            self.llm_client = LLMClient(api_key=config.deepseek_api_key,
                ai_channel=model_info["ai_channel"],
                ai_model=model_info["ai_model"],
                ai_api_url=model_info["ai_api_url"],
                ai_provider=model_info["ai_provider"],
                http_proxy=config.deepseek_proxy)
        else:
            error_console.print(f"Unsupport AI provider: {model_info['ai_provider']}")
      

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            await server.cleanup()
            

    async def process_use_prompt(self, input_prompt: str) -> list[PromptMessage]|None:
        """"
        Process the use prompt and return the list of prompts.
        Args:
            input_prompt: The prompt to process.
        Returns:
            The list of prompts.
        """
        selected_prompt = None
        selected_server = None
        for server in self.servers:
            if server in self.invalid_servers:
                continue
            try:
                mcp_prompts = await server.list_prompts()
            except RuntimeError as e:
                error_console.print(f"❌ Failed to list prompts from {server.name}. {str(e)}")
                continue
            if not mcp_prompts:
                continue
            for mcp_prompt in mcp_prompts:
                if mcp_prompt.name == input_prompt:
                    selected_prompt = mcp_prompt
                    selected_server = server
                    break
            else:
                continue
            break
        if not selected_prompt:
            return None
        self.showSysInfo(selected_prompt.format_for_rich,"[Selected Prompt]","If Arguments are present, please fill them.")
        args = {}
        if selected_prompt.arguments:
            for arg in selected_prompt.arguments:
                user_input = Prompt.ask(f"> Fill [bold bright_cyan]{arg.name}[/bold bright_cyan]").strip()
                if not user_input and not arg.required:
                    continue
                args[arg.name] = user_input
        tool_resp_prompts = await selected_server.get_prompt(selected_prompt.name, args)
        if not tool_resp_prompts:
            return None
        return tool_resp_prompts.messages
                
        

    async def process_llm_response(self, llm_response: str|dict) -> str:
        """Process the LLM response and execute tools if needed.
        Args:
            llm_response: The response from the LLM.
        Returns:
            The result of tool execution or the original response.
        """
        if isinstance(llm_response, dict):
            tool_call=llm_response
        else:
            return llm_response
        if "tool" in tool_call:
            for server in self.servers:
                if server in self.invalid_servers:
                    continue
                tools = await server.list_tools()
                if not tools:
                    continue
                if any(tool.name == tool_call["tool"] for tool in tools):
                    try:
                        with Live(auto_refresh=False) as live:
                            start_time = asyncio.get_running_loop().time()
                            task = asyncio.create_task(server.execute_tool(tool_call["tool"], tool_call.get("arguments")))
                            # 实时计算并显示耗时
                            while not task.done():
                                elapsed = asyncio.get_running_loop().time() - start_time
                                process_info = f"Running: 🕒 Cost [bold red]{elapsed:.2f}[/bold red] Sec"
                                ctPanel = self.toolCalledPanel(tool_call["tool"],tool_call.get("arguments"),process_info=process_info)
                                live.update(ctPanel)
                                live.refresh()
                                await asyncio.sleep(0.2)  # 降低 CPU 占用  
                            result = task.result()
                            if result.content and result.content[0].type=='text':
                                calledRst = result.content[0].text.strip()
                                try:
                                    data=json.loads(calledRst)
                                    out_put = f"""\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"""
                                except Exception as e:
                                    out_put = f"\n{calledRst}"
                            else:
                                calledRst = f"{result.content}"
                            finish_info = f"{'❌' if result.isError else '🚩'} Cost {elapsed:.2f} Sec"
                            ctPanel = self.toolCalledPanel(tool_call["tool"],tool_call.get("arguments"),out_put=out_put,subtitle=finish_info)
                            live.update(ctPanel)
                            live.refresh()
                        return f"Tool execution result:\n {calledRst}"
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        error_console.print(f"{error_msg}")
                        return error_msg
            return f"No server found with tool: {tool_call['tool']}"
        return llm_response

    def get_tool_details(self,all_tools:list[Tool],tool_part_name: str) -> str:
        """
        Show the details of the tools that match the given part of the name.
        Args:   
            tool_part_name: The part of the name to match.
        """
        filter_tools = [tool for tool in all_tools if tool_part_name.lower() in tool.name.lower()]
        if not filter_tools:
            return "No tools found with that name."
        tools_details = "\r\n".join([
            f"+ [bold white]{tool.server_name}[/bold white] - [bold yellow]{tool.name}[/bold yellow]" + \
            (f"\n  > [magenta]description[/magenta]: {tool.description.strip()}" if tool.description else "")
             for tool in filter_tools])
        return tools_details
    
    def get_prompt_details(self,all_prompts:list[MCPPrompt],prompt_part_name: str) -> str:
        """
        Show the details of the prompts that match the given part of the name.
        Args:   
            prompt_part_name: The part of the name to match.
        """
        # Filter prompts that match the given part of the name
        filter_prompts = [prompt for prompt in all_prompts if prompt_part_name.lower() in prompt.name.lower()]  
        # If no prompts match, return a message
        if not filter_prompts:
            return "No prompts found with that name."
        # Join the details of the prompts into a string
        prompts_details = "\r\n".join([prompt.format_for_rich for prompt in filter_prompts])
        # Return the details
        return prompts_details

    async def start(self) -> None:
        """Main chat session handler."""
        mode = 1 # AI助手模式
        inMemoryHistory = InMemoryHistory()
        input_session = PromptSession(history=inMemoryHistory)
        try:
            # 启动配置的servers
            await self.initialize_servers()
            # 命令处理格式
            commands = ("[bold yellow]/clh[/bold yellow]:clean history [white]|[/white] "
                        "[bold yellow]/cls[/bold yellow]:clean screen [white]|[/white] "
                        "[bold yellow]/swm[/bold yellow]:switch model [white]|[/white] "
                        "[bold yellow]/uml[/bold yellow]:use multiLine\n"
                        "[bold yellow]/lst[/bold yellow]:list tools [white]|[/white] "
                        "[bold yellow]/std *tool_name*[/bold yellow]:show tool details [white]|[/white] "
                        "[bold yellow]/stu[/bold yellow]:show tokenUsage\n"
                        "[bold yellow]/lsp[/bold yellow]:list prompts [white]|[/white] "
                        "[bold yellow]/spd *prompt_name*[/bold yellow]:show prompt details [white]|[/white] "
                        "[bold yellow]/usp[/bold yellow]:use prompt"
            )
            short_commands_map = {
                "/clh":HTML("<style color='red'>/clh</style>:clean history"),
                "/cls":HTML("<style color='red'>/cls</style>:clean screen"),
                "/swm":HTML("<style color='red'>/swm</style>:switch model"),
                "/uml":HTML("<style color='red'>/uml</style>:use multiLine"),
                "/lst":HTML("<style color='red'>/lst</style>:ist tools"),
                "/std":HTML("<style color='red'>/std *</style>:show tool details"),
                "/stu":HTML("<style color='red'>/stu</style>:show tokenUsage"),
                "/lsp":HTML("<style color='red'>/lsp</style>:list prompts"),
                "/spd":HTML("<style color='red'>/spd *</style>:show prompt details"),
                "/usp":HTML("<style color='red'>/usp</style>:use prompt"),
                "/tool":HTML("<style color='red'>/tool</style>:tool caller mode"),
                "/assistant":HTML("<style color='red'>/assistant</style>:assistant mode"),
                "/reload":HTML("<style color='red'>/reload</style>:reload servers"),
                "/exit":HTML("<style color='red'>/exit</style>:exit the application")
            }
            short_commands = [k for k in short_commands_map.keys()]
            # 开头的帮助提示信息
            models_options = "\n".join(f"{no}. {model['ai_channel']} [white]|[/white] {model['ai_model']} [white]|[/white] {model['ai_provider']}" for no, model in client_models.items())
            self.showSysInfo(f"{self.llm_client.ai_channel} [white]|[/white] {self.llm_client.ai_model} [white]|[/white] {self.llm_client.ai_provider}","[Current AI Model]") # 展示当前模型信息
            self.showSysInfo(commands,"[Commands]",f"{'/tool' if mode else '/assistant'} | /reload | /exit")
            # 具体的工具及变量信息
            all_tools = []
            all_tools_nameFormat = []
            all_prompts = []
            all_prompts_nameFormat = []
            tools_description = ""
            tools_name = ""
            prompts_name = ""
            system_message = ""

            async def load_servers_info():
                """
                Load the tools and prompts from all servers.
                """
                nonlocal tools_description, tools_name, prompts_name, system_message # 需要修改
                all_tools.clear()
                all_tools_nameFormat.clear()
                all_prompts.clear()
                all_prompts_nameFormat.clear()
                for server in self.servers:
                    if server in self.invalid_servers:
                        continue
                    tools = await server.list_tools()
                    if tools is None: # 针对遇到Server异常情况
                        self.invalid_servers.add(server)
                        continue
                    if not tools:
                        # 获取所有tools
                        continue
                    # 获取所有tools
                    all_tools.extend(tools) 
                    # 根据server分类tools，将tools的名称每3个一行合并展示
                    ser_tools_nameFormat=""
                    for idx,tool in enumerate(tools):
                        ser_tools_nameFormat += f"+ {tool.name}" + ((" [white]|[/white] " if (idx+1) % 3 != 0 else "\n") if idx < len(tools)-1 else "")  
                    all_tools_nameFormat.append({
                        "server_name": server.name,
                        "ser_tools_nameFormat": ser_tools_nameFormat
                    })
                    # console.log(await server.get_prompt("Debug Assistant", {"error":"the arg xx is not definined"}))
                    mcp_prompts = await server.list_prompts()
                    all_prompts.extend(mcp_prompts)
                    ser_prompts_nameFormat=""
                    for idx,prompt in enumerate(mcp_prompts):
                        ser_prompts_nameFormat += f"+ {prompt.name}" + ((" [white]|[/white] " if (idx+1) % 3 != 0 else "\n") if idx < len(mcp_prompts)-1 else "")
                    all_prompts_nameFormat.append({
                        "server_name": server.name,
                        "ser_prompts_nameFormat": ser_prompts_nameFormat
                    })
                tools_description = "\n".join([tool.format_for_llm() for tool in all_tools]) # 核心prompts使用！！
                tools_name = "\r\n".join([f"[bold yellow]{tool['server_name']}[/bold yellow]\n{tool['ser_tools_nameFormat']}" for tool in all_tools_nameFormat])
                prompts_name = "\r\n".join([f"[bold yellow]{prompt['server_name']}[/bold yellow]\n{prompt['ser_prompts_nameFormat']}" for prompt in all_prompts_nameFormat])
                system_message = (
                    "You are a helpful assistant with access to these tools:\n\n"
                    f"{tools_description}\n"
                    "Choose the appropriate tool based on the user's question. "
                    "If no tool is needed, reply directly.\n\n"
                    "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                    "the exact JSON object format below, nothing else:\n"
                    "{\n"
                    '    "tool": "tool-name",\n'
                    '    "arguments": {\n'
                    '        "argument-name": "value"\n'
                    "    }\n"
                    "}\n\n"
                    "Attention: If multiple tools (quantity > 1) with similar meanings are identified due to the user's question:\n "
                    "list these tools and provide them with serial numbers (starting from 1 and incrementing sequentially) for the user to choose from, "
                    "you must ONLY respond with the exact JSON object format below, nothing else:\n"
                    "[{\n"
                    '    "No.": 1,\n'
                    '    "tool": "tool-name",\n'
                    '    "arguments": {\n'
                    '        "argument-name": "value"\n'
                    "    }\n"
                    "},\n\n"
                    "{\n"
                    '    "No.": 2,\n'
                    '    "tool": "tool-name",\n'
                    '    "arguments": {\n'
                    '        "argument-name": "value"\n'
                    "    }\n"
                    "}]\n\n"
                    "After receiving a tool's response:\n"
                    "1. Transform the raw data into a natural, conversational response\n"
                    "2. Keep responses concise but informative\n"
                    "3. Focus on the most relevant information\n"
                    "4. Use appropriate context from the user's question\n"
                    "5. Avoid simply repeating the raw data\n\n"
                    "Please use only the tools that are explicitly defined above."
                )

            # 加载servers信息
            await load_servers_info()
            # 对话开始
            messages = [{"role": "system", "content": system_message}]
            while True:
                try:
                    console.print("") # 增加一个空行
                    # user_input = Prompt.ask("[bold bright_cyan][You 💬][/bold bright_cyan]").strip()
                    cmd_completer = WordCompleter(short_commands, display_dict=short_commands_map,ignore_case=True,match_middle=False,sentence=True)
                    user_input = (await input_session.prompt_async(HTML("<style color='cyan' bord='bord'>[You 💬]</style> "), completer=cmd_completer,multiline=False)).strip()
                    console.print("") # 增加一个空行
                    if not user_input:
                        error_console.print("⚠️ You Need Input Something...")
                        continue
                    if user_input.startswith("/") and user_input.split(" ")[0] not in short_commands:
                        console.print(f"⚠️ Invalid Command")
                        continue
                    if user_input.lower() in ["/std","/spd"]:
                        error_console.print("⚠️ Your Command is not complete...")
                        continue
                    if user_input.lower() in ["/lst", "/list tools"]:
                        self.showSysInfo(tools_name,"[MCP Tools]")
                        continue
                    if user_input.lower().startswith("/std "):
                        tool_name = user_input[4:].strip()
                        tools_details = self.get_tool_details(all_tools,tool_name)
                        self.showSysInfo(tools_details,"[Tools Details]")
                        continue
                    if user_input.lower().startswith("/spd "):
                        prompt_name = user_input[4:].strip()
                        prompts_details = self.get_prompt_details(all_prompts,prompt_name)
                        self.showSysInfo(prompts_details,"[Prompts Details]")
                        continue
                    if user_input.lower() in ["/clh", "/clean history"]:
                        del messages[1:]
                        self.usage =None
                        server_console.print("📢 Cleaned History...")
                        continue
                    if user_input.lower() in ["/cls", "/clean screen"]:
                        if os.name == 'posix':  # Unix/Linux/Mac
                            print("\033c", end="")
                        elif os.name in ('nt', 'dos'):  # Windows
                            os.system('cls')
                        self.showSysInfo(f"{self.llm_client.ai_channel} [white]|[/white] {self.llm_client.ai_model} [white]|[/white] {self.llm_client.ai_provider}","[Current AI Model]") # 展示当前模型信息
                        self.showSysInfo(commands,"[Commands]",f"{'/tool' if mode else '/assistant'} | /reload | /exit")
                        continue
                    if user_input.lower() in ["/swm","/switch model"]:
                        self.showSysInfo(models_options,"[AI Model Options]")
                        user_input = Prompt.ask("[bold cyan]Choose 🤔[/bold cyan]",choices=list(client_models.keys())).strip()
                        self.switch_model(user_input)
                        self.showSysInfo(f"{self.llm_client.ai_channel} [white]|[/white] {self.llm_client.ai_model} [white]|[/white] {self.llm_client.ai_provider}","[Current AI Model]") # 展示当前模型信息
                        del messages[1:]
                        self.usage =None
                        continue
                    if user_input.lower() in ["/stu","/show token usage"]:
                        if not self.usage:
                            usageInfo="No token usage information available"
                        else:
                            usageInfo = (f"Prompt Tokens: [yellow]{self.usage['prompt_tokens']}[/yellow]\n"
                                         f"Completion Tokens: [yellow]{self.usage['completion_tokens']}[/yellow]\n"
                                         f"Total Tokens: [yellow]{self.usage['total_tokens']}[/yellow]")
                        self.showSysInfo(usageInfo,"[Token Usage]")
                        continue
                    
                    if user_input.lower() in ["/lsp", "/list prompts"]:
                        self.showSysInfo(prompts_name,"[Prompt List]")
                        continue
                    if user_input.lower() == "/tool":
                        mode = 0
                        self.showSysInfo(commands,"[Commands]",f"{'/tool' if mode else '/assistant'} | /reload | /exit")
                        continue
                    if user_input.lower() == "/assistant":
                        mode = 1
                        self.showSysInfo(commands,"[Commands]",f"{'/tool' if mode else '/assistant'} | /reload | /exit")
                        continue
                    if user_input.lower() in ["/reload"]:
                        server_config = config.load_config("servers_config.json") # 重新读取Severs配置
                        new_servers = [
                            Server(name, srv_config)
                            for name, srv_config in server_config["mcpServers"].items() if not srv_config.get("disabled")
                        ]
                        await self.reinitialize_servers(new_servers)
                        await load_servers_info()
                        messages.clear()
                        messages = [{"role": "system", "content": system_message}]
                        self.usage =None
                        console.print("💻 Reloaded Servers...")
                        continue
                    if user_input.lower() in ["/quit", "/exit"]:
                        console.print("💻 Exiting...")
                        break
                    if user_input.lower() in ["/uml","/use multiline"]:
                        console.print("[bold]已开启多行输入[/bold](按Esc·Enter提交)") 
                        user_input = (await input_session.prompt_async("> ", multiline=True)).strip()
                        console.print("") # 增加一个空行
                        if not user_input:
                            console.print("⚠️ You Need Input Something...")
                            continue 
                    # 使用server提供的prompts,此项必须在最后位置
                    if user_input.lower() in ["/usp", "/use prompt"]:
                        console.print("[bold]输入可联想PromptName[/bold](按↑↓选择·Enter提交)")
                        word_completer = WordCompleter([prompt.name for prompt in all_prompts], ignore_case=True,match_middle=True)
                        prm_input = (await input_session.prompt_async("> ", completer=word_completer,multiline=False)).strip()
                        console.print("") # 增加一个空行
                        if prm_input not in word_completer.words:
                            console.print("⚠️ Invalid Prompt Name...")
                            continue
                        prompt_messages = await self.process_use_prompt(prm_input)
                        if not prompt_messages:
                            console.print(f"⚠️ Can't get Prompt - {prm_input} from MCP Servers...")
                            continue
                        show_prompts = []
                        for prompt_message in prompt_messages:
                            if prompt_message.role.lower() == "system" and isinstance(prompt_message.content, TextContent):
                                messages.append({"role": "system", "content": prompt_message.content.text})
                                show_prompts.append(f"[blue]System[/blue]: [white]{prompt_message.content.text}[/white]")
                            if prompt_message.role.lower() == "user" and isinstance(prompt_message.content, TextContent):
                                messages.append({"role": "user", "content": prompt_message.content.text})
                                show_prompts.append(f"[bright_cyan]User[/bright_cyan]: [white]{prompt_message.content.text}[/white]")
                            if prompt_message.role.lower() == "assistant" and isinstance(prompt_message.content, TextContent):
                                messages.append({"role": "assistant", "content": prompt_message.content.text})
                                show_prompts.append(f"[green]Assistant[/green]: [white]{prompt_message.content.text}[/white]")
                        if not show_prompts:
                            console.print(f"⚠️ No text message from Prompt - {prm_input} from MCP Servers...")
                            continue 
                        console.print("") # 增加一个空行
                        self.showSysInfo("\n".join(show_prompts),"[Used Prompt]")
                        user_input = "show me!"
                    if user_input:
                        messages.append({"role": "user", "content": user_input})
                    # 尝试获取AI的响应
                    llm_response,orig_llm_response = await self.showAndGetAssistantResponse(lambda: self.llm_client.get_response(messages))
                    # 处理MCP Tool调用
                    while True:
                        messages.append({"role": "assistant", "content": orig_llm_response})
                        try:
                            if isinstance(llm_response, list):
                                choices=[str(x["No."]) for x in llm_response if x.get("No.")]
                                if choices and choices[0]:
                                    user_input = Prompt.ask("[bold cyan]Choose NO.(0 to skip)🤔[/bold cyan]",choices=['0']+choices).strip()
                                    selectTools=list(filter(lambda x: str(x.get("No."))==str(user_input),llm_response))
                                    if selectTools:
                                        llm_response = selectTools[0]
                                        messages.append({"role": "user", "content": f"I choose No.{user_input}"})
                                    else:
                                        llm_response = None
                                        messages.append({"role": "user", "content": f"I don't need any tools"})
                                        break
                            result = await self.process_llm_response(llm_response)
                            if mode == 1 and result != llm_response:
                                messages.append({"role": "user", "content": "I can see the tool has finished execution. "+result})
                                llm_response,orig_llm_response = await self.showAndGetAssistantResponse(lambda: self.llm_client.get_response(messages),"[Tool Response Summary]")
                            else:
                                break
                        except Exception as e:
                            error_console.print(f"{str(e)}")
                            break
                    # for msg in messages[1:]:
                    #     print(f"【{msg['role']}】\n\t {msg['content'].replace('\n','\n\t')}")
                except KeyboardInterrupt:
                    console.print("💻 Exiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items() if not srv_config.get("disabled")
    ]
    llm_client = LLMClient(config.fosp_open_id, config.fosp_developer_secret, config.fosp_encode_key,http_proxy=config.fosp_proxy)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())