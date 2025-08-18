这里根据MCP github上的 MCP chat client example代码( https://github.com/modelcontextprotocol/python-sdk/tree/main/examples/clients/simple-chatbot )，改写了一下。  
主要是支持OpenAI、Deepseek的API，也可根据自己的需要改成自定义的也行。  
利用 rich、prompt_toolkit 美化了控制台的界面，不再是日志格式，并添加了更多MCP特性（比如支持server端的prompts列表读取及使用），以及聊天器必备命令。  
在输入框中输入 / 后会自动联想对应的命令：  
<img width="1467" height="931" alt="image" src="https://github.com/user-attachments/assets/31b12b7e-1d77-43f3-9c69-3b9a8bf7ee35" />  
常用命令：/clh可以清理上下文，相当开始一个新的会话；/cls清屏；/swm切换大模型；/reload重新读取server配置连接server...  
另外支持Tool连续调用处理（取决于大模型能力），也增加prompts意义含糊tool不明确可以选择调用功能。  

正常调用MCP TOOL：  
<img width="1452" height="1277" alt="image" src="https://github.com/user-attachments/assets/9d5ed70b-f419-4229-8ffa-cdc730a49996" />

使用Prompts：  
<img width="1443" height="635" alt="image" src="https://github.com/user-attachments/assets/66eca7ff-a431-466c-8315-07f405984902" />

环境及要求：  
支持系统windows，最好是安装了新版windows console.  
python>=3.13  
pip install -r requirements.txt  
启动：  
python simple_mcp_client.py  

MCP Server中配置MCP Server：servers_config.json，与其他MCP Client的配置相差不大。  
另外还在原来的基础上支持了SSE及StreamableHttp模式连接，例子中有。

