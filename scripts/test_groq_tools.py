
import json
import os

from groq import Groq


def main() -> None:
    # 检查 API Key，避免请求前才发现环境变量缺失
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq()

    # 模拟 Agent 提供给模型的工具定义
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_page_count",
                "description": "获取当前 PDF 文件的总页数。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pdf",
                "description": "在当前 PDF 文件中搜索关键词。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "需要搜索的关键词。",
                        }
                    },
                    "required": ["keyword"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    # 用一个明确的问题测试模型能否自主选择工具
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "请在当前 PDF 文件中搜索关键词 Python。",
            }
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    message = response.choices[0].message

    # 输出模型选择的工具和参数，不执行真实的 PDF 操作
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")

            # 检查模型返回的参数是否为合法 JSON
            arguments = json.loads(tool_call.function.arguments)
            print(f"Parsed arguments: {arguments}")
    else:
        print("No tool call returned.")
        print(f"Model response: {message.content}")


if __name__ == "__main__":
    main()