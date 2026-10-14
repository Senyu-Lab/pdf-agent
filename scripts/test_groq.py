
import os

from groq import Groq


def main() -> None:
    # 检查 API Key，避免 SDK 报出不够明确的初始化错误
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq()

    response = client.chat.completions.create(
        # 先验证基础对话，模型名称之后可按控制台可用列表调整
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq API works.",
            }
        ],
        temperature=0,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()