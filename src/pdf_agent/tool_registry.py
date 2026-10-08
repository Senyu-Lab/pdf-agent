from pdf_agent.tools import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {}

        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        # 防止重复注册相同名称的 Tool
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Tool not found: {name}") from error

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        # 将所有已注册 Tool 转换为 LLM Tool Schema
        return [
            tool.to_schema()
            for tool in self._tools.values()
        ]