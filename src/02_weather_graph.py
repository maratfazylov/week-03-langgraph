"""Шаг 2: отдельный погодный workflow с инструментом Open-Meteo."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.settings import create_deepseek_model
from src.weather_tool import get_current_weather

SYSTEM_PROMPT = """Ты полезный помощник. Всегда отвечай по-русски.
Для вопросов о текущей погоде обязательно используй get_current_weather.
Не придумывай погоду. Если инструмент сообщил об ошибке, ненайденном или
неоднозначном городе, объясни это кратко и попроси пользователя уточнить запрос.
На обычные вопросы отвечай самостоятельно, без вызова инструмента."""


def build_graph(model=None, tools: Sequence[Any] | None = None):
    """Собрать и скомпилировать цикл LLM -> tools -> LLM."""
    available_tools = list(tools) if tools is not None else [get_current_weather]
    # При импорте Studio граф должен собраться даже без ключа. Настоящая модель
    # создаётся лениво при первом сообщении пользователя.
    llm_with_tools = model.bind_tools(available_tools) if model is not None else None

    def llm_node(state: MessagesState) -> dict:
        nonlocal llm_with_tools
        if llm_with_tools is None:
            llm_with_tools = create_deepseek_model().bind_tools(available_tools)

        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", ToolNode(available_tools))

    workflow.add_edge(START, "llm")
    workflow.add_conditional_edges(
        "llm",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "llm")

    return workflow.compile()


# Agent Server импортирует этот уже скомпилированный граф из langgraph.json.
# Ключ DeepSeek здесь ещё не читается: он понадобится только при первом invoke.
graph = build_graph()


def main() -> None:
    question = input("Ваш вопрос: ").strip()
    if not question:
        raise SystemExit("Введите непустой вопрос.")

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    print("\nDeepSeek:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
