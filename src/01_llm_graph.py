"""Шаг 1: состояние с сообщениями и один узел LLM."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.settings import create_deepseek_model

SYSTEM_PROMPT = "Отвечай по-русски, кратко и по существу."


def build_graph(model=None):
    """Собрать самый простой граф START -> llm -> END."""
    llm = model or create_deepseek_model()

    def llm_node(state: MessagesState) -> dict:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("llm", llm_node)
    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)
    return workflow.compile()


def main() -> None:
    question = input("Ваш вопрос: ").strip()
    if not question:
        raise SystemExit("Введите непустой вопрос.")

    result = build_graph().invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    print("\nDeepSeek:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
