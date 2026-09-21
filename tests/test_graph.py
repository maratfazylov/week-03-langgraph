"""Проверяем полный цикл графа без настоящего запроса к DeepSeek."""

import importlib
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

weather_graph = importlib.import_module("src.02_weather_graph")
web_search_graph = importlib.import_module("src.03_web_search_graph")
ROOT = Path(__file__).resolve().parents[1]


@tool
def get_current_weather(city: str) -> str:
    """Вернуть тестовую погоду для города."""
    return f"{city}: +15 °C, ясно"


class FakeToolCallingModel:
    """Сначала просит инструмент, затем формулирует финальный ответ."""

    def __init__(self) -> None:
        self.calls = 0
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return RunnableLambda(self._respond)

    def _respond(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_weather",
                        "args": {"city": "Казань"},
                        "id": "weather-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        assert tool_messages[-1].content == "Казань: +15 °C, ясно"
        return AIMessage(content="Сейчас в Казани +15 °C и ясно.")


def test_graph_calls_tool_and_returns_result_to_model() -> None:
    model = FakeToolCallingModel()
    graph = weather_graph.build_graph(model=model, tools=[get_current_weather])

    result = graph.invoke({"messages": [("user", "Какая погода в Казани?")]})

    assert model.calls == 2
    assert model.bound_tools == [get_current_weather]
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "Сейчас в Казани +15 °C и ясно."


def test_graph_finishes_without_tool_for_ordinary_question() -> None:
    class DirectAnswerModel(FakeToolCallingModel):
        def _respond(self, messages):
            self.calls += 1
            return AIMessage(content="TCP — транспортный протокол.")

    model = DirectAnswerModel()
    graph = weather_graph.build_graph(model=model, tools=[get_current_weather])

    result = graph.invoke({"messages": [("user", "Что такое TCP?")]})

    assert model.calls == 1
    assert result["messages"][-1].content == "TCP — транспортный протокол."


def test_exported_graph_does_not_create_model_during_import(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("DeepSeek не должен создаваться при импорте графа")

    monkeypatch.setattr(weather_graph, "create_deepseek_model", fail_if_called)

    exported_graph = weather_graph.build_graph()

    assert exported_graph.get_graph().nodes
    assert weather_graph.graph.get_graph().nodes


def test_weather_graph_binds_only_weather_tool() -> None:
    model = FakeToolCallingModel()

    weather_graph.build_graph(model=model)

    assert [tool.name for tool in model.bound_tools] == ["get_current_weather"]


def test_web_search_graph_binds_only_web_search_tool() -> None:
    model = FakeToolCallingModel()

    web_search_graph.build_graph(model=model)

    assert [tool.name for tool in model.bound_tools] == ["web_search"]


def test_web_search_graph_does_not_create_model_during_import(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("DeepSeek не должен создаваться при импорте графа")

    monkeypatch.setattr(web_search_graph, "create_deepseek_model", fail_if_called)

    exported_graph = web_search_graph.build_graph()

    assert exported_graph.get_graph().nodes
    assert web_search_graph.graph.get_graph().nodes


def test_langgraph_config_points_to_exported_graph() -> None:
    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))

    assert config["dependencies"] == ["."]
    assert config["env"] == ".env"
    assert config["python_version"] == "3.11"
    assert config["graphs"]["weather_agent"]["path"] == (
        "./src/02_weather_graph.py:graph"
    )
    assert config["graphs"]["web_search_agent"]["path"] == (
        "./src/03_web_search_graph.py:graph"
    )
