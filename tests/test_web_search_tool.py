"""Проверки веб-поиска без настоящих сетевых запросов."""

from src.web_search_tool import search_web


class StubSearchClient:
    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.call = None

    def text(self, query: str, **kwargs):
        self.call = (query, kwargs)
        if self.error:
            raise self.error
        return self.results


def test_search_returns_titles_links_and_snippets() -> None:
    client = StubSearchClient(
        [
            {
                "title": "Документация Python",
                "href": "https://docs.python.org/3/",
                "body": "Официальная документация языка Python.",
            },
            {
                "title": "Python.org",
                "href": "https://www.python.org/",
                "body": "Официальный сайт проекта.",
            },
        ]
    )

    result = search_web("  официальный   сайт Python  ", client=client)

    assert client.call[0] == "официальный сайт Python"
    assert client.call[1]["region"] == "ru-ru"
    assert client.call[1]["max_results"] == 5
    assert "1. Документация Python" in result
    assert "https://docs.python.org/3/" in result
    assert "Официальная документация" in result


def test_search_without_results_asks_to_refine_query() -> None:
    result = search_web("редкий запрос", client=StubSearchClient())
    assert "не нашёл" in result
    assert "Уточните запрос" in result


def test_search_rejects_empty_query() -> None:
    result = search_web(" ", client=StubSearchClient())
    assert "слишком короткий" in result


def test_search_handles_client_failure() -> None:
    client = StubSearchClient(error=RuntimeError("offline"))
    result = search_web("новости Python", client=client)
    assert "Не удалось выполнить веб-поиск" in result
