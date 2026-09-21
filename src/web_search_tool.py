"""Простой веб-поиск без отдельного API-ключа."""

from typing import Any, Protocol

from langchain_core.tools import tool

MAX_RESULTS = 5
MAX_SNIPPET_LENGTH = 350


class SearchClient(Protocol):
    """Минимальный интерфейс клиента, который удобно подменять в тестах."""

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]: ...


def _one_line(value: Any) -> str:
    """Убрать лишние пробелы и переводы строк из поискового результата."""
    return " ".join(str(value or "").split())


def _format_results(results: list[dict[str, Any]]) -> str:
    formatted: list[str] = []

    for result in results[:MAX_RESULTS]:
        title = _one_line(result.get("title")) or "Без названия"
        url = _one_line(result.get("href"))
        snippet = _one_line(result.get("body"))
        if not url:
            continue
        if len(snippet) > MAX_SNIPPET_LENGTH:
            snippet = snippet[: MAX_SNIPPET_LENGTH - 1].rstrip() + "…"

        item = f"{len(formatted) + 1}. {title}\nURL: {url}"
        if snippet:
            item += f"\nФрагмент: {snippet}"
        formatted.append(item)

    if not formatted:
        return "Поиск не нашёл подходящих результатов. Уточните запрос."

    return "Результаты веб-поиска:\n\n" + "\n\n".join(formatted)


def search_web(query: str, *, client: SearchClient | None = None) -> str:
    """Найти веб-страницы и вернуть заголовки, ссылки и короткие фрагменты."""
    normalized_query = " ".join(query.split())
    if len(normalized_query) < 2:
        return "Поисковый запрос слишком короткий. Уточните, что нужно найти."
    if len(normalized_query) > 300:
        return "Поисковый запрос слишком длинный. Сформулируйте его короче."

    try:
        if client is None:
            # Импортируем сетевой клиент только при вызове инструмента.
            from ddgs import DDGS

            with DDGS(timeout=10) as search_client:
                results = search_client.text(
                    normalized_query,
                    region="ru-ru",
                    safesearch="moderate",
                    max_results=MAX_RESULTS,
                )
        else:
            results = client.text(
                normalized_query,
                region="ru-ru",
                safesearch="moderate",
                max_results=MAX_RESULTS,
            )
    except Exception:
        return "Не удалось выполнить веб-поиск. Проверьте сеть и попробуйте позже."

    if not isinstance(results, list):
        return "Поисковый сервис вернул данные в неожиданном формате. Попробуйте позже."
    return _format_results(results)


@tool
def web_search(query: str) -> str:
    """Искать в интернете свежие сведения и источники по текстовому запросу."""
    return search_web(query)
