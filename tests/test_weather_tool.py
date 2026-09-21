"""Погодные тесты используют подставной HTTP-транспорт, а не интернет."""

import httpx

from src.weather_tool import fetch_current_weather


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_weather_for_found_city() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "Казань"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Казань",
                            "admin1": "Татарстан",
                            "country": "Россия",
                            "latitude": 55.79,
                            "longitude": 49.12,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-09-21T12:00",
                    "temperature_2m": 14.2,
                    "apparent_temperature": 13.1,
                    "relative_humidity_2m": 71,
                    "precipitation": 0.0,
                    "weather_code": 1,
                    "wind_speed_10m": 8.4,
                },
                "current_units": {
                    "temperature_2m": "°C",
                    "apparent_temperature": "°C",
                    "relative_humidity_2m": "%",
                    "precipitation": "mm",
                    "wind_speed_10m": "km/h",
                },
            },
        )

    with make_client(handler) as client:
        result = fetch_current_weather("Казань", client=client)

    assert "Казань, Татарстан, Россия" in result
    assert "преимущественно ясно" in result
    assert "14.2 °C" in result


def test_city_not_found() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"results": []}))
    with client:
        result = fetch_current_weather("Несуществующийгород", client=client)
    assert "не найден" in result


def test_ambiguous_city_asks_for_qualification() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Спрингфилд",
                        "admin1": "Illinois",
                        "country": "США",
                        "latitude": 39.8,
                        "longitude": -89.6,
                    },
                    {
                        "name": "Спрингфилд",
                        "admin1": "Massachusetts",
                        "country": "США",
                        "latitude": 42.1,
                        "longitude": -72.6,
                    },
                ]
            },
        )

    with make_client(handler) as client:
        result = fetch_current_weather("Springfield", client=client)

    assert "в нескольких местах" in result
    assert "Уточните регион или страну" in result


def test_network_error_is_readable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with make_client(handler) as client:
        result = fetch_current_weather("Казань", client=client)
    assert "Не удалось связаться" in result


def test_forecast_http_error_is_readable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Казань",
                            "country": "Россия",
                            "latitude": 55.79,
                            "longitude": 49.12,
                        }
                    ]
                },
            )
        return httpx.Response(503, json={"error": True})

    with make_client(handler) as client:
        result = fetch_current_weather("Казань", client=client)
    assert "HTTP 503" in result
