"""Погодный инструмент на бесплатных API Open-Meteo."""

from collections.abc import Mapping
from typing import Any

import httpx
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10.0

WEATHER_DESCRIPTIONS = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "туман с изморозью",
    51: "лёгкая морось",
    53: "морось",
    55: "сильная морось",
    56: "лёгкая ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "небольшой ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "небольшой снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}


def _location_label(location: Mapping[str, Any]) -> str:
    """Собрать короткое, но различимое название места."""
    parts: list[str] = []
    for field in ("name", "admin1", "country"):
        value = location.get(field)
        if value and str(value) not in parts:
            parts.append(str(value))
    return ", ".join(parts)


def _ambiguity_message(city: str, results: list[Mapping[str, Any]]) -> str | None:
    """Попросить уточнение, если API вернул одноимённые города."""
    if "," in city:
        return None

    # Сравниваем с первым результатом, а не с исходным запросом: при language=ru
    # API может локализовать, например, Springfield в «Спрингфилд».
    first_name = str(results[0].get("name", "")).casefold()
    same_name_matches = [
        location
        for location in results
        if str(location.get("name", "")).casefold() == first_name
    ]
    labels = list(dict.fromkeys(_location_label(item) for item in same_name_matches))

    if len(labels) <= 1:
        return None

    variants = "; ".join(labels[:3])
    return (
        f"Город «{city}» найден в нескольких местах: {variants}. "
        "Уточните регион или страну через запятую."
    )


def _request_weather(city: str, client: httpx.Client) -> str:
    geocoding_response = client.get(
        GEOCODING_URL,
        params={"name": city, "count": 5, "language": "ru", "format": "json"},
    )
    geocoding_response.raise_for_status()
    geocoding_data = geocoding_response.json()
    results = geocoding_data.get("results") if isinstance(geocoding_data, dict) else None

    if not isinstance(results, list) or not results:
        return (
            f"Город «{city}» не найден. "
            "Попросите пользователя проверить написание или добавить страну."
        )

    ambiguity = _ambiguity_message(city, results)
    if ambiguity:
        return ambiguity

    location = results[0]
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return "Сервис геокодирования вернул неполные координаты. Попробуйте позже."

    forecast_response = client.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
    )
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()
    current = forecast_data.get("current") if isinstance(forecast_data, dict) else None
    units = forecast_data.get("current_units") if isinstance(forecast_data, dict) else None

    required = {
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
    }
    if not isinstance(current, dict) or not required.issubset(current):
        return "Сервис погоды вернул неполные текущие данные. Попробуйте позже."
    if not isinstance(units, dict):
        units = {}

    code = current["weather_code"]
    description = WEATHER_DESCRIPTIONS.get(code, f"код погоды WMO {code}")
    label = _location_label(location)
    observed_at = current.get("time", "время не указано")

    return (
        f"{label}; данные на {observed_at}; {description}; "
        f"температура {current['temperature_2m']} {units.get('temperature_2m', '°C')}, "
        f"ощущается как {current['apparent_temperature']} "
        f"{units.get('apparent_temperature', '°C')}; "
        f"влажность {current['relative_humidity_2m']} "
        f"{units.get('relative_humidity_2m', '%')}; "
        f"осадки {current['precipitation']} {units.get('precipitation', 'мм')}; "
        f"ветер {current['wind_speed_10m']} "
        f"{units.get('wind_speed_10m', 'км/ч')}."
    )


def fetch_current_weather(city: str, *, client: httpx.Client | None = None) -> str:
    """Получить краткие текущие условия или понятное сообщение об ошибке."""
    normalized_city = city.strip()
    if len(normalized_city) < 2:
        return "Название города слишком короткое. Уточните город."

    owns_client = client is None
    http_client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        return _request_weather(normalized_city, http_client)
    except httpx.TimeoutException:
        return "Open-Meteo не ответил вовремя. Попробуйте ещё раз позже."
    except httpx.HTTPStatusError as exc:
        return f"Open-Meteo вернул ошибку HTTP {exc.response.status_code}. Попробуйте позже."
    except httpx.RequestError:
        return "Не удалось связаться с Open-Meteo. Проверьте сеть и попробуйте позже."
    except (TypeError, ValueError):
        return "Open-Meteo вернул данные в неожиданном формате. Попробуйте позже."
    finally:
        if owns_client:
            http_client.close()


@tool
def get_current_weather(city: str) -> str:
    """Узнать текущую погоду по названию города; город можно уточнить страной."""
    return fetch_current_weather(city)
