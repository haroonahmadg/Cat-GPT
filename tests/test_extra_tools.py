import io
import sqlite3

import pytest
from PIL import Image

import agent.tools as agent_tools
from agent import graph


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, headers=None, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture(autouse=True)
def no_live_keys(monkeypatch):
    monkeypatch.setenv("CRICAPI_KEY", "")
    monkeypatch.setenv("OMDB_API_KEY", "")
    monkeypatch.setenv("RESEND_API_KEY", "")
    monkeypatch.setattr(agent_tools, "emit_tool_event", lambda *args, **kwargs: None)


@pytest.fixture
def email_database(tmp_path, monkeypatch):
    import api.store as store

    db_path = tmp_path / "users.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE users(user_id TEXT PRIMARY KEY, email TEXT, google_id TEXT)"
        )
        connection.execute(
            "INSERT INTO users(user_id, email, google_id) VALUES (?, ?, ?)",
            ("user-1", "verified@example.com", "google-subject"),
        )

    def connect():
        return sqlite3.connect(db_path, timeout=10)

    monkeypatch.setattr(store, "get_ui_connection", connect)
    token = agent_tools.active_user_id.set("user-1")
    try:
        yield db_path
    finally:
        agent_tools.active_user_id.reset(token)


def _png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (20, 20), color="white").save(output, format="PNG")
    return output.getvalue()


def test_get_weather_success_and_unknown_city(monkeypatch):
    payloads = [
        {"results": [{"name": "Lahore", "latitude": 31.5, "longitude": 74.3}]},
        {
            "current": {
                "temperature_2m": 28.3,
                "relative_humidity_2m": 40,
                "weather_code": 1,
                "wind_speed_10m": 8.2,
            },
            "daily": {
                "time": ["2026-10-22", "2026-10-23", "2026-10-24"],
                "temperature_2m_max": [30, 31, 29],
                "temperature_2m_min": [20, 21, 19],
                "weather_code": [1, 2, 3],
            },
        },
    ]
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(agent_tools.httpx, "get", get)
    result = agent_tools.get_weather.invoke({"city": "Lahore"})
    assert "28.3°C" in result
    assert "2026-10-24" in result
    assert len(calls) == 2
    assert all(call["timeout"] == 10 for call in calls)

    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({"results": []}),
    )
    assert "couldn't find a city" in agent_tools.get_weather.invoke({"city": "Atlantis"})


def test_get_prayer_times_success_and_http_failure(monkeypatch):
    timings = {name: f"{hour:02d}:00" for name, hour in zip(
        ("Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"), range(5, 10)
    )}
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({"data": {"timings": timings}}),
    )
    result = agent_tools.get_prayer_times.invoke({"city": "Lahore"})
    assert all(name in result for name in timings)
    assert "Sunrise" not in result

    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({}, status_code=503),
    )
    assert "HTTP 503" in agent_tools.get_prayer_times.invoke({"city": "Lahore"})


def test_convert_currency_success_and_unknown_code(monkeypatch):
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "result": "success",
            "rates": {"PKR": 278.5},
            "time_last_update_utc": "Thu, 22 Oct 2026 00:00:01 +0000",
        }),
    )
    result = agent_tools.convert_currency.invoke(
        {"amount": 10, "from_code": "USD", "to_code": "PKR"}
    )
    assert "2,785.00 PKR" in result
    assert "rate 278.5" in result

    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({"result": "success", "rates": {}}),
    )
    assert "doesn't recognize currency code JPY" in agent_tools.convert_currency.invoke(
        {"amount": 1, "from_code": "USD", "to_code": "JPY"}
    )


def test_get_crypto_price_success_and_unsupported_symbol(monkeypatch):
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "bitcoin": {
                "usd": 65000,
                "pkr": 18000000,
                "usd_24h_change": 1.25,
                "pkr_24h_change": 1.3,
            }
        }),
    )
    result = agent_tools.get_crypto_price.invoke({"symbol": "BTC"})
    assert "$65,000.00" in result
    assert "PKR 18,000,000.00" in result
    assert "1.25%" in result
    assert "Unsupported symbol" in agent_tools.get_crypto_price.invoke({"symbol": "NOPE"})


def test_get_cricket_score_success_and_daily_limit(monkeypatch):
    assert "not configured" in agent_tools.get_cricket_score.invoke({})
    monkeypatch.setenv("CRICAPI_KEY", "test-cricket-key")
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "status": "success",
            "data": [{
                "teams": ["Pakistan", "Australia"],
                "score": [{"inning": "Pakistan Inning", "r": 200, "w": 4, "o": 40}],
                "status": "Pakistan need 100 runs",
            }],
        }),
    )
    result = agent_tools.get_cricket_score.invoke({"team": "pak"})
    assert "Pakistan vs Australia" in result
    assert "200/4" in result

    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({}, status_code=429),
    )
    assert "daily limit reached, try tomorrow" in agent_tools.get_cricket_score.invoke({})


def test_get_movie_info_success_and_not_found(monkeypatch):
    assert "not configured" in agent_tools.get_movie_info.invoke({"title": "Example Film"})
    monkeypatch.setenv("OMDB_API_KEY", "test-omdb-key")
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "Response": "True",
            "Title": "Example Film",
            "Year": "2020",
            "Runtime": "100 min",
            "Genre": "Drama",
            "imdbRating": "8.1",
            "Plot": "A short plot.",
        }),
    )
    result = agent_tools.get_movie_info.invoke({"title": "Example Film"})
    assert "Example Film (2020)" in result
    assert "IMDb 8.1" in result
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "Response": "False",
            "Error": "Movie not found!",
        }),
    )
    assert "Movie not found" in agent_tools.get_movie_info.invoke({"title": "Missing"})


def test_get_datetime_success_and_invalid_zone():
    result = agent_tools.get_datetime.invoke({
        "timezone_name": "Asia/Karachi",
        "convert_from": "UTC",
    })
    assert "UTC" in result and "Asia/Karachi" in result
    assert "Unknown timezone" in agent_tools.get_datetime.invoke(
        {"timezone_name": "Not/A_Real_Zone"}
    )


def test_generate_qr_code_saves_user_scoped_png(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_tools, "GENERATED_IMAGE_DIR", tmp_path / "images")
    token = agent_tools.active_user_id.set("qr-user")
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse(
            status_code=200,
            headers={"content-type": "image/png"},
            content=_png_bytes(),
        ),
    )
    try:
        result = agent_tools.generate_qr_code.invoke({"text": "https://example.com"})
    finally:
        agent_tools.active_user_id.reset(token)
    assert result.startswith(agent_tools.GENERATED_IMAGE_MARKER)
    filename = result.removeprefix(agent_tools.GENERATED_IMAGE_MARKER)
    saved = agent_tools.get_generated_image_dir("qr-user") / filename
    assert saved.is_file()
    with Image.open(saved) as image:
        assert image.format == "PNG"
    assert "non-empty text" in agent_tools.generate_qr_code.invoke({"text": ""})


def test_generate_password_guarantees_classes_and_validates_length():
    password = agent_tools.generate_password.invoke({"length": 24})
    assert len(password) == 24
    assert any(char.isupper() for char in password)
    assert any(char.islower() for char in password)
    assert any(char.isdigit() for char in password)
    assert any(not char.isalnum() for char in password)
    assert "between 8 and 64" in agent_tools.generate_password.invoke({"length": 7})


def test_define_word_success_and_unknown_word(monkeypatch):
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse([{
            "meanings": [{
                "partOfSpeech": "noun",
                "definitions": [
                    {"definition": "A small feline."},
                    {"definition": "A domesticated animal."},
                ],
                "synonyms": ["feline", "kitty", "puss", "mouser"],
            }]
        }]),
    )
    result = agent_tools.define_word.invoke({"word": "cat"})
    assert "noun" in result
    assert "A small feline." in result and "domesticated animal" in result
    assert "feline, kitty, puss" in result
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({}, status_code=404),
    )
    assert "No dictionary definition found" in agent_tools.define_word.invoke(
        {"word": "notaword"}
    )


def test_get_cat_fact_success_and_missing_fact(monkeypatch):
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({"fact": "Cats sleep for many hours."}),
    )
    assert agent_tools.get_cat_fact.invoke({}) == "Cats sleep for many hours."
    monkeypatch.setattr(
        agent_tools.httpx,
        "get",
        lambda *args, **kwargs: FakeResponse({}),
    )
    assert "no fact" in agent_tools.get_cat_fact.invoke({})


def test_http_timeout_is_reported_without_network(monkeypatch):
    def timeout(*args, **kwargs):
        raise agent_tools.httpx.TimeoutException("mock timeout")

    monkeypatch.setattr(agent_tools.httpx, "get", timeout)
    result = agent_tools.get_cat_fact.invoke({})
    assert "timed out" in result


def test_send_email_uses_verified_user_and_ignores_recipient_override(
    monkeypatch, email_database
):
    monkeypatch.setenv("RESEND_API_KEY", "test-resend-key")
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        return FakeResponse({"id": "email-test-id"})

    monkeypatch.setattr(agent_tools.httpx, "post", post)
    schema_fields = agent_tools.send_email.get_input_schema().model_fields
    assert "to" not in schema_fields and "recipient" not in schema_fields
    result = agent_tools.send_email.invoke({
        "subject": "A note",
        "body": "x" * 2500,
        "to": "attacker@example.com",
    })
    assert "Email sent" in result
    assert sent["json"]["to"] == ["verified@example.com"]
    assert len(sent["json"]["text"]) == 2000
    assert sent["timeout"] == 10


def test_send_email_daily_cap_and_unconfigured_key(monkeypatch, email_database):
    assert "not configured" in agent_tools.send_email.invoke(
        {"subject": "Subject", "body": "Body"}
    )
    monkeypatch.setenv("RESEND_API_KEY", "test-resend-key")
    monkeypatch.setenv("EMAIL_DAILY_LIMIT", "1")
    monkeypatch.setattr(
        agent_tools.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse({"id": "first-email"}),
    )
    assert "Email sent" in agent_tools.send_email.invoke(
        {"subject": "Subject", "body": "Body"}
    )
    assert "daily email limit reached" in agent_tools.send_email.invoke(
        {"subject": "Subject", "body": "Body"}
    ).lower()
    assert "subject cannot be empty" in agent_tools.send_email.invoke(
        {"subject": " ", "body": "Body"}
    )


def test_all_extra_tools_are_registered():
    extra_names = {
        "get_weather", "get_prayer_times", "convert_currency", "get_crypto_price",
        "get_cricket_score", "get_movie_info", "get_datetime", "generate_qr_code",
        "generate_password", "define_word", "get_cat_fact", "send_email",
    }
    registered_names = [item.name for item in graph.TOOLS]
    assert extra_names <= set(registered_names)
    assert all(registered_names.count(name) == 1 for name in extra_names)
