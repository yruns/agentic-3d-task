"""HTTP envelope and admission invariants without SQLite or codec dependencies."""

from traceback import format_exception

import pytest

from codex_adapter.application.requests import parse_envelope
from codex_adapter.config import ServerSettings
from codex_adapter.domain.errors import AdapterError
from codex_adapter.interfaces.admission import Admission


@pytest.mark.parametrize(
    "number",
    [
        pytest.param("1e" + "9" * 40, id="huge-exponent"),
        pytest.param("9" * 5000, id="huge-integer"),
    ],
)
def test_native_envelope_preserves_arbitrarily_large_number_lexemes(
    number: str,
) -> None:
    raw = ('{"model":"gpt-test","future":' + number + "}").encode()
    assert parse_envelope(raw).raw == raw


@pytest.mark.parametrize("raw", [b'{"model":1}', b'{"model":"gpt-test","stream":1}'])
def test_numeric_routing_fields_are_not_coerced(raw: bytes) -> None:
    with pytest.raises(AdapterError):
        parse_envelope(raw)


def test_invalid_json_has_no_sensitive_exception_context() -> None:
    raw = b'{"model":"gpt-test","password":"sensitive-marker",'
    with pytest.raises(AdapterError) as raised:
        parse_envelope(raw)
    assert raised.value.__context__ is None
    assert "sensitive-marker" not in "".join(format_exception(raised.value))


def test_admission_enforces_rate_and_concurrency() -> None:
    now = [0.0]
    admission = Admission(
        ServerSettings(max_concurrent_requests=1, requests_per_minute=2),
        clock=lambda: now[0],
    )
    admission.acquire()
    with pytest.raises(AdapterError) as busy:
        admission.acquire()
    assert busy.value.status_code == 503
    admission.release()
    admission.acquire()
    admission.release()
    with pytest.raises(AdapterError) as limited:
        admission.acquire()
    assert limited.value.status_code == 429
    now[0] = 60.0
    admission.acquire()
    admission.release()


def test_admission_double_release_is_a_programming_error() -> None:
    admission = Admission(ServerSettings())
    with pytest.raises(RuntimeError):
        admission.release()


def test_application_defines_an_upstream_transport_port() -> None:
    from codex_adapter.application.ports import UpstreamTransport

    assert callable(UpstreamTransport.open)
    assert callable(UpstreamTransport.read)
    assert callable(UpstreamTransport.close)
