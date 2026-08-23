"""Tests for failure and retry paths against a mocked PubChem server."""

from __future__ import annotations

import functools
import gzip
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import yaspin
from conftest import fetch_all_handler, gz_response, redirect_html

import pubchem_property_fetcher
from pubchem_property_fetcher import (
    PUBCHEM_BASE_URL,
    PubChemClient,
    SpinnerDummy,
    call_with_backoff,
    fetch_all,
    fetch_cid_bulk,
    fetch_properties_bulk,
    fetch_synonyms_bulk,
    fetch_with_backoff,
    main,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    Handler = Callable[[httpx.Request], httpx.Response]
    ClientFactory = Callable[[], PubChemClient]

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def mock_client(handler: Handler) -> AsyncGenerator[PubChemClient]:
    """A `PubChemClient` whose session is swapped for one backed by a mock transport."""
    client = PubChemClient()
    await client.session.aclose()
    client._session = httpx.AsyncClient(  # noqa: SLF001
        transport=httpx.MockTransport(handler), base_url=PUBCHEM_BASE_URL
    )
    try:
        yield client
    finally:
        await client.session.aclose()


def mock_pubchem_client_factory(handler: Handler) -> ClientFactory:
    """A factory suitable for monkeypatching `PubChemClient`."""

    def factory() -> PubChemClient:
        client = PubChemClient()
        client._session = httpx.AsyncClient(  # noqa: SLF001
            transport=httpx.MockTransport(handler), base_url=PUBCHEM_BASE_URL
        )
        return client

    return factory


async def test_fetch_with_backoff_non_recoverable_status() -> None:
    calls = 0

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
        result = await fetch_with_backoff("https://example.test/x", session, max_retries=3)

    assert result is None
    assert calls == 1


async def test_fetch_with_backoff_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)
    calls = 0

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
        result = await fetch_with_backoff("https://example.test/x", session, max_retries=3)

    assert result is None
    assert calls == 3


async def test_fetch_with_backoff_recovers_after_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, text="ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as session:
        result = await fetch_with_backoff("https://example.test/x", session, max_retries=3)

    assert result is not None
    assert result.text == "ok"
    assert calls == 2


async def test_call_with_backoff_non_recoverable_status() -> None:
    calls = 0

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    async with mock_client(handler) as client:
        result = await call_with_backoff(
            functools.partial(client.get_properties, "1", "MolecularWeight")
        )

    assert result is None
    assert calls == 1


async def test_call_with_backoff_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)
    calls = 0

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with mock_client(handler) as client:
        result = await call_with_backoff(
            functools.partial(client.get_properties, "1", "MolecularWeight"), max_retries=3
        )

    assert result is None
    assert calls == 3


async def test_call_with_backoff_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    async with mock_client(handler) as client:
        result = await call_with_backoff(
            functools.partial(client.get_properties, "1", "MolecularWeight"), max_retries=2
        )

    assert result is None
    assert calls == 2


async def test_fetch_cid_bulk_no_link_found() -> None:
    def handler(dummy_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>no link here</html>")

    async with mock_client(handler) as client:
        with pytest.raises(ValueError, match="No download or redirect link found"):
            await fetch_cid_bulk(["CCO"], client)


async def test_fetch_cid_bulk_batch_never_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "POLL_DELAY", 0.001)
    redirect_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/pending.html"

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=redirect_html(redirect_url))

    async with mock_client(handler) as client:
        with pytest.raises(ValueError, match="Batch processing not finished"):
            await fetch_cid_bulk(["CCO"], client)


async def test_fetch_cid_bulk_download_fails_repeatedly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)
    monkeypatch.setattr(pubchem_property_fetcher, "POLL_DELAY", 0.001)
    gz_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/result.txt.gz"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".gz"):
            return httpx.Response(503)
        return httpx.Response(200, text=redirect_html(gz_url))

    async with mock_client(handler) as client:
        with pytest.raises(ValueError, match="File download failed repeatedly"):
            await fetch_cid_bulk(["CCO"], client)


async def test_fetch_cid_bulk_submit_fails_repeatedly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "POLL_DELAY", 0.001)

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with mock_client(handler) as client:
        with pytest.raises(ValueError, match="Submitting batch job failed"):
            await fetch_cid_bulk(["CCO"], client)


async def test_fetch_cid_bulk_polling_recovers_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "POLL_DELAY", 0.001)
    redirect_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/pending.html"
    gz_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/result.txt.gz"
    poll_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_calls
        if request.url.path.endswith(".gz"):
            buf = BytesIO()
            with gzip.open(buf, "wb") as f:
                f.write(b"CCO\t702\n")
            return httpx.Response(200, content=buf.getvalue())
        if request.method == "POST":
            return httpx.Response(200, text=redirect_html(redirect_url))

        poll_calls += 1
        if poll_calls == 1:
            return httpx.Response(500)
        return httpx.Response(200, text=redirect_html(gz_url))

    async with mock_client(handler) as client:
        result = await fetch_cid_bulk(["CCO"], client)

    assert result == {"CCO": 702}
    assert poll_calls >= 2


async def test_fetch_properties_bulk_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with mock_client(handler) as client:
        result = await fetch_properties_bulk([1, 2], ["MolecularWeight"], client)

    assert result == {}


async def test_fetch_properties_bulk_non_recoverable() -> None:
    calls = 0

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    async with mock_client(handler) as client:
        result = await fetch_properties_bulk([1, 2], ["MolecularWeight"], client)

    assert result == {}
    assert calls == 1


async def test_fetch_synonyms_bulk_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem_property_fetcher, "BACKOFF_DELAY", 0.001)

    def handler(dummy_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with mock_client(handler) as client:
        result = await fetch_synonyms_bulk([1, 2], client)

    assert result == {}


async def test_fetch_all_verbose_uses_dummy_spinner_not_yaspin(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(fetch_all_handler)
    )

    spinner_instances: list[SpinnerDummy] = []
    real_spinner_dummy = SpinnerDummy

    def spying_spinner_dummy(*args: str, **kwargs: str) -> SpinnerDummy:
        instance = real_spinner_dummy(*args, **kwargs)
        spinner_instances.append(instance)
        return instance

    def failing_yaspin(*dummy_args: object, **dummy_kwargs: object) -> None:
        raise AssertionError("yaspin.yaspin should not run when verbose=True")  # pragma: no cover

    monkeypatch.setattr(pubchem_property_fetcher, "SpinnerDummy", spying_spinner_dummy)
    monkeypatch.setattr(yaspin, "yaspin", failing_yaspin)

    with caplog.at_level(logging.INFO, logger="pubchem_property_fetcher"):
        smiles_to_cid, cid_to_props = await fetch_all(
            ["CCO"], ["MolecularWeight"], None, 5, verbose=True
        )

    assert smiles_to_cid == {"CCO": 702}
    assert cid_to_props == {702: {"MolecularWeight": "46.07"}}

    # dummy_spinner ran
    assert len(spinner_instances) == 1
    assert spinner_instances[0].text == "Fetching properties..."

    # log messages actually went out
    assert any("Unique canonical SMILES to look up" in r.message for r in caplog.records)


async def test_fetch_all_formats_synonyms_and_removes_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(fetch_all_handler)
    )
    properties = ["synonyms", "MolecularWeight"]

    smiles_to_cid, cid_to_props = await fetch_all(["CCO"], properties, None, 5, verbose=False)

    assert smiles_to_cid == {"CCO": 702}
    assert cid_to_props[702]["MolecularWeight"] == "46.07"
    assert cid_to_props[702]["synonyms"] == "ethanol|EtOH"
    assert properties == ["MolecularWeight"]


def test_main_prints_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(fetch_all_handler)
    )

    main(["CCO"], properties=["MolecularWeight"], output=Path("-"))

    out = capsys.readouterr().out
    assert "MolecularWeight" in out
    assert "46.07" in out


async def test_fetch_all_label_without_iupac_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(fetch_all_handler)
    )
    properties = ["label"]

    _, cid_to_props = await fetch_all(["CCO"], properties, None, 5, verbose=False)

    assert cid_to_props[702] == {"label": "ethanol"}
    assert properties == ["IUPACName"]


async def test_fetch_all_label_with_iupac_also_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(fetch_all_handler)
    )
    properties = ["label", "IUPACName"]

    _, cid_to_props = await fetch_all(["CCO"], properties, None, 5, verbose=False)

    assert cid_to_props[702] == {"IUPACName": "ethanol", "label": "ethanol"}
    assert properties == ["IUPACName"]


async def test_fetch_all_no_valid_smiles() -> None:
    with pytest.raises(SystemExit) as excinfo:
        await fetch_all(["invalid"], ["IUPACName"], None, 5, verbose=False)

    assert excinfo.value.code == 1


async def test_fetch_all_no_cid_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        gz_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/result.txt.gz"
        if request.method == "POST":
            return httpx.Response(200, text=redirect_html(gz_url))
        return gz_response({})

    monkeypatch.setattr(
        pubchem_property_fetcher, "PubChemClient", mock_pubchem_client_factory(handler)
    )

    with pytest.raises(SystemExit) as excinfo:
        await fetch_all(["CCO"], ["IUPACName"], None, 5, verbose=False)

    assert excinfo.value.code == 1
