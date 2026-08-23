"""Configuration and fixtures for the `pubchem_property_fetcher` tests."""

from __future__ import annotations

import gzip
from io import BytesIO
from typing import TYPE_CHECKING

import httpx
import pytest

from pubchem_property_fetcher import PUBCHEM_BASE_URL, PubChemClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[PubChemClient]:
    async with PubChemClient() as client:
        yield client


PROPERTY_VALUES = {"MolecularWeight": "46.07", "IUPACName": "ethanol"}


def redirect_html(url: str) -> str:
    """Build a fake PubChem ID Exchange response body pointing to `url`."""
    return f'<html><script>document.location.replace("{url}", "");</script></html>'


def gz_response(smiles_to_cid: dict[str, int]) -> httpx.Response:
    """Build a mocked ID Exchange gzip result mapping SMILES to CIDs."""
    buf = BytesIO()
    with gzip.open(buf, "wb") as f:
        f.write("".join(f"{s}\t{cid}\n" for s, cid in smiles_to_cid.items()).encode())
    return httpx.Response(200, content=buf.getvalue())


def fetch_all_handler(request: httpx.Request) -> httpx.Response:
    gz_url = f"{PUBCHEM_BASE_URL}/tools/idexchange/result.txt.gz"

    if request.method == "POST":
        return httpx.Response(200, text=redirect_html(gz_url))
    if request.url.path.endswith(".gz"):
        return gz_response({"CCO": 702})
    if "/property/" in request.url.path:
        props = request.url.path.split("/property/")[1].split("/")[0].split(",")
        row = {"CID": 702, **{p: PROPERTY_VALUES[p] for p in props}}
        return httpx.Response(200, json={"PropertyTable": {"Properties": [row]}})
    return httpx.Response(
        200,
        json={"InformationList": {"Information": [{"CID": 702, "Synonym": ["ethanol", "EtOH"]}]}},
    )
