"""Resolve SMILES to CIDs and fetch user-defined properties from PubChem."""

from __future__ import annotations

import asyncio
import functools
import gzip
import http
import logging
import re
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypeVar

import doctyper
import httpx
import yaspin
from mxhttp import AsyncConsumer, Part, PartValue, get, post
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Generator, Sequence

__version__ = "0.1.2"

logger = logging.getLogger(__name__)

T = TypeVar("T")

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
RATE_LIMIT = 5.0  # requests per second
BACKOFF_DELAY = 1.0  # seconds
BACKOFF_EXPONENT = 1.5
POLL_DELAY = 1.0  # seconds


class HasText(Protocol):
    """Protocol with single text attribute."""

    text: str


@dataclass
class SpinnerDummy:
    """Dummy spinner class."""

    text: str = ""


class PubChemClient(AsyncConsumer):
    """Declarative async client for PubChem's ID Exchange and PUG REST endpoints."""

    def __init__(self) -> None:
        """Binds the client to PubChem's base URL."""
        super().__init__(PUBCHEM_BASE_URL)

    @post("/idexchange/idexchange.cgi")
    async def submit_batch(  # type: ignore[empty-body] # noqa: PLR0913,PLR0917
        self,
        idstr: Annotated[PartValue, Part],
        inputtype: Annotated[PartValue, Part] = (None, "smiles"),
        inputdsn: Annotated[PartValue, Part] = (None, ""),
        idinput: Annotated[PartValue, Part] = (None, "str"),
        idfile: Annotated[PartValue, Part] = ("", "", "application/octet-stream"),
        operatortype: Annotated[PartValue, Part] = (None, "samecid"),
        outputtype: Annotated[PartValue, Part] = (None, "cid"),
        outputdsn: Annotated[PartValue, Part] = (None, ""),
        method: Annotated[PartValue, Part] = (None, "file-pair"),
        compression: Annotated[PartValue, Part] = (None, "gzip"),
        submitjob: Annotated[PartValue, Part] = (None, "Submit Job"),
        xmlfile: Annotated[PartValue, Part] = ("", "", "application/octet-stream"),
    ) -> str:
        """Submits a SMILES batch to PubChem's ID Exchange, returning the response HTML."""

    @get("/rest/pug/compound/cid/{cids}/property/{props}/JSON")
    async def get_properties(self, cids: str, props: str) -> dict[str, Any]:  # type: ignore[empty-body]
        """Fetches the JSON property table for the given comma-separated CIDs."""

    @get("/rest/pug/compound/cid/{cids}/synonyms/JSON")
    async def get_synonyms(self, cids: str) -> dict[str, Any]:  # type: ignore[empty-body]
        """Fetches the JSON synonym list for the given comma-separated CIDs."""


async def fetch_with_backoff(
    url: str, session: httpx.AsyncClient, max_retries: int = 5
) -> httpx.Response | None:
    """Fetch a raw URL using exponential backoff for rate limits and server errors."""
    delay = BACKOFF_DELAY
    for _ in range(max_retries):
        try:
            r = await session.get(url, timeout=30)
            if r.status_code == http.HTTPStatus.OK:
                return r

            # Retry on rate limit or server errors
            if r.status_code in {429, 500, 502, 503, 504}:
                logger.warning("HTTP %d, retrying in %.1fs...", r.status_code, delay)
            else:
                logger.error("HTTP %d: Request failed non-recoverably.", r.status_code)
                return None
        except httpx.HTTPError as e:
            logger.warning("Request exception: %s. Retrying in %.1fs...", e, delay)

        await asyncio.sleep(delay)
        delay *= BACKOFF_EXPONENT

    logger.error("Max retries reached for %s", url)
    return None


async def call_with_backoff(
    call: Callable[[], Coroutine[Any, Any, T]], max_retries: int = 5
) -> T | None:
    """Calls a REST API endpoint using exponential backoff for retryable failures."""
    delay = BACKOFF_DELAY
    for _ in range(max_retries):
        try:
            return await call()
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in {429, 500, 502, 503, 504}:
                logger.warning("HTTP %d, retrying in %.1fs...", code, delay)
            else:
                logger.exception("HTTP %d: Request failed non-recoverably.", code)
                return None
        except httpx.HTTPError as e:
            logger.warning("Request exception: %s. Retrying in %.1fs...", e, delay)

        await asyncio.sleep(delay)
        delay *= BACKOFF_EXPONENT

    logger.error("Max retries reached")
    return None


def synonym_score(s: str, ix: int, index_factor: float) -> float:
    """Calculate a heuristic score for a given synonym string to determine its usability."""
    score = -ix * index_factor

    # simple lowercase common-looking name
    if re.fullmatch(r"[A-Za-z ]+", s):
        score += 5 * index_factor

    if len(s) <= 3:  # noqa: PLR2004
        score -= 5 * index_factor

    if re.fullmatch(r"\d{2,7}-\d{2}-\d", s):
        score -= 100 * index_factor

    return score


def sort_synonyms(synonyms: Sequence[str], index_factor: float = 1000.0) -> list[str]:
    """Sort a sequence of synonyms based on a calculated heuristic score."""
    scored = [(synonym_score(s, i, index_factor), s) for i, s in enumerate(synonyms)]

    return [s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)]


def pick_label(synonyms: list[str], iupac: str | None) -> str | None:
    """Select the most readable label, preferring the first synonym over the IUPAC name."""
    for s in synonyms:
        return s.lower()
    if iupac:
        return iupac.lower()
    return None


def clean_smiles(raw_inputs: Sequence[str]) -> list[str]:
    """Split fragments, canonicalize with RDKit, and deduplicate the given SMILES.

    Args:
        raw_inputs: A sequence of raw SMILES strings.

    Returns:
        A list of unique, canonicalized SMILES strings.
    """
    from rdkit import Chem, rdBase

    rdBase.DisableLog("rdApp.*")

    # Flatten: "A.B" → "A", "B"
    fragments = [s.strip() for expr in raw_inputs for s in expr.split(".") if s.strip()]

    canons: list[str] = []

    for frag in fragments:
        mol: Chem.Mol | None = Chem.MolFromSmiles(frag)
        if mol is None:
            logger.warning("skipping invalid SMILES: %s", frag)
            continue
        canons.append(Chem.MolToSmiles(mol))

    return list(dict.fromkeys(canons))


async def rate_sleep(prev: float) -> None:
    """Pause execution to ensure the API request rate limit is not exceeded."""
    elapsed = time.monotonic() - prev
    delay = 1.0 / RATE_LIMIT - elapsed
    if delay > 0:  # pragma: no branch
        await asyncio.sleep(delay)


async def fetch_cid_bulk(  # noqa: C901,PLR0912
    smiles_list: Sequence[str],
    client: PubChemClient,
    redirect_url: str | None = None,
) -> dict[str, int]:
    """Resolve a batch of SMILES to their corresponding PubChem CIDs.

    Args:
        smiles_list: A list of canonical SMILES strings.
        client: The declarative PubChem client to use for HTTP calls.
        redirect_url: An existing batch job URL to poll.

    Returns:
        A dictionary mapping the original SMILES strings to their PubChem CIDs.

    Raises:
        ValueError: If the file download fails, no link is found, or the batch job fails.
    """
    if not smiles_list:  # pragma: no cover
        return {}

    status_code: int | None = None
    delay = POLL_DELAY

    for _ in range(10):
        if redirect_url:
            logger.info("Batch processing in progress, polling results...")
            r = await client.session.get(redirect_url)
            status_code = r.status_code
            if r.status_code != http.HTTPStatus.OK:  # pragma: no cover
                logger.warning("Request failed with status code: %d", r.status_code)
                await asyncio.sleep(delay)
                continue
            text = r.text
        else:
            logger.info("Submitting batch job...")
            try:
                text = await client.submit_batch(idstr=(None, "\n".join(smiles_list)))
                status_code = http.HTTPStatus.OK
            except httpx.HTTPStatusError as e:  # pragma: no cover
                status_code = e.response.status_code
                logger.warning("Request failed with status code: %d", status_code)
                await asyncio.sleep(delay)
                continue

        match = re.search(
            r'document.location.replace\("(https://pubchem.ncbi.nlm.nih.gov/[^"]+)"',
            text,
        )

        if match:
            url = match.group(1)

            if url.endswith(".gz"):
                logger.info("Batch processing finished, downloading results...")
                fr = await fetch_with_backoff(url, client.session)
                if fr:
                    with gzip.open(BytesIO(fr.content), "rt") as f:
                        smiles_to_cid: dict[str, int] = {}
                        for line in f.read().splitlines():
                            smiles, cid = line.split("\t", maxsplit=1)
                            smiles_to_cid[smiles] = int(cid)
                        return smiles_to_cid
                else:  # pragma: no cover
                    raise ValueError("File download failed repeatedly.")

            redirect_url = url
            await asyncio.sleep(delay)
        else:
            raise ValueError("No download or redirect link found")

    if redirect_url:
        raise ValueError(
            f"Batch processing not finished, reuse link: {urllib.parse.quote(redirect_url)}"
        )

    raise ValueError(f"Submitting batch job failed, last status code: {status_code}")


async def fetch_properties_bulk(
    cids: Sequence[int],
    properties: Sequence[str],
    client: PubChemClient,
    verbose: bool = False,
) -> dict[int, dict[str, Any]]:
    """Fetch user-defined properties for a list of CIDs in batches.

    PubChem accepts up to 5000 CIDs per request, so lists exceeding this limit
    are automatically batched.

    Args:
        cids: A list of PubChem Compound IDs.
        properties: A list of PubChem properties (e.g. 'IUPACName', 'MolecularWeight').
        client: The declarative PubChem client to use for HTTP calls.
        verbose: Whether to display a progress bar.

    Returns:
        A dictionary mapping CIDs to their requested properties.
    """
    if not cids or not properties:  # pragma: no cover
        return {}

    results: dict[int, dict[str, Any]] = {}
    t0 = time.monotonic()
    props_str = ",".join(properties)

    for batch_start in tqdm(
        range(0, len(cids), 5000), desc="Fetching properties...", disable=not verbose
    ):
        await rate_sleep(t0)
        t0 = time.monotonic()

        batch = list(cids[batch_start : batch_start + 5000])
        cid_str = ",".join(str(c) for c in batch)

        payload = await call_with_backoff(
            functools.partial(client.get_properties, cid_str, props_str)
        )
        if payload:
            props = payload.get("PropertyTable", {}).get("Properties", [])
            for row in props:
                cid = row.pop("CID")
                if cid is not None:  # pragma: no branch
                    results[cid] = row

    return results


async def fetch_synonyms_bulk(
    cids: Sequence[int], client: PubChemClient, verbose: bool = False
) -> dict[int, list[str]]:
    """Fetch common synonyms for a list of CIDs.

    PubChem accepts up to 5000 CIDs per request, so lists exceeding this limit
    are automatically batched.

    Args:
        cids: A list of PubChem Compound IDs.
        client: The declarative PubChem client to use for HTTP calls.
        verbose: Whether to display a progress bar.

    Returns:
        A dictionary mapping CIDs to lists of formatted synonym strings.
    """
    if not cids:  # pragma: no cover
        return {}

    results: dict[int, list[str]] = {}
    t0 = time.monotonic()

    for batch_start in tqdm(
        range(0, len(cids), 5000), desc="Fetching synonyms...", disable=not verbose
    ):
        await rate_sleep(t0)
        t0 = time.monotonic()
        batch = list(cids[batch_start : batch_start + 5000])
        cid_str = ",".join(str(c) for c in batch)

        payload = await call_with_backoff(functools.partial(client.get_synonyms, cid_str))
        if payload:
            infos = payload.get("InformationList", {}).get("Information", [])
            for info in infos:
                cid = info.get("CID")
                syns = info.get("Synonym", [])
                if cid is not None:  # pragma: no branch
                    results[cid] = sort_synonyms(syns)

    return results


async def fetch_all(  # noqa: C901
    smiles_inputs: Sequence[str],
    properties: list[str],
    batch_url: str | None,
    max_synonyms: int,
    verbose: bool,
) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    """Drive the CID/property/synonym lookups for `main` over a single declarative client."""

    @contextmanager
    def dummy_spinner() -> Generator[SpinnerDummy]:
        yield SpinnerDummy()

    spinner_fn = dummy_spinner if verbose else yaspin.yaspin

    async with PubChemClient() as client:
        with spinner_fn() as sp:
            sp.text = "Cleaning SMILES..."
            canonical_smiles = clean_smiles(smiles_inputs)

            if not canonical_smiles:
                logger.error("No valid SMILES after canonicalization.")
                raise SystemExit(1)

            logger.info("Unique canonical SMILES to look up: %d", len(canonical_smiles))

            sp.text = "Requesting CIDs..."
            smiles_to_cid = await fetch_cid_bulk(canonical_smiles, client, batch_url)

            if not smiles_to_cid:
                logger.error("No SMILES resolved to a PubChem CID.")
                raise SystemExit(1)

            logger.info("Resolved %d/%d SMILES → CID", len(smiles_to_cid), len(canonical_smiles))

            cids = list(smiles_to_cid.values())

            cid_to_syns: dict[int, list[str]] | None = None
            fetch_synonyms = "synonyms" in properties
            fetch_label = "label" in properties
            fetch_iupac = "IUPACName" in properties

            if fetch_synonyms or fetch_label:
                sp.text = "Fetching synonyms..."
                cid_to_syns = await fetch_synonyms_bulk(cids, client, verbose=verbose)
                if fetch_synonyms:
                    properties.remove("synonyms")
                if fetch_label:
                    properties.remove("label")
                    if not fetch_iupac:
                        properties.append("IUPACName")

            sp.text = "Fetching properties..."
            cid_to_props = await fetch_properties_bulk(cids, properties, client, verbose)

            if cid_to_syns:
                for cid, syns in cid_to_syns.items():
                    data = cid_to_props[cid]
                    if fetch_synonyms:
                        data["synonyms"] = "|".join(
                            s.replace("|", "\\|") for s in syns[:max_synonyms]
                        )
                    if fetch_label:
                        iupac_name = data["IUPACName"] if fetch_iupac else data.pop("IUPACName")
                        data["label"] = pick_label(syns, iupac_name)

    return smiles_to_cid, cid_to_props


def main(  # noqa: PLR0913,PLR0917
    smiles_inputs: list[str],
    properties: list[str] = ("IUPACName",),  # type:ignore[assignment]
    output: Path | None = None,
    keep_stereo: bool = False,
    batch_url: str | None = None,
    max_synonyms: int | None = None,
    verbose: bool = False,
) -> None:
    """Resolve a list of SMILES to canonical SMILES and text labels, then output to CSV.

    Properties can be looked up in the PubChem PUG docs; special properties are synonyms
    and label (the most readable common name chosen from synonyms, falling back to IUPAC).

    Args:
        smiles_inputs: A list of raw SMILES strings to resolve.
        properties: Properties to fetch.
        output: Path to the output CSV file. If None, prints to stdout.
        keep_stereo: Whether to retain stereochemical information in the SMILES.
        batch_url: An existing PubChem batch URL to poll instead of starting a new job.
        max_synonyms: Number of synonyms to fetch, if synonyms are included.
        verbose: Whether to log detailed process information and show progress bars.
    """
    import polars as pl
    from rdkit import Chem

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-8s %(message)s",
    )

    smiles_to_cid, cid_to_props = asyncio.run(
        fetch_all(smiles_inputs, list(properties), batch_url, max(max_synonyms or 1, 1), verbose)
    )

    # Compile the resulting DataFrame
    rows: list[dict[str, Any]] = []
    for canon, cid in smiles_to_cid.items():
        base_dict = (
            {"SMILES": canon}
            if keep_stereo
            else {"SMILES": Chem.MolToSmiles(Chem.MolFromSmiles(canon), isomericSmiles=False)}
        )

        # Merge properties payload for this CID
        props = cid_to_props.get(cid, {})
        base_dict.update(props)
        rows.append(base_dict)

    df = pl.DataFrame(rows, orient="row")

    if not output or output == Path("-"):
        buffer = StringIO()
        df.write_csv(buffer)
        print(buffer.getvalue().strip())  # noqa: T201
    else:
        # https://github.com/python/cpython/issues/106749 (Python 3.11 only)
        df.write_csv(output)  # pragma: no cover


if __name__ == "__main__":
    app = doctyper.DocTyper()
    app.command()(main)
    app()
