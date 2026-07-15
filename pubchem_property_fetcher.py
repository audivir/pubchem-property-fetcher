"""Resolve SMILES to CIDs and fetch user-defined properties from PubChem."""

from __future__ import annotations

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
from typing import TYPE_CHECKING, Any, Protocol

import doctyper
import yaspin
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import requests


logger = logging.getLogger(__name__)

PUBCHEM_ID_EXCHANGE = "https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi"
PUBCHEM_PROPERTIES = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/{props}/JSON"
)
PUBCHEM_SYNONYMS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/synonyms/JSON"
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


def _fetch_with_backoff(
    url: str, session: requests.Session | None = None, max_retries: int = 5
) -> requests.Response | None:
    """Fetch a URL using exponential backoff for rate limits and server errors."""
    import requests

    if not session:
        session = requests.Session()

    delay = BACKOFF_DELAY
    for _ in range(max_retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == http.HTTPStatus.OK:
                return r

            # Retry on rate limit or server errors
            if r.status_code in {429, 500, 502, 503, 504}:
                logger.warning("HTTP %d, retrying in %.1fs...", r.status_code, delay)
            else:
                logger.error("HTTP %d: Request failed non-recoverably.", r.status_code)
                return None
        except requests.RequestException as e:
            logger.warning("Request exception: %s. Retrying in %.1fs...", e, delay)

        time.sleep(delay)
        delay *= BACKOFF_EXPONENT

    logger.error("Max retries reached for %s", url)
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


def rate_sleep(prev: float) -> None:
    """Pause execution to ensure the API request rate limit is not exceeded."""
    elapsed = time.monotonic() - prev
    delay = 1.0 / RATE_LIMIT - elapsed
    if delay > 0:
        time.sleep(delay)


def fetch_cid_bulk(
    smiles_list: Sequence[str],
    session: requests.Session,
    redirect_url: str | None = None,
) -> dict[str, int]:
    """Resolve a batch of SMILES to their corresponding PubChem CIDs.

    Args:
        smiles_list: A list of canonical SMILES strings.
        session: The requests session to use for HTTP calls.
        redirect_url: An existing batch job URL to poll.

    Returns:
        A dictionary mapping the original SMILES strings to their PubChem CIDs.

    Raises:
        ValueError: If the file download fails, no link is found, or the batch job fails.
    """
    if not smiles_list:  # pragma: no cover
        return {}

    files = {
        "inputtype": (None, "smiles"),
        "inputdsn": (None, ""),
        "idinput": (None, "str"),
        "idstr": (None, "\n".join(smiles_list)),
        "idfile": ("", "", "application/octet-stream"),
        "operatortype": (None, "samecid"),
        "outputtype": (None, "cid"),
        "outputdsn": (None, ""),
        "method": (None, "file-pair"),
        "compression": (None, "gzip"),
        "submitjob": (None, "Submit Job"),
        "xmlfile": ("", "", "application/octet-stream"),
    }

    status_code: int | None = None
    delay = POLL_DELAY

    for _ in range(10):
        if redirect_url:
            logger.info("Batch processing in progress, polling results...")
            r = session.get(redirect_url)
        else:
            logger.info("Submitting batch job...")
            r = session.post(PUBCHEM_ID_EXCHANGE, files=files)  # type: ignore[arg-type]

        status_code = r.status_code
        if r.status_code != http.HTTPStatus.OK:  # pragma: no cover
            logger.warning("Request failed with status code: %d", r.status_code)
            time.sleep(delay)
            continue

        match = re.search(
            r'document.location.replace\("(https://pubchem.ncbi.nlm.nih.gov/[^"]+)"',
            r.text,
        )

        if match:
            url = match.group(1)

            if url.endswith(".gz"):
                logger.info("Batch processing finished, downloading results...")
                fr = _fetch_with_backoff(url, session)
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
            time.sleep(delay)
        else:
            raise ValueError("No download or redirect link found")

    if redirect_url:
        raise ValueError(
            f"Batch processing not finished, reuse link: {urllib.parse.quote(redirect_url)}"
        )

    raise ValueError(f"Submitting batch job failed, last status code: {status_code}")


def fetch_properties_bulk(
    cids: Sequence[int], properties: Sequence[str], session: requests.Session, verbose: bool = False
) -> dict[int, dict[str, Any]]:
    """Fetch user-defined properties for a list of CIDs in batches.

    PubChem accepts up to 5000 CIDs per request, so lists exceeding this limit
    are automatically batched.

    Args:
        cids: A list of PubChem Compound IDs.
        properties: A list of PubChem properties (e.g. 'IUPACName', 'MolecularWeight').
        session: The requests session to use for HTTP calls.
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
        rate_sleep(t0)
        t0 = time.monotonic()

        batch = list(cids[batch_start : batch_start + 5000])
        cid_str = ",".join(str(c) for c in batch)
        url = PUBCHEM_PROPERTIES.format(cids=cid_str, props=props_str)

        r = _fetch_with_backoff(url, session)
        if r:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            for row in props:
                cid = row.pop("CID")
                if cid is not None:
                    results[cid] = row

    return results


def fetch_synonyms_bulk(
    cids: Sequence[int], session: requests.Session, verbose: bool = False
) -> dict[int, list[str]]:
    """Fetch common synonyms for a list of CIDs.

    PubChem accepts up to 5000 CIDs per request, so lists exceeding this limit
    are automatically batched.

    Args:
        cids: A list of PubChem Compound IDs.
        session: The requests session to use for HTTP calls.
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
        rate_sleep(t0)
        t0 = time.monotonic()
        batch = list(cids[batch_start : batch_start + 5000])
        cid_str = ",".join(str(c) for c in batch)
        url = PUBCHEM_SYNONYMS.format(cids=cid_str)

        r = _fetch_with_backoff(url, session)
        if r:
            infos = r.json().get("InformationList", {}).get("Information", [])
            for info in infos:
                cid = info.get("CID")
                syns = info.get("Synonym", [])
                if cid is not None:  # pragma: no branch
                    results[cid] = sort_synonyms(syns)

    return results


def main(  # noqa: C901,PLR0912,PLR0913,PLR0915
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
    import requests
    from rdkit import Chem

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-8s %(message)s",
    )

    session = requests.Session()
    max_synonyms = max(max_synonyms or 1, 1)

    @contextmanager
    def dummy_spinner() -> Generator[SpinnerDummy]:
        yield SpinnerDummy()

    spinner_fn = dummy_spinner if verbose else yaspin.yaspin
    with spinner_fn() as sp:
        sp.text = "Cleaning SMILES..."
        canonical_smiles = clean_smiles(smiles_inputs)

        if not canonical_smiles:  # pragma: no cover
            logger.error("No valid SMILES after canonicalization.")
            raise SystemExit(1)

        logger.info("Unique canonical SMILES to look up: %d", len(canonical_smiles))

        sp.text = "Requesting CIDs..."
        smiles_to_cid = fetch_cid_bulk(canonical_smiles, session, batch_url)

        if not smiles_to_cid:  # pragma: no cover
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
            cid_to_syns = fetch_synonyms_bulk(cids, session, verbose=verbose)
            if fetch_synonyms:
                properties.remove("synonyms")
            if fetch_label:
                properties.remove("label")
                if not fetch_iupac:
                    properties.append("IUPACName")

        sp.text = "Fetching properties..."
        cid_to_props = fetch_properties_bulk(cids, properties, session, verbose)

        if cid_to_syns:
            for cid, syns in cid_to_syns.items():
                data = cid_to_props[cid]
                if fetch_synonyms:
                    data["synonyms"] = "|".join(s.replace("|", "\\|") for s in syns[:max_synonyms])
                if fetch_label:
                    iupac_name = data["IUPACName"] if fetch_iupac else data.pop("IUPACName")
                    data["label"] = pick_label(syns, iupac_name)

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
        df.write_csv(output)


if __name__ == "__main__":
    app = doctyper.DocTyper()
    app.command()(main)
    app()
