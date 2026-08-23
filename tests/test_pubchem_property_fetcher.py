"""Test the pubchem_property_fetcher module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from pubchem_property_fetcher import (
    clean_smiles,
    fetch_cid_bulk,
    fetch_properties_bulk,
    fetch_synonyms_bulk,
    main,
    pick_label,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pubchem_property_fetcher import PubChemClient

pytestmark = pytest.mark.anyio


def test_clean_smiles() -> None:
    assert clean_smiles(["OCC.CC", "CCO", "invalid", "CCC"]) == ["CCO", "CC", "CCC"]


async def test_fetch_cid_bulk(client: PubChemClient) -> None:
    result = await fetch_cid_bulk(["CCC", "CCO", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"], client)

    assert result == {
        "CCC": 6334,
        "CCO": 702,
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1": 3672,
    }


async def test_fetch_properties_bulk(client: PubChemClient) -> None:
    result = await fetch_properties_bulk(
        [6334, 702, 3672], ["IUPACName", "MolecularWeight"], client
    )

    assert result == {
        6334: {"IUPACName": "propane", "MolecularWeight": "44.10"},
        702: {"IUPACName": "ethanol", "MolecularWeight": "46.07"},
        3672: {
            "IUPACName": "2-[4-(2-methylpropyl)phenyl]propanoic acid",
            "MolecularWeight": "206.28",
        },
    }


async def test_fetch_synonyms_bulk(client: PubChemClient) -> None:
    cid_to_syns = await fetch_synonyms_bulk([6334, 702, 3672], client)
    cid_to_syns = {k: v[:3] for k, v in cid_to_syns.items()}
    assert cid_to_syns == {
        6334: ["PROPANE", "Dimethylmethane", "Propyl hydride"],
        702: ["ethanol", "ethyl alcohol", "alcohol"],
        3672: ["ibuprofen", "IBUPROFEN, (+-)-", "Brufen"],
    }


def test_pick_label() -> None:
    assert pick_label(["A", "B"], "C") == "a"
    assert pick_label([], "C") == "c"
    assert pick_label([], None) is None


def test_main(tmp_path: Path) -> None:
    main(["OCC.CC", "CCO", "invalid", "CCC"], properties=["label"], output=tmp_path / "output.csv")
    # https://github.com/python/cpython/issues/106749 (Python 3.11 only)
    assert pl.read_csv(tmp_path / "output.csv").equals(  # pragma: no cover
        pl.DataFrame({"SMILES": ["CCO", "CC", "CCC"], "label": ["ethanol", "ethane", "propane"]})
    )
