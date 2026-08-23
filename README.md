# pubchem-property-fetcher

A CLI tool and Python API to resolve chemical SMILES strings and batch-fetch properties using the PubChem PUG REST API.

It handles SMILES canonicalization via RDKit, groups properties to minimize API calls, respects PubChem's rate limits,
and uses exponential backoff to handle server-side throttling (`429`, `503`). Results are exported as CSV using Polars.

## Installation

Ensure you have Python 3.10+ installed. Install the package including its dependencies:

```bash
pip install pubchem-property-fetcher
```

## Usage

### CLI

Run the script directly from the command line. Pass in SMILES strings, specify the properties, and set an output file. If no output is provided, it prints the CSV to standard output.

```bash
# Fetch MolecularWeight and IUPACName
python -m pubchem_property_fetcher "OCC.CC" "CCO" "CCC" \
    --properties MolecularWeight --properties IUPACName \
    --output results.csv
```

**Special Properties:**

* `--properties synonyms`: Fetches common aliases for the structures. Limit the amount returned with `--max-synonyms N`.
* `--properties label`: A custom pipeline property that heuristically picks the most human-readable name from the synonym list, falling back to the formal IUPAC name if no good synonyms exist.

*For standard properties like `MolecularWeight`, `XLogP`, `ExactMass`, or `TPSA`, see the [PubChem PUG REST Docs](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial).*

### Python API

```python
import asyncio
from typing import Any

from pubchem_property_fetcher import (
    PubChemClient,
    clean_smiles,
    fetch_cid_bulk,
    fetch_properties_bulk,
)


# fetch properties
async def main() -> dict[int, dict[str, Any]]:
    # standardize SMILES
    smiles = clean_smiles(["CCO", "CCC"])

    async with PubChemClient() as client:
        # get PubChem Compound IDs (CIDs)
        cids_map = await fetch_cid_bulk(smiles, client)
        cids = list(cids_map.values())
        return await fetch_properties_bulk(cids, ["IUPACName", "MolecularWeight"], client)


print(asyncio.run(main()))
# {702: {'IUPACName': 'ethanol', 'MolecularWeight': 46.07}, ...}
```
