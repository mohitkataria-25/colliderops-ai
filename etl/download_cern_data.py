from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "collider_events.csv"
SEARCH_DATA_PATH = RAW_DATA_DIR / "cern_search_results.json"
DATA_URL = "https://opendata.cern.ch"


FALLBACK_ROWS = [
    {
        "event_id": "E001",
        "DER_mass_MMC": 138.4,
        "DER_mass_transverse_met_lep": 51.6,
        "DER_mass_vis": 97.8,
        "PRI_tau_pt": 32.6,
        "PRI_lep_pt": 44.1,
        "label": "signal",
    },
    {
        "event_id": "E002",
        "DER_mass_MMC": 92.3,
        "DER_mass_transverse_met_lep": 76.2,
        "DER_mass_vis": 65.4,
        "PRI_tau_pt": 21.2,
        "PRI_lep_pt": 28.4,
        "label": "background",
    },
    {
        "event_id": "E003",
        "DER_mass_MMC": 145.8,
        "DER_mass_transverse_met_lep": 49.1,
        "DER_mass_vis": 102.2,
        "PRI_tau_pt": 35.7,
        "PRI_lep_pt": 46.9,
        "label": "signal",
    },
    {
        "event_id": "E004",
        "DER_mass_MMC": 80.1,
        "DER_mass_transverse_met_lep": 88.4,
        "DER_mass_vis": 58.3,
        "PRI_tau_pt": 18.4,
        "PRI_lep_pt": 24.1,
        "label": "background",
    },
    {
        "event_id": "E005",
        "DER_mass_MMC": 132.7,
        "DER_mass_transverse_met_lep": 55.8,
        "DER_mass_vis": 94.6,
        "PRI_tau_pt": 31.2,
        "PRI_lep_pt": 42.5,
        "label": "signal",
    },
    {
        "event_id": "E006",
        "DER_mass_MMC": 70.4,
        "DER_mass_transverse_met_lep": 91.5,
        "DER_mass_vis": 52.7,
        "PRI_tau_pt": 16.8,
        "PRI_lep_pt": 22.3,
        "label": "background",
    },
]
def ensure_directory_exists():
    """ create raw directory if it does not exists"""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def search_cern_records (query:str, size:int)->dict | bool:
    """ search cern database based on query """
    try:
        search_url = f"{DATA_URL}/api/records/?q={query}&size={size}"
        search_response = requests.get(search_url, timeout=30)
        search_response.raise_for_status()
        return search_response.json()
    except requests.RequestException as error:
        print (f"An error occured with retreiving cern data {error}")
        return False
    except ValueError as error:
        print(f"CERN response was not a valid JSON {error}")
        return False

def save_search_results(results: dict, output_path: Path) -> None:

    if not results:
        raise ValueError("No search results were returned")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=2)
    
    print(f"Search reulsts saved at {output_path}")

def write_fallback_dataset(rows:Iterable[dict], output_path: Path):
    rows = list(rows)
    if not rows:
        raise ValueError("Fallback dataset is empty")
    
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(query, size) ->None:

    ensure_directory_exists()

    results = search_cern_records(query=query,size=size)


    if results:
        print(f"CERN Dataset is downloaded at the folliwing dir {SEARCH_DATA_PATH}")
        save_search_results(results=results, output_path=SEARCH_DATA_PATH) 
    
    else:
        print("CERN search failed. Continuing with fallback training dataset.")
    
    write_fallback_dataset(rows=FALLBACK_ROWS, output_path=RAW_DATA_PATH)
    
    print (f"Search results available at {SEARCH_DATA_PATH}")
    print(f"Raw dataset available at {RAW_DATA_PATH}")

if __name__ == "__main__":
    main(query="CMS collision data", size=1)