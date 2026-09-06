"""Preserve official source evidence without retaining temporary signed URLs."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/external/independent_check_20260905'
BOULDER = 'https://services.arcgis.com/ePKBjXrBZ2vEEgWd/arcgis/rest/services/Electric_Vehicle_Charging_Station_Data/FeatureServer/0'
PALO_ALTO = 'https://data.paloalto.gov/datasets/194693-electric-vehicle-charging-station-usage-july-2011-dec-2020.download/'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT/'palo_alto_official_sessions.csv'
    if destination.exists():
        raise RuntimeError('Keep the downloaded original unchanged')
    session = requests.Session()
    metadata = session.get(BOULDER, params={'f':'json'}, timeout=30)
    metadata.raise_for_status()
    recent = session.get(BOULDER+'/query', params={'f':'json','where':'1=1',
        'outFields':'Start_Date___Time,End_Date___Time','orderByFields':'ObjectId2 DESC',
        'resultRecordCount':5,'returnGeometry':'false'}, timeout=30)
    recent.raise_for_status()
    (OUT/'boulder_source_check.json').write_text(json.dumps(dict(source=BOULDER,
        checked_unix=time.time(), editing=metadata.json().get('editingInfo'),
        recent_records=recent.json()), indent=2), encoding='utf-8')
    response = session.get(PALO_ALTO, stream=True, timeout=(30,60))
    response.raise_for_status()
    size = 0
    checksum = hashlib.sha256()
    temporary = destination.with_suffix('.csv.part')
    with temporary.open('xb') as handle:
        for chunk in response.iter_content(2**20):
            if chunk:
                handle.write(chunk)
                checksum.update(chunk)
                size += len(chunk)
    if size < 10000:
        raise RuntimeError('Unexpectedly small official data response')
    with temporary.open(encoding='utf-8-sig', errors='replace', newline='') as handle:
        reader = csv.reader(handle)
        fields = next(reader)
        rows = sum(1 for _ in reader)
    if not any('Energy' in field for field in fields) or rows < 10000:
        raise RuntimeError('Downloaded response is not the expected session table')
    temporary.rename(destination)
    endpoint = urlsplit(response.url)
    report = dict(source=PALO_ALTO,
        landing_page='https://data.paloalto.gov/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/',
        downloaded_unix=time.time(), final_host=endpoint.netloc,
        signed_query_not_saved=True, bytes=size, sha256=checksum.hexdigest(),
        records=rows, fields=fields,
        purpose='New geographic demand-input evaluation. Not a joint infrastructure field record.',
        licence_status='Confirm the official reuse terms before redistributing the original table.')
    (OUT/'palo_alto_source.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
