#!/usr/bin/env python3
"""Recompute the published allocation and API usage totals from the ledgers."""
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parent
seconds = defaultdict(int)
for row in csv.DictReader((root / 'local_compute.csv').open()):
    expected = int(row['elapsed_seconds']) * int(row['allocated_devices'])
    assert expected == int(row['allocated_device_seconds'])
    if row['unit'] != 'none': seconds[row['unit']] += expected
rows = list(csv.DictReader((root / 'api_usage_records.csv').open()))
result = {
    'local': {unit: {'device_seconds': total, 'hours': total / 3600} for unit,total in seconds.items()},
    'api': {
        'records': len(rows), 'usage_records': sum(bool(r['cost_usd']) for r in rows),
        'cost_usd': str(sum((Decimal(r['cost_usd']) for r in rows if r['cost_usd']), Decimal(0))),
        'prompt_tokens': sum(int(r['prompt_tokens'] or 0) for r in rows),
        'completion_tokens': sum(int(r['completion_tokens'] or 0) for r in rows),
    }
}
print(json.dumps(result, indent=2))
