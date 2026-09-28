#!/usr/bin/env python3
"""Verify the SHA-256 and size of every bundled benchmark input."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    entries=json.loads((ROOT/'data/manifest.json').read_text())['files']
    for entry in entries:
        path=ROOT/entry['path']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise SystemExit(f"Checksum mismatch: {entry['path']}")
        print(f"OK {entry['path']}")
    print(f'{len(entries)} inputs verified.')
if __name__=='__main__':
    main()
