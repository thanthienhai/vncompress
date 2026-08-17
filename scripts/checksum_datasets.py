#!/usr/bin/env python3
"""Generate or verify vcc_bench_data/CHECKSUMS.json.

Resolves the "checksum/hash cho dataset artifacts quan trong" item of P1
issue #8 (dataset provenance): a static hash committed to the repo would
go stale silently the moment a dataset file changes, so this script both
writes the manifest (--write, e.g. after regenerating a dataset) and
verifies it (--check, the default -- suitable for CI) against the files
currently on disk.

Usage:
    python scripts/checksum_datasets.py          # verify (exit 1 on mismatch)
    python scripts/checksum_datasets.py --write   # (re)generate CHECKSUMS.json
"""
import argparse
import hashlib
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vcc_bench_data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'CHECKSUMS.json')


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest() -> dict:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.json') and f != 'CHECKSUMS.json')
    return {
        f: {'sha256': _sha256(os.path.join(DATA_DIR, f)), 'size_bytes': os.path.getsize(os.path.join(DATA_DIR, f))}
        for f in files
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='(Re)generate CHECKSUMS.json from files on disk')
    args = parser.parse_args()

    manifest = compute_manifest()

    if args.write:
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f"Wrote {MANIFEST_PATH} ({len(manifest)} files)")
        return 0

    if not os.path.exists(MANIFEST_PATH):
        print(f"[FAIL] {MANIFEST_PATH} does not exist. Run with --write to generate it.")
        return 1

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        recorded = json.load(f)

    ok = True
    for name, expected in recorded.items():
        actual = manifest.get(name)
        if actual is None:
            print(f"[FAIL] {name}: recorded in manifest but missing on disk")
            ok = False
        elif actual['sha256'] != expected['sha256']:
            print(f"[FAIL] {name}: checksum mismatch (dataset changed since manifest was written)")
            ok = False
    for name in manifest:
        if name not in recorded:
            print(f"[FAIL] {name}: present on disk but not in manifest. Run with --write to update it.")
            ok = False

    if ok:
        print(f"[OK] All {len(recorded)} dataset checksums match {MANIFEST_PATH}")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
