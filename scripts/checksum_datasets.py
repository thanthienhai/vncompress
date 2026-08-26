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


# Artifacts the documentation tells users to build locally
# (scripts/build_training_corpus.py, scripts/build_viquad_eval.py). They are
# not part of the committed benchmark, and whether they exist depends on what
# the user has run -- so their mere presence must not fail verification.
# They are still verified normally once recorded in the manifest, so
# deliberately committing one keeps working.
LOCALLY_GENERATED = {
    'training_corpus_v1.json',
    'vcc_bench_uit_viquad_qa.json',
}


def compute_manifest(include_generated: bool = True) -> dict:
    """Hash every dataset JSON on disk.

    include_generated=False drops the locally-generated artifacts, which is
    what --write should record: baking a machine-specific file into the
    committed manifest would make CI fail for everyone else, since the file
    is not in git.
    """
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.json') and f != 'CHECKSUMS.json'
        and (include_generated or f not in LOCALLY_GENERATED)
    )
    return {
        f: {'sha256': _sha256(os.path.join(DATA_DIR, f)), 'size_bytes': os.path.getsize(os.path.join(DATA_DIR, f))}
        for f in files
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='(Re)generate CHECKSUMS.json from files on disk')
    parser.add_argument('--include-generated', action='store_true',
                        help='With --write, also record the locally-generated artifacts '
                             f'({", ".join(sorted(LOCALLY_GENERATED))}). Only do this if you are '
                             'committing those files to git as well.')
    args = parser.parse_args()

    manifest = compute_manifest()

    if args.write:
        # Record only the committed benchmark artifacts unless the caller
        # explicitly opts a generated file in -- see LOCALLY_GENERATED.
        to_write = compute_manifest(include_generated=args.include_generated)
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            json.dump(to_write, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f"Wrote {MANIFEST_PATH} ({len(to_write)} files)")
        skipped = sorted(set(manifest) - set(to_write))
        if skipped:
            print(f"  Skipped locally-generated: {', '.join(skipped)}")
            print("  (pass --include-generated only if you are also committing them to git,")
            print("   otherwise CI fails: the manifest would reference files not in the repo)")
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
            if name in LOCALLY_GENERATED:
                # Expected: the docs tell users to build these. Not an error.
                print(f"[skip] {name}: locally generated, not part of the committed benchmark")
                continue
            print(f"[FAIL] {name}: present on disk but not in manifest. Run with --write to update it.")
            ok = False

    if ok:
        print(f"[OK] All {len(recorded)} dataset checksums match {MANIFEST_PATH}")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
