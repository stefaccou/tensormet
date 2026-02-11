#!/usr/bin/env python3
import argparse
import importlib
import sys
from pathlib import Path

import torch
import tensorly


def install_tensorly_aliases():
    # Alias the top-level module
    sys.modules["tensorly_custom"] = tensorly

    # If your pickle references submodules, alias those too (common ones)
    for name in [
        "decomposition",
        "tenalg",
        "backend",
        "base",
        "tucker_tensor",
        "cp_tensor",
    ]:
        try:
            sys.modules[f"tensorly_custom.{name}"] = importlib.import_module(f"tensorly.{name}")
        except Exception:
            pass  # ignore missing submodules


def rewrite_one_pt(path: Path) -> None:
    # Load with remapped modules, then save back to the same file
    obj = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(obj, path)


def iter_target_files(data_dir: Path):
    # EXACTLY: DATA_DIR / tensors / [dataset_name] / decomposition / [name_of_decomp].pt
    root = data_dir / "tensors"
    if not root.exists():
        return []

    # Only .pt directly inside each "decomposition" folder
    return sorted(root.glob("*/decomposition/*.pt"))


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite tensorly_custom pickled Torch .pt files under DATA_DIR/tensors/*/decomposition/*.pt"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="DATA_DIR (the directory that contains the 'tensors' folder)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be rewritten, but don't modify anything",
    )
    args = parser.parse_args()

    install_tensorly_aliases()

    files = iter_target_files(args.data_dir)
    if not files:
        print(f"No matching files found under: {args.data_dir / 'tensors' / '* / decomposition / *.pt'}")
        return

    if args.dry_run:
        print("Dry run. Would rewrite:")
        for p in files:
            print(f"  {p}")
        print(f"Total: {len(files)}")
        return

    ok = 0
    failed = 0
    for p in files:
        try:
            rewrite_one_pt(p)
            ok += 1
            print(f"[OK] {p}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {p}: {e}")

    print(f"Done. Rewritten: {ok}, Failed: {failed}, Total matched: {len(files)}")


if __name__ == "__main__":
    main()
