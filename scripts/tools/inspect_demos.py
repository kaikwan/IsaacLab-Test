#!/usr/bin/env python3
"""
inspect_demos.py  <file.hdf5>

Prints a tree‑like listing of the contents of an Isaac‑Lab dataset created by
ActionStateRecorderManager.  No external dependencies except h5py.
"""
import argparse, h5py, textwrap, sys

def print_tree(h5obj, indent=""):
    """Recursively print groups / datasets with shape + dtype."""
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}📁 {key}/")
            print_tree(item, indent + "    ")
        else:  # Dataset
            shape = "×".join(map(str, item.shape))
            print(f"{indent}📄 {key}   {shape}  {item.dtype}")

def main(path):
    try:
        with h5py.File(path, "r") as f:
            print(f"\nFile: {path}\n{'='* (len(path)+6)}")
            print_tree(f)
    except FileNotFoundError:
        print(f"ERROR: file '{path}' not found.")
        sys.exit(1)

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="List contents of Isaac‑Lab demo HDF5")
    cli.add_argument("file", help="Path to .hdf5 file")
    args = cli.parse_args()
    main(args.file)
