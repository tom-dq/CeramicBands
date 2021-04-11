# Save DB stuff on github

import os
import pathlib
import shutil
import zipfile

ORIG_DIR = r"E:\Simulations\CeramicBands\v7\pics"
out_dir = r"H:\2021"

def main():
    conts = pathlib.Path(ORIG_DIR).iterdir()
    sub_dirs = (f for f in conts if f.is_dir() and f.name.startswith("8") or f.name.startswith("9"))
    for sd in sub_dirs:

        print(sd)

        new_dir = pathlib.Path(out_dir) / sd.name
        os.makedirs(new_dir, exist_ok=True)

        for fn in ["history.DB", "meta.txt"]:
            fn_source_full = sd / fn
            try:
                shutil.copy2(fn_source_full, new_dir)
                fn_dest = new_dir / fn

            except FileNotFoundError:
                pass

if __name__ == "__main__":
    main()