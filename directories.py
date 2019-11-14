import os
import glob
import pathlib


def base36encode(number, min_len: int):
    """https://stackoverflow.com/a/1181924/11308690"""
    if not isinstance(number, int):
        raise TypeError('number must be an integer')

    if number < 0:
        raise ValueError('number must be positive')

    alphabet, base36 = ['0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', '']

    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36

    num_as_str = base36 or alphabet[0]
    return num_as_str.rjust(min_len, alphabet[0])


def get_unique_sub_dir(base_dir) -> pathlib.Path:
    """Creates and returns a unique directory."""

    have_one_yet = False
    num_tries = 0

    while not have_one_yet and (num_tries < 10):
        num_tries += 1

        existing_subdirs = (one for one in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, one)))
        existing_nums = (int(one, base=36) for one in existing_subdirs)
        max_existing = max(existing_nums, default=0)
        new_dir_name = base36encode(max_existing+1, 2)

        new_full_path = os.path.join(base_dir, new_dir_name)

        try:
            os.makedirs(new_full_path, exist_ok=False)
            have_one_yet = True
            return pathlib.Path(new_full_path)

        except OSError:
            # Race condition - someone else tried to make the directory.
            print(f"{num_tries} clash on {new_full_path}")

    raise OSError(f"Could not make a new directory in {base_dir}")





