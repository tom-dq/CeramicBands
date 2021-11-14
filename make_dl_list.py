# Generate the input for, say, "C:\utils\wget.exe -x -c -i .\files.txt"

import directories
import pathlib
import collections
import itertools

START = "C3"
END = "E5"

ALL_FILES = r"C:\Users\Tom Wilson\Documents\CeramicBandData\outputs\all_files.txt"

def main():
    nums = range(int(START, base=36), int(END, base=36)+1)
    sub_dirs = [directories.base36encode(n, 2) for n in nums]

    for sub_dir in sub_dirs:
        yield f"http://192.168.1.109:8080/v7/pics/{sub_dir}/current_state.pickle"

    for sub_dir in sub_dirs:    
        yield f"http://192.168.1.109:8080/v7/pics/{sub_dir}/history.db"

    for sub_dir in sub_dirs:
        yield f"http://192.168.1.109:8080/v7/pics/{sub_dir}/{sub_dir}-x264.mp4"


def last_image_file_in_folder():

    nums = range(int(START, base=36), int(END, base=36)+1)
    sub_dirs = {directories.base36encode(n, 2) for n in nums}

    def get_files():
        with open(ALL_FILES) as f:
            paths = (pathlib.Path(p.strip()) for p in f)

            good_subdirs = (p for p in paths if p.parts[-2] in sub_dirs)
            images = (p for p in good_subdirs if p.name.endswith('.png') and p.name.startswith('Case-'))

            for p in images:
                case_num_a = p.name.split("Case-")[1]
                case_num_b = case_num_a.removesuffix(".png")
                case_num = int(case_num_b)

                sd = p.parts[-2]
                yield sd, case_num, p

    sd_to_casenum_to_path = collections.defaultdict(dict)
    for sd, case_num, p in get_files():
        sd_to_casenum_to_path[sd][case_num] = p

    for sub_dir, case_num_to_path in sd_to_casenum_to_path.items():
        max_case = max(case_num_to_path.keys())
        max_case_p = case_num_to_path[max_case]

        yield f"http://192.168.1.109:8080/v7/pics/{sub_dir}/{max_case_p.name}"


if __name__ == "__main__":
    for fn in itertools.chain(main(), last_image_file_in_folder()):
        print(fn)
    



