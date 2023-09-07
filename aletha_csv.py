# Read in the data from Aletha's csv (exported from the .opj files)

import itertools
import typing


_fn = "data\Depth and thickness - angle constant.csv"

class ExpData(typing.NamedTuple):
    x: list[float]
    y: list[float]
    key: str


def _data_from_col(c):
    data = []
    for x in c[2:]:
        if x in ('', '--'):
            return data
        
        data.append(float(x))

    return data

def read_experimental_data() -> list[ExpData]:
    with open(_fn) as f:
        f_data = f.readlines()

    comma_split = [l.strip().split(',') for l in f_data]

    # Rotate so we go by columns
    cols =itertools.zip_longest(*comma_split)
        
    # Get pairs of columns
    out_data = []

    while True:
        
        c1, c2 = next(cols, None), next(cols, None)
        if not (c1 and c2):
            break

        # Titles - don't worry
        # Keys should be the same
        if c1[1] != c2[1]:
            raise ValueError(c1[1], c2[1])
        

        d1, d2 = _data_from_col(c1), _data_from_col(c2)

        out_data.append(ExpData(x=d1, y=d2, key=c1[1]))

    return out_data

if __name__ == "__main__":
    x = list(read_experimental_data())
    for xx in x:
        print(xx)
