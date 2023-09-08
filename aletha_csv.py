# Read in the data from Aletha's csv (exported from the .opj files)

import itertools
import typing
import collections

_fn = "data\Depth and thickness - angle constant.csv"
_fn_bands = "data\All samples - INSA.csv"

class ExpData(typing.NamedTuple):
    x: list[float]
    y: list[float]
    y_sd: list[float] | None
    key: str


def _data_from_col(c):
    data = []
    for x in c[2:]:
        if x in ('', '--'):
            return data
        
        data.append(float(x))

    return data

class SuffixMaker:
    _c = None

    def __init__(self):
        self._c = collections.Counter()


    def get_suffix_for(self, k) -> str:
        self._c[k] += 1

        curr_val = self._c[k]
        return chr(curr_val + 64)


suffix_maker = SuffixMaker()

def _get_cols_from_file(fn):
    # Read a csv from Origin viewer. First two cols are non-numbers
    with open(_fn) as f:
        f_data = f.readlines()

    comma_split = [l.strip().split(',') for l in f_data]

    # Rotate so we go by columns
    cols =itertools.zip_longest(*comma_split)
        


    

    while True:
        
        c1, c2 = next(cols, None), next(cols, None)
        if not (c1 and c2):
            break
            
        yield c1, c2


def read_experimental_data() -> list[ExpData]:

    # Get pairs of columns
    out_data = []

    for c1, c2 in _get_cols_from_file(_fn):    
        # Titles - don't worry
        # Keys should be the same
        if c1[1] != c2[1]:
            raise ValueError(c1[1], c2[1])
        

        d1, d2 = _data_from_col(c1), _data_from_col(c2)

        key = c1[1]
        out_data.append(ExpData(x=d1, y=d2, key=key + " (Experimental) " + suffix_maker.get_suffix_for(key)))

    return out_data


def get_band_exp_data():
    
    all_cols = list(_get_cols_from_file(_fn_bands))
    data_cols = [_data_from_col(c) for c in all_cols]

    x_axis_thickness = data_cols[0]
    y1_num_bands = data_cols[1]
    y2_spacing = data_cols[4]
    y2_sd = data_cols[5]
    
    yield ExpData(x=x_axis_thickness, y=y1_num_bands, y_sd=None, key="Number of bands")
    yield ExpData(x=x_axis_thickness, y=y2_spacing, y_sd=y2_sd, key="Spacing of bands (um)")


if __name__ == "__main__":
    x = list(read_experimental_data())
    for xx in x:
        print(xx)
