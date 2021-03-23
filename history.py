"""Keeps track of all the results in a database for quicker """
import hashlib
import itertools
import pathlib
import sqlite3
import typing
import enum
import statistics

import common_types



class SqliteEnum(enum.Enum):
    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return self.name


class StatData(SqliteEnum):
    minimum = enum.auto()
    mean = enum.auto()
    maximum = enum.auto()

    @property
    def reduce_func(self) -> typing.Callable[[typing.List[float]], float]:
        if self == StatData.minimum:
            return min

        elif self == StatData.maximum:
            return max

        elif self == StatData.mean:
            return statistics.fmean

        else:
            raise ValueError(self)



class ContourKey(SqliteEnum):
    prestrain_x = enum.auto()
    prestrain_y = enum.auto()
    prestrain_z = enum.auto()  # This comes up in the iteration bit
    prestrain_mag = enum.auto()
    total_strain_x = enum.auto()
    total_strain_y = enum.auto()
    total_strain_z = enum.auto()

    @staticmethod
    def from_idx_total_strain(idx: int):
        d = {
            0: ContourKey.total_strain_x,
            1: ContourKey.total_strain_y,
            2: ContourKey.total_strain_z,
        }

        return d[idx]

    @staticmethod
    def from_idx_pre_strain(idx: int):
        d = {
            0: ContourKey.prestrain_x,
            1: ContourKey.prestrain_y,
            2: ContourKey.prestrain_z,
        }

        return d[idx]




sqlite3.register_adapter(bool, int)
sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))


class ResultCase(typing.NamedTuple):
    num: typing.Optional[int]  # Hack! This means an auto increment primary key.
    name: str
    major_inc: int
    minor_inc: int


class ContourKeyLookup(typing.NamedTuple):
    num: int
    name: str


class ContourValue(typing.NamedTuple):
    result_case_num: int
    contour_key_num: int
    elem_num: int
    value: float

    @classmethod
    def _all_nones(cls) -> "ContourValue":
        nones = [None for _ in cls._fields]
        return ContourValue(*nones)


class NodePosition(typing.NamedTuple):
    result_case_num: int
    node_num: int
    x: float
    y: float
    z: float

    @classmethod
    def _all_nones(cls) -> "NodePosition":
        nones = [None for _ in cls._fields]
        return NodePosition(*nones)


class ElementNodeConnection(typing.NamedTuple):
    """This is really just for internal use - to record the connectivity of the mesh once for plotting purposes later."""
    elem_num: int
    node_local_idx: int
    node_num: int


class ElementPrestrain(typing.NamedTuple):
    result_case_num: int
    elem_num: int
    axis: int
    pre_strain: float


class ColumnResult(typing.NamedTuple):
    result_case_num: int
    x: float
    yielded: bool
    contour_key: ContourKey
    stat_data: StatData
    value: float


def _make_table_statement(nt_class) -> str:
    type_lookup = {
        int: "INTEGER",
        float: "REAL",
        str: "TEXT",
        bool: "BOOLEAN",
        typing.Optional[int]: "INTEGER PRIMARY KEY",
        ContourKey: "TEXT",
        StatData: "TEXT",
    }

    def make_one_line(field_type_item):
        key, this_type = field_type_item
        sql_type = type_lookup[this_type]
        return f"{key} {sql_type}"

    table_data = ',\n'.join(make_one_line(item) for item in nt_class._field_types.items())

    return f"CREATE TABLE IF NOT EXISTS {nt_class.__name__}(\n {table_data} )"


def _make_insert_string(nt_class) -> str:
    qms = ",".join("?" for _ in nt_class._fields)
    return f"INSERT INTO {nt_class.__name__} VALUES ({qms})"

def _make_select_all(nt_class) -> str:
    return f"SELECT * FROM {nt_class.__name__}"

def _make_select_all_with_result_case_num(nt_class) -> str:
    return f"SELECT * FROM {nt_class.__name__} WHERE result_case_num = ?"


_T_any_db_able = typing.Union[
    ResultCase,
    ContourKeyLookup,
    ContourValue,
    NodePosition,
    ElementNodeConnection,
    ElementPrestrain,
    ColumnResult,
]

_all_contour_keys_ = [
    ContourKeyLookup(num=x.value, name=x.name) for x in ContourKey.__members__.values()
]


class DB:
    def __init__(self, db_fn: typing.Union[str, pathlib.Path]):
        self.connection = sqlite3.connect(str(db_fn))
        self.cur = self.connection.cursor()
        self._make_tables()
        self._add_contour_key_lookups_if_needed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cur.close()
        self.connection.close()

    def _make_tables(self):
        with self.connection:
            for one_table in _T_any_db_able.__args__:
                create_statement = _make_table_statement(one_table)
                self.connection.execute(create_statement)

    def _add_contour_key_lookups_if_needed(self):
        existing_rows = list(self.get_all(ContourKeyLookup))

        if not existing_rows:
            # add the rows - new DB.
            self.add_many(_all_contour_keys_)

        else:
            # check they're the same
            if sorted(existing_rows) != sorted(_all_contour_keys_):
                print(existing_rows)
                print(_all_contour_keys_)
                raise ValueError("Mismatched ContourKeyLookup.")


    def add(self, row: _T_any_db_able) -> int:
        """Add a single row, and return the rowid"""
        ins_str = _make_insert_string(row.__class__)
        with self.connection:
            self.cur.execute(ins_str, row)
            return self.cur.lastrowid

    def add_many(self, rows: typing.Iterable[_T_any_db_able]):
        """Add lots of rows at once. For it to be as fast as possible, they should be grouped by type like [ContourValue, ContourValue, ..., NodePosition, NodePosition, ...]"""

        def get_class(row):
            return row.__class__

        for klass, same_type_rows in itertools.groupby(rows, get_class):
            ins_str = _make_insert_string(klass)
            with self.connection:
                self.cur.executemany(ins_str, same_type_rows)

    def get_all(self, row_type: _T_any_db_able) -> typing.Iterable[ResultCase]:
        get_all_str = _make_select_all(row_type)
        with self.connection:
            rows = self.cur.execute(get_all_str)
            for r in rows:
                yield row_type(*r)

    def _create_index(self, row_skeleton: _T_any_db_able):

        not_none_bits = {key: val for key, val in row_skeleton._asdict().items() if val is not None}

        keys = tuple([key for key in not_none_bits.keys()])
        keys_bytes = b'_'.join(key.encode() for key in keys)
        checksum_head = hashlib.md5(keys_bytes).hexdigest()[0:8]

        name = row_skeleton.__class__.__name__

        index_name = f"{name}_{len(keys)}_{checksum_head}"
        cols = ', '.join(keys)
        index_str = f"CREATE INDEX IF NOT EXISTS {index_name} ON {name}({cols})"
        with self.connection:
            self.cur.execute(index_str)


    def get_all_matching(self, row_skeleton: _T_any_db_able):
        """input a row like ContourValue(result_case_num=2, contour_key_num=1, elem_num=None, value=None"""

        not_none_bits = {key: val for key, val in row_skeleton._asdict().items() if val is not None}

        # Make the select string
        sel_terms = [f"{key} = ?" for key in not_none_bits]
        sel_all = " AND ".join(sel_terms)
        sel_str = f"SELECT * FROM {row_skeleton.__class__.__name__} WHERE {sel_all}"
        args = list(not_none_bits.values())

        self._create_index(row_skeleton)

        row_class = row_skeleton.__class__
        with self.connection:
            rows = self.cur.execute(sel_str, args)
            yield from (row_class(*row) for row in rows)

    def add_element_connections(self, elem_conn: typing.Dict[int, typing.Tuple[int]]):
        """Saves all the element connections"""
        def make_rows():
            for elem_num, nodes in elem_conn.items():
                for node_idx, node_num in enumerate(nodes):
                    yield ElementNodeConnection(elem_num=elem_num, node_local_idx=node_idx, node_num=node_num)

        self.add_many(make_rows())

    def get_element_connections(self) -> typing.Dict[int, typing.Tuple[int]]:
        def get_rows():
            sel_str = "SELECT * FROM ElementNodeConnection ORDER BY elem_num, node_local_idx"
            with self.connection:
                rows = self.cur.execute(sel_str)
                for r in rows:
                    yield ElementNodeConnection(*r)

        def group_key(elem_node_conn: ElementNodeConnection):
            return elem_node_conn.elem_num

        out_dict = {}
        for elem_num, node_iter in itertools.groupby(get_rows(), group_key):
            node_nums = [elem_node_conn.node_num for elem_node_conn in node_iter]
            out_dict[elem_num] = tuple(node_nums)

        return out_dict


def do_stuff():
    with DB(r"E:\Simulations\CeramicBands\v5\pics\4L\history.db") as db:
        for row in db.get_all(ResultCase):
            print(row)

        skeleton = ContourValue(result_case_num=21, contour_key_num=3, elem_num=None, value=None)
        for row in db.get_all_matching(skeleton):
            print(row)

        for row in db.get_element_connections().items():
            print(row)


if __name__ == "__main__":
    do_stuff()
