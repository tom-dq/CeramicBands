"""Keeps track of all the results in a database for quicker """
import itertools
import sqlite3
import typing


class ResultCase(typing.NamedTuple):
    num: int
    name: str
    major_inc: int
    minor_inc: int


class ContourKey(typing.NamedTuple):
    num: int
    name: str


class ContourValue(typing.NamedTuple):
    result_case_num: int
    contour_key_num: int
    elem_num: int
    value: float


class NodePosition(typing.NamedTuple):
    result_case_num: int
    deformed: bool
    x: float
    y: float
    z: float


def _make_table_statement(nt_class) -> str:
    type_lookup = {
        int: "INTEGER",
        float: "REAL",
        str: "TEXT",
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


_T_any_db_able = typing.Union[
    ResultCase,
    ContourKey,
    ContourValue,
    NodePosition,
]

_all_contour_keys_ = [
    ContourKey(num=1, name="PreStrain-X"),
    ContourKey(num=2, name="PreStrain-Y"),
    ContourKey(num=3, name="PreStrain-Mag"),
]


class DB:

    def __init__(self, db_fn: str):
        self.connection = sqlite3.connect(db_fn)
        self.cur = self.connection.cursor()

    def add(self, row: _T_any_db_able) -> int:
        """Add a single row, and return the rowid"""
        ins_str = _make_insert_string(row.__class__)
        with self.connection:
            self.cur.execute(ins_str, row)
            return self.cur.lastrowid

    def add_many(self, rows: typing.Iterable[_T_any_db_able]):
        def get_class(row):
            return row.__class__

        for klass, same_type_rows in itertools.groupby(rows, get_class):
            ins_str = _make_insert_string(klass)
            with self.connection:
                self.cur.executemany(ins_str, same_type_rows)

