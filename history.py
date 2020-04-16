"""Keeps track of all the results in a database for quicker """
import itertools
import pathlib
import sqlite3
import typing
import enum


class ContourKey(enum.Enum):
    prestrain_x = enum.auto()
    prestrain_y = enum.auto()
    prestrain_mag = enum.auto()


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


class NodePosition(typing.NamedTuple):
    result_case_num: int
    node_num: int
    x: float
    y: float
    z: float


def _make_table_statement(nt_class) -> str:
    type_lookup = {
        int: "INTEGER",
        float: "REAL",
        str: "TEXT",
        typing.Optional[int]: "INTEGER PRIMARY KEY"
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

    def get_all_matching(self, ):
        # TODO - up to here - maybe make the input a row like ContourValue(result_case_num=2, contour_key_num=1, elem_num=None, value=None
        pass

def do_stuff():
    with DB(r"E:\Simulations\CeramicBands\v5\pics\3E\history.db") as db:
        for row in db.get_all(ResultCase):
            print(row)

if __name__ == "__main__":
    do_stuff()
