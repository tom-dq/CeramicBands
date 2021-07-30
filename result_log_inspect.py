import datetime
import typing
import enum
import os


class St7LogState(enum.Enum):
    completed = enum.auto()
    terminated = enum.auto()


class LogReport(typing.NamedTuple):
    log_file_last_modified: datetime.datetime
    state: St7LogState


def log_inspect(log_fn: str) -> LogReport:
    s = os.stat(log_fn)

    with open(log_fn) as log:
        lines = [l.strip() for l in log.readlines() if l.strip()]

    end_line = lines[-2]

    if end_line.startswith("*SOLUTION TERMINATED AT"):
        state = St7LogState.terminated

    elif end_line.startswith("*SOLUTION COMPLETED"):
        state = St7LogState.completed

    else:
        raise ValueError(end_line)

    return LogReport(log_file_last_modified=datetime.datetime.fromtimestamp(s.st_mtime), state=state)


if __name__ == "__main__":
    term = log_inspect("QSLTerm.QSL")
    good = log_inspect("QSLGood.QSL")

    print(term)
    print(good)
    