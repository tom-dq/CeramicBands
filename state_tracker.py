# Tracks the state of the process so I can change what it's doing in flight.

import enum
import typing
import os


class WriteSt7(enum.Enum):
    every_iter = enum.auto()
    equi_step = enum.auto()

    def nice_text(self) -> str:
        d = {
            WriteSt7.every_iter: "Write the .st7 file out at every iteration.",
            WriteSt7.equi_step: "Write the .st7 file out once the solution has equilibrated.",
        }
        return d[self]


class Execution(enum.Enum):
    running = enum.auto()
    pause = enum.auto()
    stop = enum.auto()

    def nice_text(self) -> str:
        d = {
            Execution.running: "Run as normal.",
            Execution.pause: "Write the .st7 file out and pause for the user to press enter.",
            Execution.stop: "Write the .st7 file out and exit.",
        }
        return d[self]


class State(typing.NamedTuple):
    write_st7: WriteSt7
    execution: Execution

    def need_to_write_st7(self) -> bool:
        if self.write_st7 == WriteSt7.every_iter:
            return True

        if self.execution in (Execution.stop, Execution.pause):
            return True

        return False

    def update_from_fn(self, look_dir: str) -> "State":
        """Return a new copy of the state with modifications based on the files found in the search directory."""

        working_state = self._replace()

        def have_file(name):
            full_fn = os.path.join(look_dir, name)
            file_exists = os.path.exists(full_fn)
            if file_exists:
                os.remove(full_fn)

            return file_exists

        def update_type(one_type):
            # Update one of the enums in the state.
            potential_new = [name for name in one_type.__members__.keys() if have_file(name)]

            if len(potential_new) == 1:
                return one_type[potential_new.pop()]
                #working_state = working_state._replace(write_st7=WriteSt7[potential_new])

            elif len(potential_new) > 1:
                print(f"Got multiple updates in {look_dir}... {potential_new}... not changing.")

        maybe_new_write_st7 = update_type(WriteSt7)
        if maybe_new_write_st7:
            working_state = working_state._replace(write_st7=maybe_new_write_st7)

        maybe_new_exec = update_type(Execution)
        if maybe_new_exec:
            working_state = working_state._replace(execution=maybe_new_exec)

        return working_state

    def print_signal_file_names(self, look_dir: str):
        for field_name, field_type in self._field_types.items():
            for one_val in field_type:
                signal_file = os.path.join(look_dir, one_val.name)
                print(signal_file, one_val.nice_text(), sep='\t')

    def unpause(self) -> "State":
        return self._replace(execution=Execution.running)


default_state = State(
    write_st7=WriteSt7.equi_step,
    execution=Execution.running,
)


