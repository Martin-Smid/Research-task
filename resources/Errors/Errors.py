import traceback

class MissingArgumentError(Exception):
    """Raised when a required argument is missing."""

    def __init__(self, arg_name, func_name):
        tb = traceback.extract_stack()[-2]  # Get traceback info
        self.file = tb.filename
        self.line = tb.lineno
        self.message = (f"Missing argument error: '{arg_name}' is required in function '{func_name}'. "
                        f"(Raised in file {self.file}, line {self.line})")

        super().__init__(self.message)

    def __str__(self):
        return self.message


class BoundaryFormatError(Exception):
    """Raised when the inputted boundary format is not valid."""

    def __init__(self, message, tag: str):
        self.message = message
        self.tag = tag

        # Extract file name and line number from the traceback
        tb = traceback.extract_stack()[-2]
        self.file = tb.filename
        self.line = tb.lineno

        super().__init__(self.message)

    def __str__(self):
        return (f"`format of <{self.tag}>` {self.message} "
                f"(raised in file {self.file}, line {self.line})")


class InitWaveParamsError(Exception):
    """Raised when the inputted means and standard deviations are not valid
    probably not enough values in the list for given dimension."""

    def __init__(self, message, tag: str):
        self.message = message
        self.tag = tag

        # Extract file name and line number from the traceback
        tb = traceback.extract_stack()[-2]
        self.file = tb.filename
        self.line = tb.lineno

        super().__init__(self.message)

    def __str__(self):
        return (f"`incorrect number of <{self.tag}> or their format` {self.message} "
                f"(raised in file {self.file}, line {self.line})")


class IncorrectPacketTypeError(Exception):
    """Raised when the packet type provided is invalid."""

    def __init__(self, message, packet_type: str):
        self.message = message
        self.packet_type = packet_type

        # Extract file name and line number from the traceback
        tb = traceback.extract_stack()[-2]
        self.file = tb.filename
        self.line = tb.lineno

        super().__init__(self.message)

    def __str__(self):
        return (f"`packet type <{self.packet_type}> is invalid` {self.message} "
                f"(raised in file {self.file}, line {self.line})")

class TypeMismatchError(Exception):
    """Raised when an argument has an incorrect type."""

    def __init__(self, arg_name, expected_type, actual_type, func_name):
        tb = traceback.extract_stack()[-2]  # Get traceback info
        self.file = tb.filename
        self.line = tb.lineno
        self.message = (f"Type Error: Expected '{expected_type.__name__}' for argument '{arg_name}', "
                        f"but got '{actual_type.__name__}' instead.\n"
                        f"Exiting with status 0. Raised in function '{func_name}' "
                        f"from file '{self.file}'.")

        super().__init__(self.message)

    def __str__(self):
        return self.message

class IncorrectWaveBlueprintError(Exception):
    """Raised when the wave blueprint provided is invalid."""

    def __init__(self, message, provided=None):
        self.message = message
        self.provided = provided

        tb = traceback.extract_stack()[-2]
        self.file = tb.filename
        self.line = tb.lineno

        super().__init__(self.message)

    def __str__(self):
        return (f"Wave blueprint error: {self.message}. Provided: {self.provided} "
                f"(raised in {self.file}, line {self.line})")
