import traceback


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
