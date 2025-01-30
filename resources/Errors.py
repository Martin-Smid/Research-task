class BoundaryFormatError(Exception):
    """Raised when the inputted boundary format is not valid."""

    def __init__(self, message, tag: str):
        self.message = message
        self.tag = tag
        super().__init__(self.message)

    def __str__(self):
        return f"`format of <{self.tag}>`  {self.message}"

class InitWaveParamsError(Exception):
    """Raised when the inputted means and standard deviations are not valid
    probably not enough values in the list for given dimension."""

    def __init__(self, message, tag: str):
        self.message = message
        self.tag = tag
        super().__init__(self.message)

    def __str__(self):
        return f"`incorrect number of <{self.tag}> or their format` {self.message}"