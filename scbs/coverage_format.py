class CoverageFormat:
    """Describes the columns in the coverage file."""

    def __init__(self, chrom, pos, meth, umeth, coverage, sep, header):
        self.chrom = chrom
        self.pos = pos
        self.meth = meth
        self.umeth = umeth
        self.cov = coverage
        self.sep = sep
        self.header = header

    def totuple(self):
        """Transform to use it in non-refactored code for now."""
        return (
            self.chrom,
            self.pos,
            self.meth,
            self.umeth,
            self.cov,
            self.sep,
            self.header,
        )


def create_custom_format(format_string):
    """Create from user specified string."""
    format_string = format_string.lower().split(":")
    if len(format_string) != 6:
        raise Exception("Invalid number of ':'-separated values in custom input format")
    chrom = int(format_string[0]) - 1
    pos = int(format_string[1]) - 1
    meth = int(format_string[2]) - 1
    info = format_string[3][-1]
    if info == "c":
        coverage = True
    elif info == "m":
        coverage = False
    else:
        raise Exception(
            "The 4th column of a custom input format must contain an integer and "
            "either 'c' for coverage or 'm' for methylation (e.g. '4c'), but you "
            f"provided '{format_string[3]}'."
        )
    umeth = int(format_string[3][0:-1]) - 1
    sep = str(format_string[4])
    if sep == "\\t":
        sep = "\t"
    header = bool(int(format_string[5]))
    return CoverageFormat(chrom, pos, meth, umeth, coverage, sep, header)


def create_standard_format(format_name):
    """Create a format object on the basis of the format name."""
    if format_name in ("bismarck", "bismark"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            False,
            "\t",
            False,
        )
    elif format_name in ("allc", "methylpy"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            True,
            "\t",
            True,
        )
    else:
        raise Exception(f"{format_name} is not a known format")
