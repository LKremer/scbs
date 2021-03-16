import click
from click import echo, secho, style
from datetime import datetime, timedelta
from scbs.scbs import make_profile, _get_filepath
from click_help_colors import HelpColorsGroup, HelpColorsCommand


class Timer(object):
    def __init__(self, label="run", fmt="%a %b %d %H:%M:%S %Y"):
        self.label = style(label, bold=True)
        self.fmt = fmt
        self.begin_time = datetime.now()
        echo(f"\nStarted {self.label} on {self.begin_time.strftime(self.fmt)}.")
        return

    def stop(self):
        end_time = datetime.now()
        runtime = timedelta(seconds=(end_time - self.begin_time).seconds)
        echo(f"\nFinished {self.label} on {end_time.strftime(self.fmt)}. "
             f"Total runtime was {runtime} (hour:min:s).")
        return


def _print_kwargs(kwargs):
    echo("\nCommand line arguments:")
    for arg, value in kwargs.items():
        if not value is None:
            value_fmt = style(str(_get_filepath(value)), fg="blue")
            echo(f"{arg: >15}: {value_fmt}")
    echo()


@click.group(
    cls=HelpColorsGroup,
    help_headers_color='bright_white',
    help_options_color='green',
    help=f"""
        Below you find a list of all available commands.
        To find out what they do and how to use them, check
        their help like this:

        {style("scbs profile --help", fg="blue")}
        
        To use stdin or stdout, use the character
        {style("-", fg="blue")} instead of a file path.
        """
)
def cli():
    pass


@cli.command()
@click.argument("cov_file_paths", nargs=-1, required=True, type=click.File())
@click.option("-o", "--output-dir", required=True, type=click.Path(), show_default=True)
@click.option("--format", default=(1, 2, 5, 6), nargs=4, show_default=True)
@click.option("--header", is_flag=True, show_default=True)
def matrix(**kwargs):
    echo(f"format is {format}.")
    echo(f"header is {header}.")
    echo(f"output dir is {output_dir}.")
    for f in cov_file_paths:
        echo(f"reading {f} ...")


@cli.command(
    help=f"""
    From single cell methylation or NOMe-seq data,
    calculates the average methylation profile of a set of
    genomic regions. Useful for plotting and visually comparing
    methylation between groups of regions or cells.

    {style("REGIONS", fg="green")} is an alphabetically sorted (!) .bed file of regions
    for which the methylation profile will be produced.

    {style("DATA_DIR", fg="green")} is the directory containing the methylation matrices
    produced by running 'scbs matrix'.

    {style("OUTPUT", fg="green")} is the file path where the methylation profile data
    will be written. Should end with '.csv'.
    """,
    short_help="plot the mean methylation around a group of genomic features"
)
@click.argument("regions", type=click.File("r"))
@click.argument(
    "data-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
)
@click.argument("output", type=click.File("w"))
@click.option(
    "--width",
    default=4000,
    show_default=True,
    type=click.IntRange(min=1, max=None),
    help="The total width of the profile plot in bp. "
    "The center of all bed regions will be "
    "extended in both directions by half of this amount. "
    "Shorter regions will be extended, longer regions "
    "will be shortened accordingly.",
)
@click.option(
    "--strand-column",
    type=click.IntRange(min=1, max=None),
    help="The bed column number (1-indexed) denoting "
    "the DNA strand of the region [optional].",
)
@click.option(
    "--label",
    help="Specify a constant value to be added as a "
    "column to the output table. This can be "
    "useful to give each output a unique label when "
    "you want to concatenate multiple outputs [optional].",
)
def profile(**kwargs):
    timer = Timer(label="profile")
    _print_kwargs(kwargs)
    make_profile(**kwargs)
    timer.stop()

cli.add_command(matrix)
cli.add_command(profile)
