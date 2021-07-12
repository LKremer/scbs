
# print messages go to stderr
# output file goes to stdout (when using "-" as output file)
# that way you can pipe the output file e.g. into bedtools
def echo(*args, **kwargs):
    click.echo(*args, **kwargs, err=True)
    return


def secho(*args, **kwargs):
    click.secho(*args, **kwargs, err=True)
    return

