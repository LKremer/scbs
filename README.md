# Command line utility for downstream analysis of single cell methylation data

## Installation
1. clone the repo
```
git clone https://github.com/LKremer/scbs.git
```
2. install the package with Python3 pip
```
python3 -m pip install --upgrade pip  # you need a recent pip version
python3 -m pip install scbs/dist/scbs-0.1.2.tar.gz
```
The command line interface should now be available when typing the command `scbs` in a terminal. If this is the case, the installation is finished. If not, try these steps:  
First, restart the terminal or use `source ~/.bashrc`. If that doesn't help, carefully check the output log of pip. Look for a message like `WARNING: The script scbs is installed in '/home/ubuntu/.local/bin' which is not on PATH.`, which would indicate that you need to add `/home/ubuntu/.local/bin` to your path. Alternatively, you can copy `/home/ubuntu/.local/bin/scbs` to e.g. `/usr/local/bin`.

## Usage
Check the help in the command line interface like this:
```
scbs --help
```
or to get help for a subcommand, e.g. "prepare":
```
scbs prepare --help
```

## Troubleshooting
If you encounter a "too many open files" error during `scbs prepare` (`OSError: [Errno 24] Too many open files`), you need to increase the maximum number of files that can be opened. In Unix systems, try `ulimit -n 9999`.

## TODO (in order of importance):
- finish implementing "scbs scan" (overlapping variable windows should be merged automatically, the variance threshold should be estimated from the data itself e.g. by taking the 5% most variable parts of the genome instead of regions above a hard variance threshold)
- allow other input formats (currently only Bismark is supported)
- make things fast with numba (especially "scbs scan")
- test on simulated data and other datasets
- write basic tests & docs
- add further functionality (e.g. SVD, plotting, clustering, testing for differential methylation between clusters...)
