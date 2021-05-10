# Command line utility for downstream analysis of single cell methylation data

## Installation
1. clone the repo
```
git clone https://github.com/LKremer/scbs.git
```
2. install the package with Python3 pip
```
python3 -m pip install --upgrade pip  # you need a recent pip version
python3 -m pip install scbs/dist/scbs-0.2.0.tar.gz
```
The command line interface should now be available when typing the command `scbs` in a terminal. If this is the case, the installation is finished. If not, try these steps:  
First, restart the terminal or use `source ~/.bashrc`. If that doesn't help, carefully check the output log of pip. Look for a message like `WARNING: The script scbs is installed in '/home/ubuntu/.local/bin' which is not on PATH.`, which would indicate that you need to add `/home/ubuntu/.local/bin` to your path. Alternatively, you can copy `/home/ubuntu/.local/bin/scbs` to e.g. `/usr/local/bin`.

## Updating to the latest version
Just use `--upgrade` when installing the package, otherwise it's the same process as installing:
```
python3 -m pip install --upgrade scbs/dist/scbs-0.2.0.tar.gz
```
Afterwards, make sure that the latest version is correctly installed:
```
scbs --version
```

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
- [x] finish implementing "scbs scan" (overlapping variable windows are now merged automatically, the variance threshold is now estimated from the data itself by taking the top x% (default 5%) most variable genomic windows)
- [x] allow other input formats (now we support bismark, allc=methylpy and custom user-specified formats)
- [x] make things fast with numba (at least scan and matrix are now fast, I/O and gzip is now a bottleneck)
- [ ] test on simulated data and other datasets
- [ ] write basic tests & docs
- [ ] add further functionality (e.g. SVD, plotting, clustering, testing for differential methylation between clusters, usage without CLI...)
