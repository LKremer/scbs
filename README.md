# Command line utility for downstream analysis of single cell methylation data

## Installation
1. clone the repo
```
git clone https://github.com/LKremer/scbs.git
```
2. install the package with Python3 pip
```
python3 -m pip install --upgrade pip  # you need a recent pip version
python3 -m pip install scbs/dist/scbs-[choose-version].tar.gz
```
The command line interface should now be available when typing the command `scbs` in a terminal. If this is the case, the installation is finished. If not, try these steps:  
First, restart the terminal or use `source ~/.bashrc`. If that doesn't help, carefully check the output log of pip. Look for a message like `WARNING: The script scbs is installed in '/home/ubuntu/.local/bin' which is not on PATH.`, which would indicate that you need to add `/home/ubuntu/.local/bin` to your path. Alternatively, you can copy `/home/ubuntu/.local/bin/scbs` to e.g. `/usr/local/bin`.


## Updating to the latest version
Just use `--upgrade` when installing the package, otherwise it's the same process as installing:
```
python3 -m pip install --upgrade scbs/dist/scbs-[choose-version].tar.gz
```
Afterwards, make sure that the latest version is correctly installed:
```
scbs --version
```

## [Tutorial](docs/tutorial.md) of a typical `scbs` run
A tutorial can be found [here](docs/tutorial.md).

Also make sure to read the help by typing `scbs --help`.


## Troubleshooting
If you encounter a "too many open files" error during `scbs prepare` (`OSError: [Errno 24] Too many open files`), you need to increase the maximum number of files that can be opened. In Unix systems, try `ulimit -n 9999`.  
If you encounter problems during installation, make sure you have Python3.8 or higher. If the problem persists, consider installing scbs in a clean Python environment (for example using [venv](https://docs.python.org/3/library/venv.html)).


## Contributors
- [Lukas PM Kremer](https://github.com/LKremer)
- [Leonie Küchenhoff](https://github.com/LeonieKuechenhoff)
- [Alexey Uvarovskii](https://github.com/alexey0308)
- [Simon Anders](https://github.com/simon-anders)
