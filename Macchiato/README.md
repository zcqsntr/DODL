# The Macchiato algorithm 

Runs the Macchiato algorithm which finds the minimal set of bacterial colonies to implement the given digitial function

### Help
```
usage: macchiato.py [-h] [--outpath OUTPATH] T

Run the Macchiato algorithm

positional arguments:
  T                  the output of the truth table to be encoded

optional arguments:
  -h, --help         show this help message and exit
  --outpath OUTPATH  the filepath to save output in, default is
                     Macchiato/output
```

### Examples

```
./macchiato.py 00011010
./macchiato.py 00011010 --outpath /Users/neythen/Desktop/Projects/DODL/Macchiato/test
```