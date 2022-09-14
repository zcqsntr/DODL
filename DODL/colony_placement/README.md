
usage: design_logic_gates.py [-h] [--outpath OUTPATH] T

Run the colony placement algorithm

positional arguments:
  T                  the path of the saved output from macchiato

optional arguments:
  -h, --help         show this help message and exit
  --outpath OUTPATH  the filepath to save output in, default is
                     colony_placement/output


### Examples 
```
python design_logic_gates.py ../Macchiato/output/00011011.json
python simulate_plot.py --in_file ../colony_placement/output/placement.json --field 0

```