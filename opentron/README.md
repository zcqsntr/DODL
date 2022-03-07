

###Usage

```
usage: make_plates.py [-h] [--in_file IN_FILE] [--out_file OUT_FILE]

Produce plate configurations for the opentron from colony placements 

optional arguments:
  -h, --help           show this help message and exit
  --in_file IN_FILE    the input data from colony_placement, default is
                       colony_placement/output/placement.json
  --out_file OUT_FILE  the filepath to save output in, default is
                       opentron/output/plate_config.json

```


###Examples

```
python make_plates.py
```