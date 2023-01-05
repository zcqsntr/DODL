
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

### Fitting 

In the model/fitting directory is the code to run the fitting. The growth_fitting script runs a simple curve fit to parameterise a Gompertz growth model for the bacterial colonies. The fitting_spatial model then uses these parameters to fit the full spatial model. There are options to run a particle swarm, Bayesian and pseudo evolutionary algorithms. I used the particle swarm to get a good rough guess of the parameters which was then fine tuned using the evolutionary algorithm. The Bayesian algorithm never seemed to work well so I didnt end up using it. 