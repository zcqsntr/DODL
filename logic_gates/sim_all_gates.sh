#!/bin/sh


cd configs;

for filename in ./*
do
  echo "$filename"
  python ../../DODL/colony_placement/simulate_plot.py --in_file "$filename" --outpath ../sims/"$filename"

done