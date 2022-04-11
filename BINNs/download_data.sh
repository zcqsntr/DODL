#!/usr/bin/env bash

root=/Volumes/groupfolders/DBIO_CSSB/projects-synbio/spatial-computing/IPTGreceivers/data # the root of the IPTG receivers data

dest=./data/

for dir in "$root"/* ; do
  for file in "$dir"/processed/* ; do


      if [[ $(basename "$file") =~ "processed" ]]; then

        cp "$file" "$dest"
        echo $(basename "$file")
      fi
  done
done