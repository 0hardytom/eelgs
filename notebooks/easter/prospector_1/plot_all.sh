#!/bin/bash

for f in out/peas/*.h5; do
    python prospector_plot.py --file "$f"
done