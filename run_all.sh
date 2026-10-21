#!/bin/bash
# Reproduces every number, table and figure of the paper (about 10 minutes on one CPU core).
set -e
python3 final_A1.py      # single SU: main table, horizons, logarithmic growth
python3 final_A2.py      # emergency share, convergence windows, class errors
python3 final_B.py       # multi-SU: configurations, sweep over M, scalability, weights
python3 final_C.py       # sensitivity to channel persistence and bandwidth profile
python3 make_data_figures.py      # Figures 7-18 (note: files fig16/fig17 are numbered 17/16 in the paper)
python3 make_diagrams.py          # Figures 1-6
python3 make_graphical_abstract.py
python3 facts.py > facts.txt      # compact numbers sheet used when writing Sections 5-6
