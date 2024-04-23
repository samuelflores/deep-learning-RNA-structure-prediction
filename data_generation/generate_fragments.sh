#!/bin/sh
# Shell script for generating fragments ranging from length 8-14

python generate_annotated_chains.py
python generate_fragments.py
python generate_fragments.py -f 10
python generate_fragments.py -f 12
python generate_fragments.py -f 14