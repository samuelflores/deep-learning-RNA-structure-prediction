#!/bin/sh
# Shell script for generating fragments ranging from length 8-14

# python generate_annotated_chains.py

# # Generate fragments with singular cluster IDs
# python generate_fragments.py
# python generate_fragments.py -f 10
# python generate_fragments.py -f 12
# python generate_fragments.py -f 14
# python generate_fragments.py -f 16
# python generate_fragments.py -f 18
# python generate_fragments.py -f 20
# python generate_fragments.py -f 22
# python generate_fragments.py -f 24

# Generate fragments with complex cluster IDs
python generate_fragments.py -p 'fragments_multi' -m
python generate_fragments.py -p 'fragments_multi' -m -f 24
python generate_fragments.py -p 'fragments_multi' -m -f 48
