#!/bin/bash

case "$1" in
	s|svd)
		python3 src/singular_value_decomposition.py
		;;
	d|dimred)
		python3 src/dimensionality_reduction.py
esac
