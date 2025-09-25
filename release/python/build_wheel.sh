#!/bin/bash

echo "Clearing build and dists"
python setup.py clean --all

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?

echo "Restoring the original project structure"