#!/bin/bash

mkdir -r input
mkdir -r intermediate_data
mkdir -r product
mkdir -r product/figures

# prepare paths to data
pushd input
ln -s ~/data/5Compound_22PointDoseResponse_Full.csv cell_features.csv
popd
