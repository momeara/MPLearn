
import pandas as pd
import joblib

print("Loading cell features from 'input/cell_features.csv' ...")
cell_features = pd.read_csv('input/cell_features.csv', sep=',')

# split out the metadata for each cell
cell_meta = cell_features[[
    'Product Name',
    'Metadata_PlateID',
    'Metadata_WellID',
    'Number_Object_Number',
    'Concentration',
    'Condition',
    'Unique']]

joblib.dump(
    value=cell_meta,
    filename="intermediate_data/cell_meta.joblib")

cell_features = cell_features[
    [c for c in cell_features.columns if c not in cell_meta.columns]]
# [1480149 rows x 200 columns]

joblib.dump(
    value=cell_features,
    filename="intermediate_data/cell_features.joblib")


# random sample of 10k cells
cf10k = cell_features.sample(10000)
joblib.dump(
    value=cf10k,
    filename="intermediate_data/cf10k.joblib")

