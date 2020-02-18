
import pandas as pd
from sklearn import preprocessing
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

#######################
# random sample of 10k cells
cf10k = cell_features.sample(10000)
joblib.dump(
    value=cf10k,
    filename="intermediate_data/cf10k.joblib")

# random sample of 100k cells
cf100k = cell_features.sample(100000)
joblib.dump(
    value=cf100k,
    filename="intermediate_data/cf100k.joblib")


# random sample of 200k cells
cf200k = cell_features.sample(200000)
joblib.dump(
    value=cf200k,
    filename="intermediate_data/cf200k.joblib")
cf200k_scaler = preprocessing.StandardScaler().fit(cf200k)
cf200k_normed = cf200k_scaler.transform(cf200k)

joblib.dump(
    value=cf200k_scaler,
    filename="intermediate_data/cf200k_scaler.joblib")
joblib.dump(
    value=cf200k_normed,
    filename="intermediate_data/cf200k_normed.joblib")



######################
joblib.dump(
    value=cell_features,
    filename="intermediate_data/cell_features.joblib")

cell_features_scaler = preprocessing.StandardScaler().fit(cell_features)
cell_features_normed = cell_features_scaler.transform(cell_features)
joblib.dump(
    value=cell_features_scaler,
    filename="intermediate_data/cell_features_scaler.joblib")
joblib.dump(
    value=cell_features_normed,
    filename="intermediate_data/cell_features_normed.joblib")
