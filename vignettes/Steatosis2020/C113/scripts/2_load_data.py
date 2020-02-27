
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import pyarrow.parquet
import pyarrow as pa


# Birthday of Nicolaus Copernicus
random_state = np.random.RandomState(seed=14730219)


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

# cleanup Condition and Treatment columns
cell_meta.loc[ cell_meta.Condition == 'PC', 'Condition'] = 'Positive Control'
cell_meta.loc[ cell_meta.Condition == 'NC', 'Condition'] = 'Negative Control'
cell_meta.loc[cell_meta.Condition == 'Treatment', 'Condition'] = cell_meta.loc[ cell_meta.Condition == 'Treatment', 'Product Name']

pa.parquet.write_table(
    table=pa.Table.from_pandas(cell_meta),
    where='intermediate_data/cell_meta.parquet')

cell_features = cell_features[
    [c for c in cell_features.columns if c not in cell_meta.columns]]

# filter out LipidsFromTrans features as it's not in later the DR_5 set
cell_features = cell_features [
    [ c for c in cell_features.columns if not 'LipidsFromTrans' in c]]
# [1480149 rows x 196 columns]

# full set of cells
pa.parquet.write_table(
    table=pa.Table.from_pandas(cell_features),
    where="intermediate_data/cell_features.parquet")
cell_features_scaler = preprocessing.StandardScaler().fit(cell_features)
cell_features_normed = cell_features_scaler.transform(cell_features)
cell_features_normed = pd.DataFrame(cell_features_normed, columns=cell_features.columns)
joblib.dump(
    value=cell_features_scaler,
    filename="intermediate_data/cell_features_scaler.joblib")
pa.parquet.write_table(
    table=pa.Table.from_pandas(cell_features_normed),
    where="intermediate_data/cell_features_normed.parquet")


#######################
# random sample of 10k cells
cf10k = cell_features.sample(n=10000, random_state=random_state)
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf10k),
    where="intermediate_data/cf10k.parquet")
cf10k_scaler = preprocessing.StandardScaler().fit(cf10k)
cf10k_normed = cf10k_scaler.transform(cf10k)
cf10k_normed = pd.DataFrame(cf10k_normed, columns=cf10k.columns)
joblib.dump(
    value=cf10k_scaler,
    filename="intermediate_data/cf10k_scaler.joblib")
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf10k_normed),
    where="intermediate_data/cf10k_normed.parquet")

# random sample of 100k cells
cf100k = cell_features.sample(n=100000, random_state=random_state)
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf100k),
    where="intermediate_data/cf100k.parquet")
cf100k_scaler = preprocessing.StandardScaler().fit(cf100k)
cf100k_normed = cf100k_scaler.transform(cf100k)
cf100k_normed = pd.DataFrame(cf100k_normed, columns=cf100k.columns)
joblib.dump(
    value=cf100k_scaler,
    filename="intermediate_data/cf100k_scaler.joblib")
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf100k_normed),
    where="intermediate_data/cf100k_normed.parquet")

# random sample of 200k cells
cf200k = cell_features.sample(n=200000, random_state=random_state)
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf200k),
    where="intermediate_data/cf200k.parquet")
cf200k_scaler = preprocessing.StandardScaler().fit(cf200k)
cf200k_normed = cf200k_scaler.transform(cf200k)
cf200k_normed = pd.DataFrame(cf200k_normed, columns=cf200k.columns)
joblib.dump(
    value=cf200k_scaler,
    filename="intermediate_data/cf200k_scaler.joblib")
pa.parquet.write_table(
    table=pa.Table.from_pandas(cf200k_normed),
    where="intermediate_data/cf200k_normed.parquet")
