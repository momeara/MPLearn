
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
import pyarrow.parquet
import pyarrow as pa

# Birthday of Nicolaus Copernicus
random_state = np.random.RandomState(seed=14730219)


# copy pruned features from the 113 compound screen
pruned_features = pa.parquet.read_table(
    source="../umap_embedding_200217/intermediate_data/cell_features.parquet").to_pandas().columns
with open('input/pruned_features.csv', 'w') as f:
    f.write('feature_id\n')
    for feature in pruned_features:
        f.write("{}\n".format(feature))

print("Loading cell features from 'input/cell_features.csv' ...")
cell_features = pd.read_csv('input/cell_features.csv', sep=',')

# split out the metadata for each cell
cell_meta = cell_features[
    ['ImageNumber'] +
    ['ObjectNumber'] +
    [feature for feature in cell_features.columns if feature.startswith("Metadata")] +
    [feature for feature in cell_features.columns if feature.startswith("Number")] +
    [feature for feature in cell_features.columns if feature.startswith("Parent")] +
    [feature for feature in cell_features.columns if feature.startswith("Location")] +
    ['Plate_ID', 'Compound_ID', 'Concentration (uM)', 'Condition']]

# rename concentration

cell_meta.loc[ cell_meta.Condition == 'PC', 'Condition'] = 'Positive Control'
cell_meta.loc[ cell_meta.Condition == 'NC', 'Condition'] = 'Negative Control'
cell_meta.loc[cell_meta.Condition == 'Compound', 'Condition'] = cell_meta.loc[ cell_meta.Condition == 'Compound', 'Compound_ID']
cell_meta = cell_meta.rename(columns={'Concentration (uM)':'Concentration'})


pa.parquet.write_table(
    table=pa.Table.from_pandas(cell_meta),
    where='intermediate_data/cell_meta.parquet')

cell_features = cell_features[pruned_features]

# Check for NaNs
if np.any(np.isnan(cell_features)):
    print("Cell features have NaN values:")    
    nan_by_feature = np.sum(np.isnan(cell_features))
    nan_by_feature = nan_by_feature[nan_by_feature != 0]
    print("Features with NaN values:")
    print(nan_by_feature)
    print("")

    nan_by_cell = np.sum(np.isnan(cell_features), axis=1)
    nan_by_cell_meta = cell_meta[nan_by_cell != 0]
    print("Metadata for cells with NaNs:")
    print(nan_by_cell_meta)

    print("Impute NaNs for 'Number_Object_Number' features as 0's")
    for nan_column in nan_by_feature.index:
        if 'Number_Object_Number' in nan_column:
            print("   {}:".format(nan_column))
            cell_features.loc[np.isnan(cell_features[nan_column]), nan_column] = 0

    print("Multivariate Imputation for features derived from nonexistant objects")
    imp = IterativeImputer(max_iter=10, random_state=random_state)
    imp.fit(cell_features)
    cell_features = imp.transform(cell_features)
    cell_features = pd.DataFrame(cell_features)
    
# full set of cells
pa.parquet.write_table(
    table=pa.Table.from_pandas(cell_features),
    where="intermediate_data/cell_features.joblib")
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


