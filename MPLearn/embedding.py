# -*- tab-width:4;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import time
import os
import glob
import joblib
import pyarrow.parquet
import pyarrow as pa
import numpy as np
from sklearn.decomposition import (PCA, IncrementalPCA)
import umap
import pandas as pd
import datashader
from datashader.transfer_functions import set_background
from colorcet import fire


def fit_embedding(
	dataset,
	embed_dir,
	pca_n_components=None,
	umap_n_components=2,
	umap_init='random',
	umap_n_neighbors=100,
	umap_min_dist=0.0,
	umap_metric='euclidean',
	seed=None,
	verbose=True):
	"""
	train_set: a feature matrix e.g. of (N, F) dimensions, used to define the embedding
	After some experimentation with chemical features of 1024 of 50k-.5M compounds,
	a reasonable embedding first reduces by PCA to 20 dimensions and then UMAP to 2 dimensions.
	UMAP parameters of 100 neighbors and min_dist of 0.0 seem to work well too. init='random'
	can help UMAP from getting stuck.
	return:
		saves embedding data to
			../intermediate_data/embeddings/tag/embedding_info.tsv
			../intermediate_data/embeddings/tag/pca_reducer.joblib
			../intermediate_data/embeddings/tag/umap_reducer.joblib
	"""
	if not os.path.exists(embed_dir):
		os.mkdir(embed_dir)
	else:
		print("WARNING: embed_dir already exists: {}".format(embed_dir))

	random_state = np.random.RandomState(seed=seed)

	begin_time = time.time()
	if pca_n_components is not None:
		if verbose:
			print("Reducing the dimension by PCA to {} dimensions".format(pca_n_components))
		#pca_reducer = PCA(n_components = pca_n_components)
		pca_reducer = IncrementalPCA(
			n_components = pca_n_components,
			batch_size=1000,
			copy=False)
		pca_reducer.fit(dataset)
		pca_embedding = pca_reducer.transform(dataset)
	else:
		pca_embedding = dataset

	if verbose:
		print("Reducing the dimension by UMAP to {} dimensions".format(umap_n_components))
	umap_reducer = umap.UMAP(
		n_components=umap_n_components,
		metric=umap_metric,
		n_neighbors=umap_n_neighbors,
		min_dist=umap_min_dist,
		init=umap_init,
		random_state=random_state,
		verbose=True)
	umap_embedding = umap_reducer.fit_transform(pca_embedding)
	umap_embedding = pd.DataFrame(
		data=umap_embedding,
		columns=["UMAP_" + str(i+1) for i in range(umap_n_components)])
	end_time = time.time()

	if verbose:
		print("created embedding {0} runtime: {1:.2f}s".format(embed_dir, end_time-begin_time))
		print("saving embedding to {}".format(embed_dir))
	with open("{}/model_info.tsv".format(embed_dir), 'w') as f:
		f.write("key\tvalue\n")
		f.write("seed\t{}\n".format(seed))
		f.write("input_dim\t{}\n".format(dataset.shape))
		f.write("pca_n_component\t{}\n".format(pca_n_components))
		f.write("umap_n_component\t{}\n".format(umap_n_components))
		f.write("umap_metric\t{}\n".format(umap_metric))
		f.write("umap_n_neighbors\t{}\n".format(umap_n_neighbors))
		f.write("umap_min_dist\t{}\n".format(umap_min_dist))
		f.write("umap_init\t{}\n".format(umap_init))

	if pca_n_components is not None:
		joblib.dump(
			value=pca_reducer,
			filename="{}/pca_reducer.joblib".format(embed_dir))

	joblib.dump(
		value=umap_reducer,
		filename="{}/umap_reducer.joblib".format(embed_dir))
	pa.parquet.write_table(
		table=pa.Table.from_pandas(umap_embedding),
		where="{}/umap_embedding.parquet".format(embed_dir))
	return umap_embedding


def embed(
	embed_dir,
	data,
	verbose=True):
	"""
	Given a previously defined embedding, embed a new dataset into it
	"""
	begin_time = time.time()
	print("Loading PCA->UMAP reducer ...")
	try:
		pca_reducer = joblib.load(filename="{}/pca_reducer.joblib".format(embed_dir))
	except:
		if verbose:
			print("PCA Reducer not found at '{}/pca_reducer.joblib', assume no pre-dimensionality reduction by PCA".format(embed_dir))
		pca_reducer = None

	umap_reducer = joblib.load(filename="{}/umap_reducer.joblib".format(embed_dir))

	if pca_reducer:
		pca_embedding = pca_reducer.transform(data)
	else:
		pca_embdding = data

	umap_embedding = umap_reducer.transform(pca_embedding)
	umap_embedding = pd.DataFrame(
		data=umap_embedding,
		columns=["UMAP_" + str(i+1) for i in range(umap_embedding.shape[1])])
	end_time = time.time()
	if verbose:
		print("embedded data onto {0} runtime: {1:.2f}s".format(embed_dir, begin_time-end_time))
	return umap_embedding

def embed_dataset(
	embed_dir,
	dataset,
	output_dir,
	verbose=True):
	"""
	Given a previously defined embedding, embed a new dataset into it
	"""
	begin_time = time.time()

	print("Loading PCA->UMAP reducer ...")
	try:
		pca_reducer = joblib.load(filename="{}/pca_reducer.joblib".format(embed_dir))
	except:
		if verbose:
			print("PCA Reducer not found at '{}/pca_reducer.joblib', assume no pre-dimensionality reduction by PCA".format(embed_dir))
		pca_reducer = None

	umap_reducer = joblib.load(filename="{}/umap_reducer.joblib".format(embed_dir))


	for i, shard_fname in enumerate(dataset.metadata_df['X']):
		print("embedding shard {}".format(i))
		output_path = "{}/umap_embedding_shard-{}.parquet".format(output_dir, i)
		if os.path.exists(output_path):
			print("  Skipping because {} exists".format(output_path))
			continue
		shard = joblib.load(os.path.join(dataset.data_dir, shard_fname))
		if pca_reducer is not None:
			pca_embedding = pca_reducer.transform(shard)
		else:
			pca_embedding = shard

		umap_embedding = umap_reducer.transform(pca_embedding)
		umap_embedding = pd.DataFrame(
			data=umap_embedding,
			columns=["UMAP_" + str(i+1) for i in range(umap_embedding.shape[1])])
		pa.parquet.write_table(
			table=pa.Table.from_pandas(umap_embedding),
			where=output_path)

	end_time = time.time()
	if verbose:
		print("embedded data onto {0} runtime: {1:.2f}s".format(embed_dir, begin_time-end_time))


def gather_embedding_shards(
	embedding_dir):
	shard_fnames = glob.glob(os.path.join(embedding_dir, "umap_embedding_shard-*.parquet"))

	# load the first shard to get the typical shape:
	shard_0 = pa.parquet.read_table(shard_fnames[0]).to_pandas()
	shape_0 = shard_0.shape[0]
	shape_1 = shard_0.shape[1]

	# load the last shard to get the number of rows of the last shard:
	shard_last = pa.parquet.read_table(shard_fnames[-1]).to_pandas()
	shape_last_0 = shard_last.shape[0]

	# compute the overall shape of the data
	n_shards = len(shard_fnames)
	shape = ((n_shards - 1) * shape_0 + shape_last_0, shape_1)

	print("loading embedding of shape {} from {}".format(shape, embedding_dir))
	embedding = np.zeros(shape)
	for i, shard_fname in enumerate(shard_fnames):
		if i % 50 == 0:
			print("Loading shard {}".format(shard_fname))
		shard = pa.parquet.read_table(shard_fname).to_pandas()
		embedding[shape_0*i:shape_0*i + shard.shape[0], :] = shard
	return embedding

def plot_embedding(
	embedding,
	output_fname,
	plot_width=400,
	plot_height=400,
	cmap=fire,
	shade_how='eq_hist',
	background=""):
	embedding = pd.DataFrame(data=embedding, columns = ["x", "y"])
	canvas = datashader.Canvas(
		plot_width=plot_width,
		plot_height=plot_height).points(embedding, 'x', 'y')
	canvas = datashader.transfer_functions.shade(canvas, how=shade_how, cmap=fire)
	if background:
		canvas = set_background(canvas, background)
	canvas.to_pil().convert('RGB').save(output_fname)


def plot_embedding_labels(
	embedding,
	output_fname,
	plot_width=400,
	plot_height=400,
	cmap=fire,
	shade_how='eq_hist',
	background=""):
	assert(embedding.shape[1] == 2)
	canvas = datashader.Canvas(
		plot_width=plot_width,
		plot_height=plot_height).points(embedding, 'UMAP_1', 'UMAP_2')
	canvas = datashader.transfer_functions.shade(canvas, how=shade_how, cmap=fire)
	if background:
		canvas = set_background(canvas, background)
	canvas.to_pil().convert('RGB').save(output_fname)
