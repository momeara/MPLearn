
install_prereqs:
	conda install setuptools pandas joblib scikit-learn umap-learn
	conda install -c conda-forge umap-learn hbscan jupyterlab datashader holoviews
	pip install nbdev

install:
	python setup.py install

run_Steatosis2020_vignette:
	# generate umap embeddings and density based clusterings
	# produces
	#    intermediate_data/<dataset_tag> with processed features different data subsamples
	#    intermediate_data/<embedding_tag> directory with embeddings and clusterings
	#    product/figures/<embedding_tag>_embedding.png with images of the embeddings
	cd vignettes/Steatosis2020/umap_embedding_202017
	python scripts/1_init.py
	python scripts/2_load_data.py
	./scripts/3_embed_umap_2D.sh
	python scripts/4_cluster_hdbscan.py

start_local_dask_cluster:
	dask-scheduler --scheduler-file temp/scheduler.json &
	dask-worker --shceduler-file temp/scheduler.json &

start_Steatosis2020_vignette_notebooks:
	./vignettes/Steatosis2020/start_jupyter.sh
	# note the secret token
	# on local machine:
	#   ssh -i "sextonlab_linux.pem" -NfL 8887:localhost:8888 ubuntu@ec2-3-20-192-55.us-east-2.compute.amazonaws.com
	#   natigate browser to localhost:8887
	#   put in security token
