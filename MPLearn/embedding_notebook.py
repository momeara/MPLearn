# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import os
import numpy as np

import pyarrow.parquet
import pyarrow as pa
import pandas as pd

import holoviews
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread
from holoviews.plotting.util import process_cmap
from holoviews.operation import decimate
decimate.max_samples = 5000

import datashader
from datashader.colors import viridis
from datashader.colors import colormap_select
from datashader.transfer_functions import set_background
import colorcet as cc
big_color_key = list(set(cc.glasbey_cool + cc.glasbey_warm + cc.glasbey_dark))

from bokeh.plotting import figure, show, output_notebook
from IPython.core.display import display, HTML

import param
import panel as pn

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.prepared import prep




def initialize_notebook():
    """
    Initialize Notebook to have width 100% and hide logos and banners
    """

    output_notebook(hide_banner=True)
    holoviews.extension('bokeh', logo=False)
    display(HTML("<style>.container { width:100% !important; }</style>"))

def load_single_embedding(
        experiment_path,
        embedding_tag,
        cluster_embedding_tag=None,
        plate_id=None,
        meta_columns=['Compound', 'dose_nM']):
    """load cell embedding from an embed_umap run

    Example:
        Run from notebook in MPLearn/vignettes/SARS-CoV-2/notebooks/<notebook>.ipynb

        Load data from these locations:
             <experiment_path>/raw_data/<plate_id>_Cell_MasterDataTable.parquet
             S25/intermediate_data/<embedding_tag>/
                  umap_embedding.parquet
                  hdbscan_clustering_min100.parquet

        load_single_embedding(
            experiment_path="../S25",
            embedding_tag="hits_plate_scaled_200522a_umap2_2M_15_0.0",
            plate_id="covid19cq1_SARS_2007A_plate_scaled",
            meta_columns=meta_columns),

    If cluster_embedding_tag
        is None --> Use <embedding_tag>
        is False --> Don't load cluster labels
        is given --> Use cluster labels defined in the given directory

    Returns: pd.DataFrame with for each cell in <experiment> with columns:
             <meta_columns> UMAP_1 ... cluster_label
    """

    if not os.path.exists(experiment_path):
        raise Exception(
            f"ERROR: Experiment path '{experiment_path}' does not exist.\n",
            f"ERROR: Current working directory is '{os.getcwd()}'")

    cell_meta_path = f"{experiment_path}/raw_data/{plate_id}_Cell_MasterDataTable.parquet"
    if not os.path.exists(cell_meta_path):
        raise Exception(
            f"ERROR: Unable to cell meta data at path '{cell_meta_path}'")

    cell_meta = pa.parquet.read_table(
        source="{}/raw_data/{}_Cell_MasterDataTable.parquet".format(experiment_path, plate_id),
        columns=meta_columns).to_pandas()


    embed_dir = "{}/intermediate_data/{}".format(experiment_path, embedding_tag)
    if not os.path.exists(embed_dir):
        raise Exception(
            f"ERROR: Unable to locate embedding in '{embed_dir}'")


    # have we only embedded a sample of the dataset? if so make sure the meta-data is also sampled.
    if os.path.exists('{}/sample_indices.tsv'.format(embed_dir)):
        sample_indices = pd.read_csv(
            '{}/sample_indices.tsv'.format(embed_dir), header=None).loc[:, 0]
        cell_meta = cell_meta.iloc[sample_indices].reset_index()

    embedding = pa.parquet.read_table(
        source="{}/umap_embedding.parquet".format(embed_dir)).to_pandas()

    if cluster_embedding_tag is False:
        embedding = pd.concat([cell_meta, embedding], axis=1)
    else:
        if cluster_embedding_tag is not None:
            cluster_embed_dir = "{}/intermediate_data/{}".format(
                experiment_path, cluster_embedding_tag)
        else:
            cluster_embed_dir = embed_dir
        cluster_labels = pa.parquet.read_table(
            "{}/hdbscan_clustering_min100.parquet".format(cluster_embed_dir)).to_pandas()
        cluster_labels['cluster_label'] = cluster_labels['cluster_label'].astype(int)
        embedding = pd.concat([cell_meta, embedding, cluster_labels], axis=1)

    return embedding


def view_UMAP(
        embedding,
        label='',
        inv_color=False,
        color_pallet=viridis,
        background_color=None):
    """Return a HoloMap of a UMAP Embedding

    Input: embedding: Embedding with columns [<cell_meta_columns>, 'UMAP_1', 'UMAP_2']
           label: text label for the plot
    Output: A resizable HoloMap with tooltips with cell_meta columns for a sample of points
    """
    points = holoviews.Points(embedding, ['UMAP_1', 'UMAP_2'], label=label)
    map = rasterize(points)

    if inv_color:
        colormap = colormap_select(color_pallet[::-1])
    else:
        colormap = colormap_select(color_pallet)
    map = shade(map, cmap=colormap)
    map = dynspread(map, threshold=0.5)
    if (background_color is None) and not inv_color:
        map = map.options(bgcolor='black')
    hover_points = decimate(points)
    hover_points.opts(tools=['hover'], alpha=0)
    return (map * hover_points)

def save_embedding_plot(
        embedding,
        output_fname,
        inv_color=False,
        color_pallet=viridis,
        background_color="",
        plot_width=400,
        plot_height=400):
    canvas = datashader.Canvas(
        plot_width=plot_width,
        plot_height=plot_height).points(embedding, 'UMAP_1', 'UMAP_2')
    canvas = datashader.transfer_functions.shade(canvas, cmap=color_pallet)
    if (background_color is None) and not inv_color:
        canvas = datashader.transfer_functions.set_background(canvas, background_color)
    canvas.to_pil().convert('RGB').save(output_fname)

def view_UMAP_clusters(embedding, label=""):
    """Return a HoloMap of a UMAP Embedding colored by cluster id

    Input: embedding: Embedding with columns [<cell_meta_columns>, 'UMAP_1', 'UMAP_2', 'cluster_label']
           label: text label for the plot
    Output: A resizable HoloMap with tooltips with cell_meta columns and cluster_label for a sample of points
    """
    # spread out the colors for each cluster in the big_color_key map
    n_clusters = len(embedding['cluster_label'].unique())
    color_key_map = np.floor(np.linspace(0, 256, n_clusters+1)).astype(int)
    color_key = [big_color_key[i] for i in color_key_map]

    # assign the UMAP coordinates of each cells in each cluster to a separate Points object
    clusters = {}
    for cluster_id in embedding['cluster_label'].unique():
        cluster = holoviews.Points(
            embedding[embedding.cluster_label == cluster_id],
            kdims=['UMAP_1', 'UMAP_2'],
            label="Cluster "+str(cluster_id))
        clusters[cluster_id] = cluster

    # plot the clusters
    map = holoviews.NdOverlay(clusters, kdims='UMAP Clusters', label=label)
    map = datashade(map, aggregator=datashader.count_cat('UMAP Clusters'), color_key=color_key)
    map = dynspread(map, threshold=.4)
    map = map.options(bgcolor='black')

    # overlay the sampled points for the tool tips
    points = holoviews.Points(embedding, ['UMAP_1', 'UMAP_2'], label=label)
    hover_points = decimate(points)
    hover_points.opts(tools=['hover'], alpha=0)

    return (map * hover_points)




def view_UMAP_ROIs(
        embedding,
        roi_membership,
        label=""):
    """Return a HoloMap of a UMAP Embedding colored by roi_id

    Input: embedding: Embedding with columns [<cell_meta_columns>, 'UMAP_1', 'UMAP_2']
           roi_membership with true/false columns [roi_1, roi_2, ...]
    Output: A resizable HoloMap with tooltips with cell_meta columns and cluster_label for a sample of points
    """

    # spread out the colors for each cluster in the big_color_key map
    n_clusters = len(roi_membership.columns)
    color_key_map = np.floor(np.linspace(0, 256, n_clusters+1)).astype(int)
    color_key = [big_color_key[i] for i in color_key_map]

    # assign the UMAP coordinates of each cells in each cluster to a separate Points object
    clusters = {}
    for cluster_i in roi_membership.columns:
        cluster = holoviews.Points(
            embedding[roi_membership[cluster_i] == True],
            kdims=['UMAP_1', 'UMAP_2'],
            label="")
        clusters[cluster_i] = cluster

    # plot the clusters
    map = holoviews.NdOverlay(clusters, kdims='ROI Clusters', label=label)
    map = datashade(
        map,
        aggregator=datashader.count_cat('ROI Clusters'),
        color_key=color_key)
    map = dynspread(map, threshold=.4)
    map = map.options(bgcolor='black')

    # overlay the sampled points for the tool tips
    points = holoviews.Points(embedding, ['UMAP_1', 'UMAP_2'], label=label)
    hover_points = decimate(points)
    hover_points.opts(tools=['hover'], alpha=0)

    return map * hover_points



def view_UMAP_select_condition(
        embedding,
        condition,
        default_value=None,
        label=''):
    """
    In a notebook, plot an embedding and dynamically be able to select a subset of data to show

    Example:
        embedding = load_single_embedding(..., meta_columns = ['Compound'])
        view_UMAP_select_condition(
            embedding=embedding,
            condition="Compound")

    Arguments:
         embedding:
             pandas DataFrame with columns ["UMAP_1", "UMAP_2", <parameter>]
             each row represents a cell with coordinates and condition <condition>
         condition:
             column of the embedding to condition on
         default_value:
             value of the condition column to be used as the default
             if not given, use the first value.
         label:
             add a label to the plot
    """

    if condition not in embedding.columns:
        print(
            f"ERROR: condition {condition} is not a column in the embedding dataframe:",
            f"{embedding.columns}")

    condition_values = embedding[condition].unique()

    if default_value is None:
        default_value = condition_values[0]

    class EmbedPlot(param.Parameterized):
        """Hold state for the plot"""
        condition = param.ObjectSelector(default=default_value, objects=condition_values)
        @param.depends('condition')
        def points(self):
            points = holoviews.Points(embedding, kdims=['UMAP_1', 'UMAP_2'])
            args = {}
            args[condition] = self.condition
            points = points.select(**args)
            return points

        def view(self, **kwargs):
            points = holoviews.DynamicMap(self.points)
            plot = rasterize(points)
            plot = shade(plot, cmap=viridis, min_alpha=10)
            plot = dynspread(plot, threshold=0.4)
            plot = plot.options(bgcolor='black')
            return plot

    embed_plot = EmbedPlot(name=label)
    embed_plot = pn.Row(embed_plot.param, embed_plot.view()).servable()
    return embed_plot

def draw_regions_of_interest(
        line_width=3):
    path_layer = holoviews.Path([])
    regions_of_interest = holoviews.streams.FreehandDraw(source=path_layer)
    path_layer.opts(holoviews.opts.Path(
        active_tools=['freehand_draw'],
        height=300, width=300, line_width=line_width))
    return path_layer, regions_of_interest

def get_ROI_membership(
        regions_of_interest,
        points,
        verbose=True):

    n_roi = len(regions_of_interest.data['xs'])
    if verbose:
        print(f"Getting membership for {n_roi} regions of interest")

    roi_membership = []
    for roi_index in range(n_roi):
        if verbose:
            print(f"   Getting membership for region {roi_index}...")
        roi = prep(Polygon(zip(
            regions_of_interest.data['xs'][roi_index],
            regions_of_interest.data['ys'][roi_index])))

        def is_member(cell_coordinates):
            return roi.contains(
                Point(
                    cell_coordinates[0],
                    cell_coordinates[1]))
        roi_membership.append(
            points.apply(func=is_member, axis=1).rename("roi_{}".format(roi_index)))
    return pd.concat(roi_membership, axis=1)

def save_regions_of_interest(
        regions_of_interest,
        output_path="regions_of_interest.parquet"):
    # save regions of interest
    n_roi = len(regions_of_interest.data['xs'])
    roi_paths = []
    for roi_index in range(n_roi):
        roi_paths.extend(zip(
            [roi_index]*len(regions_of_interest.data['xs'][roi_index]),
            regions_of_interest.data['xs'][roi_index],
            regions_of_interest.data['ys'][roi_index]))
    roi_paths = pd.DataFrame(roi_paths, columns=['roi_index', 'xs', 'yz'])

    pa.parquet.write_table(
        table=pa.Table.from_pandas(roi_paths),
        where=output_path)
