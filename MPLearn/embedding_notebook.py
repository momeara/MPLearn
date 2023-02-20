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

from . import view_cells

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
        plate_id=None,
        meta_path=None,
        meta_columns=['Compound', 'dose_nM'],
        verbose=False):
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

    Returns: pd.DataFrame with for each cell in <experiment> with columns:
             <meta_columns> UMAP_1 ... cluster_label
    """

    if not os.path.exists(experiment_path):
        raise Exception(
            f"ERROR: Experiment path '{experiment_path}' does not exist.\n",
            f"ERROR: Current working directory is '{os.getcwd()}'")

    if meta_columns is not None:
        if meta_path is None:
            meta_path = f"{experiment_path}/raw_data/{plate_id}_Cell_MasterDataTable.parquet"
        if not os.path.exists(meta_path):
            raise Exception(
                f"ERROR: Unable to cell meta data at path '{meta_path}'")

        if meta_path.endswith(".parquet"):
            cell_meta = pa.parquet.read_table(
                source=meta_path,
                columns=meta_columns).to_pandas()
            if verbose:
                print(f"cell_meta: '{meta_path}'")
                print(f"cell_meta shape: ({cell_meta.shape[0]},{cell_meta.shape[1]})")
        elif meta_path.endswith(".tsv.gz"):
            cell_meta = pd.read_csv(meta_path, sep="\t")
            cell_meta = cell_meta[meta_columns]
            if verbose:
                print(f"cell_meta: {meta_path}")
                print(f"cell_meta shape: ({cell_meta.shape[0]},{cell_meta.shape[1]})")
        else:
            raise Exception(
                f"ERROR: Unrecognized extension of metadata file {meta_path}")

        # check that all the requested meta columns were found
        for meta_column in meta_columns:
            if meta_column not in cell_meta.columns:
                print(f"ERROR: Meta column '{meta_column}' not found in dataset '{meta_path}'")



    embed_dir = "{}/intermediate_data/{}".format(experiment_path, embedding_tag)
    if not os.path.exists(embed_dir):
        raise Exception(
            f"ERROR: Unable to locate embedding in '{embed_dir}'")


    # have we only embedded a sample of the dataset? if so make sure the meta-data is also sampled.
    if os.path.exists('{}/sample_indices.tsv'.format(embed_dir)):
        sample_indices = pd.read_csv(
            '{}/sample_indices.tsv'.format(embed_dir), header=None).loc[:, 0]
        if meta_columns is not None:
            cell_meta = cell_meta.iloc[sample_indices].reset_index()
        if verbose:
            print(f"filtering down sample_indices: '{sample_indices.shape[1]}'")

    embedding = pa.parquet.read_table(
        source="{}/umap_embedding.parquet".format(embed_dir)).to_pandas()

    if not os.path.exists(f"{embed_dir}/clusters.parquet"):
        if meta_columns is not None:
            embedding = pd.concat([cell_meta, embedding], axis=1)
    else:
        cluster_labels = pa.parquet.read_table(f"{embed_dir}/clusters.parquet").to_pandas()
        cluster_labels['cluster_label'] = cluster_labels['cluster_label'].astype(int)

        if verbose:
            print(f"cluster_labels: {embed_dir}/clusters.parquet")
            print(f"cluster_label shape: ({cluster_labels.shape[0]}, {cluster_labels.shape[1]})")

        if meta_columns is not None:
            embedding = pd.concat([cell_meta, embedding, cluster_labels], axis=1)
        else:
            embedding = pd.concat([embedding, cluster_labels], axis=1)
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

def view_UMAP_clusters(
        embedding,
        label="",
        cluster_labels=True,
        cluster_label_color='red'):
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

    if cluster_labels:
        # overlay the sampled points for the tool tips
        points = holoviews.Points(embedding, ['UMAP_1', 'UMAP_2'], label=label)

        hover_points = decimate(points)
        hover_points.opts(tools=['hover'], alpha=0)

        label_coords = []
        label_text = []
        for cluster_label in embedding['cluster_label'].unique():
            center_x = np.mean(embedding[embedding.cluster_label == cluster_label]['UMAP_1'])
            center_y = np.mean(embedding[embedding.cluster_label == cluster_label]['UMAP_2'])
            label_coords.append([center_x, center_y])
            label_text.append(f"Cluster {cluster_label}")
        label_coords = np.array(label_coords)
        labels_layer = holoviews.Labels(
            {('x','y'): label_coords, 'text': label_text},
            ['x', 'y'],
            'text').opts(
                holoviews.opts.Labels(
                    text_color=cluster_label_color))
        return (map * labels_layer *  hover_points)
    else:
        return map


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
        label='',
        verbose = False):
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

    condition_values = embedding[condition].unique().tolist()
    if verbose:
        print(f"Found {len(condition_values)} values for condition={condition}")

    if default_value is None:
        if verbose:
            print(f"setting default value to {condition_values[0]}")
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


def view_UMAP_instances(
        embedding,
        label="",
        max_n_instances=40,
        random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(seed=14730219)

    dataset = holoviews.Dataset(embedding, ['UMAP_1', 'UMAP_2'])
    UMAP_1_range = (embedding.UMAP_1.min(), embedding.UMAP_1.max())
    UMAP_2_range = (embedding.UMAP_2.min(), embedding.UMAP_2.max())

    roi_box = holoviews.Polygons([])
    roi_box = roi_box.opts(holoviews.opts.Polygons(fill_alpha=0.2, line_color='white'))
    roi_stream = holoviews.streams.BoxEdit(source=roi_box)

    def cell_table(data):
        table = None
        if not data or not any(len(d) for d in data.values()):
            selection = dataset.select(UMAP_1=UMAP_1_range, UMAP_2=UMAP_2_range)
            if len(selection) > max_n_instances:
                selection = selection.iloc[random_state.choice(len(selection), max_n_instances, False)]
            table = holoviews.Table(selection)
            return table
        data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
        for i, (x0, x1, y0, y1) in enumerate(data):
            selection = dataset.select(UMAP_1=(x0, x1), UMAP_2=(y0, y1))
            if len(selection) > max_n_instances:
                selection = selection.iloc[random_state.choice(len(selection), max_n_instances, False)]
            table = holoviews.Table(selection)
        return table
    dcell_table = holoviews.DynamicMap(cell_table, streams=[roi_stream])
    embedding_plot = view_UMAP(dataset, label=label)
    return embedding_plot * roi_box + dcell_table

def draw_regions_of_interest(
        line_width=3):
    path_layer = holoviews.Path([])
    regions_of_interest = holoviews.streams.FreehandDraw(source=path_layer)
    path_layer.opts(holoviews.opts.Path(
        active_tools=['freehand_draw'],
        height=300, width=300, line_width=line_width))
    return path_layer, regions_of_interest


def plot_ROI_paths(
        regions_of_interest,
        label_color='red'):
    """
    Create Holoview layers for showing regions of interest

    """
    n_roi = len(regions_of_interest.data['xs'])
    paths = {}
    for roi_index in range(n_roi):
        path_coords = np.array([
            regions_of_interest.data['xs'][roi_index],
            regions_of_interest.data['ys'][roi_index]]).T
        paths[f"ROI {roi_index}"] = holoviews.Curve(path_coords)
    paths_layer =  holoviews.NdOverlay(paths)

    label_coords = []
    label_text = []
    for roi_index in range(n_roi):
        center_x = np.mean(regions_of_interest.data['xs'][roi_index])
        center_y = np.mean(regions_of_interest.data['ys'][roi_index])
        label_coords.append([center_x, center_y])
        label_text.append(f"ROI {roi_index}")
    label_coords = np.array(label_coords)
    labels_layer = holoviews.Labels(
        {('x','y'): label_coords, 'text': label_text},
        ['x', 'y'],
        'text').opts(
            holoviews.opts.Labels(
                text_color=label_color))
    return paths_layer * labels_layer



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
    roi_paths = pd.DataFrame(roi_paths, columns=['roi_index', 'xs', 'ys'])

    pa.parquet.write_table(
        table=pa.Table.from_pandas(roi_paths),
        where=output_path)


def load_regions_of_interest(
        source="regions_of_interest.parquet"):
    regions_of_interest = pa.parquet.read_table(source=source).to_pandas()
    xs, ys = [], []
    for roi_index in regions_of_interest.roi_index.unique():
        xs.append(regions_of_interest[regions_of_interest.roi_index == roi_index]['xs'].to_list())
        ys.append(regions_of_interest[regions_of_interest.roi_index == roi_index]['yz'].to_list())
    return holoviews.streams.FreehandDraw(
        data={'xs' : xs, 'ys' : ys})



def embedding_cell_images(
        database_options,
        S3_region,
        S3_bucket,
        S3_key_template,
        cell_ids,
        dyes,
        saturations,
        color_maps,
        width,
        height,
        verbose=False):

    import mysql.connector
    
    db_connector = mysql.conector.connect(
        option_files=database_options)
    db_cursor = db_connector.cursor()

    cell_coordinates = view_cells.retrieve_cell_coordinates_from_db(
        db_cursor, cell_ids)

    cell_images = view_cells.retrieve_cell_images_from_S3(
        region = S3_region,
        bucket = S3_bucket,
        cell_coodinates = cell_coordinates)

    cell_images = crop_cells(
        cell_images,
        cell_coordinates)

    cell_images = style_images(
        cell_images,
        cell_ids)

    cell_images = montage_images(
        cell_images)
