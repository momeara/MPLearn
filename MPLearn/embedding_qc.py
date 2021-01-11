
import numpy as np
import ot
import scipy.spatial.distance
import scipy.stats
import sklearn.neighbors



def distortion_statistics(
        native_coordinates,
        embedded_coordinates,
        metric = "euclidean",
        k = 30,
        verbose = False):
    """
    Compute distortion statistics between native and embedded cell coordinates

    Args:
        native_coordinates (np.array): Native coordinates for cells with shape
            [n_cell, n_native_coordinates]
        embedded_coordinates (np.array): Embedded coordinates for cells with shape
            [n_cell, n_embedded_coordinates]
        metric (string): distance metric to use for computing cell-cell distances
            See scipy.spatial.distance.cdist for available options [default: euclidean].
        k (int): number of neighbors for knn preservation statistics [default: 30]
        verbose (boolean): verbose logging [default: False]

    Returns:
        tuple with elements
             person_correlation (np.array): array with shape [n_cell] where 
                  the ith coordinate is the person correlation between the distances
                  between cell i and all other cells in the native coordinates
                  and the embedded coordinates.
             earth_movers_distances (float): optimal transport earth movers distance
                  between the native and embedded distance matrices.
             knn_perservation (float): percent of k nearest neighbors preserved
                  between the native and embedded distance matrices. 

    Reference:
        Cody N.Heiser, Ken S.Lau, Cell Reports Volume 31, Issue 5, 5 May 2020, 107576, DOI: 10.1016/j.celrep.2020.107576
        A Quantitative Framework for Evaluating Single-Cell Data Structure Preservation by Dimensionality Reduction Techniques
        https://github.com/KenLauLab/DR-structure-preservation
    
    """

    # Same number of cells in the native and embedded coordiantes
    assert( native_coordinates.shape[0] == embedded_coordinates.shape[0] )
    if verbose:
        print("Computing distortion statistics for {} cells over {} native and {} embedded coordinates".format(
            native_coordinates.shape[0],
            native_coordinates.shape[1],
            embedded_coordinates.shape[1]))
        
    if verbose:
        print("Computing native and embedded distances ...")
    native_distances = scipy.spatial.distance.cdist(
        native_coordinates, embedded_coordinates, metric = metric)
    embedded_distances = scipy.spatial.distance.cdist(
        embedded_coordinates, embedded_coordinates, metric = metric)

    if verbose:
        print("Computing pearson correlation ...")
    pearson_correlation = scipy.stats.pearsonr(
        x = native_distances, y = embedded_distances)

    if verbose:
        print("Normalizing distances ...")
    native_normed_distances -= native_distances.min()
    native_normed_distances /= native_distances.ptp()

    embedded_normed_distances -= native_distances.min()
    embedded_normed_distances /= native_distnaces.ptp()

    if verbose:
        print("Computing earth movers distances between normalized distances ...")
    earth_movers_distance = ot.wasserstein_1d(
        native_normed_distances, embedded_normed_distances)

    if verbose:
        print("Computing k nearest neighbor distances for k={} ...".format(k))
    native_knn_graph = sklearn.neighbors.kneighbors_graph(
        native_normed_distances,
        k,
        mode = "connectivity",
        include_self = False,
        n_jobs = -1).toarray()
    embbeded_knn_graph = sklearn.neighbors.kneighbors_graph(
        embbeded_normed_distances,
        k,
        mode = "connectivity",
        include_self = False,
        n_jobs = -1).toarray()

    if verbose:
        print("Computing k nearest neighbor preservation statistic ...")
    knn_preservation = np.isclose(
        native_knn_graph,
        embedded_knn_graph,
        rtol = 1e-05,
        atol = 1e-08).sum()
    knn_preservation = knn_preservation / (native_knn_graph.shape[0] ** 2)
    knn_preservation = np.round(knn_preservation * 100, 4)

    return pearson_correlation, earth_movers_distance, knn_preservation



