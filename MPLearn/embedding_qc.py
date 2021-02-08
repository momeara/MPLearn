
import numpy as np
import ot
import scipy.spatial.distance
import dask_distance
import scipy.stats
import sklearn.neighbors
import matplotlib.pyplot as plt
import seaborn as sns

def distortion_statistics(
        native_coordinates,
        embedded_coordinates,
        metric = "euclidean",
        k = 30,
        use_dask = False,
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
    
    if use_dask:
        if verbose:
            print("Computing native and embedded distances using dask ...")
        native_distances = dask_distance.cdist(
            native_coordinates, native_coordinates, metric = metric)
        embedded_distances = dask_distance.cdist(
            embedded_coordinates, embedded_coordinates, metric = metric)
    else:
        if verbose:
            print("Computing native and embedded distances ...")        
        native_distances = scipy.spatial.distance.cdist(
            native_coordinates, native_coordinates, metric = metric)
        embedded_distances = scipy.spatial.distance.cdist(
            embedded_coordinates, embedded_coordinates, metric = metric)
        
    native_distances = native_distances[np.triu_indices(n=native_distances.shape[0], k=1)]
    embedded_distances = embedded_distances[np.triu_indices(n=embedded_distances.shape[0], k=1)]

    
    # if verbose:
    #     print("Computing k nearest neighbor distances for k={} ...".format(k))
    #
    # native_knn_graph = sklearn.neighbors.kneighbors_graph(
    #     native_normed_distances,
    #     k,
    #     mode = "connectivity",
    #     include_self = False,
    #     n_jobs = -1).toarray()
    # embbeded_knn_graph = sklearn.neighbors.kneighbors_graph(
    #     embbeded_normed_distances,
    #     k,
    #     mode = "connectivity",
    #     include_self = False,
    #     n_jobs = -1).toarray()
    # 
    #     
    # if verbose:
    #     print("Computing k nearest neighbor preservation statistic ...")
    # 
    # knn_preservation = np.isclose(
    #     native_knn_graph,
    #     embedded_knn_graph,
    #     rtol = 1e-05,
    #     atol = 1e-08).sum()
    # knn_preservation = knn_preservation / (native_knn_graph.shape[0] ** 2)
    # knn_preservation = np.round(knn_preservation * 100, 4)
    
    
    if verbose:
        print("Computing pearson correlation ...")
    
    pearson_correlation = scipy.stats.pearsonr(
        x = native_distances, y = embedded_distances)
    
    if verbose:
        print("Normalizing distances so mean distance = 1...")

    native_distances /= native_distances.mean()
    #native_normed_distances = native_distances
    #native_normed_distances -= native_distances.min()
    #native_normed_distances /= native_distances.ptp()

    embedded_distances /= embedded_distances.mean()
    #embedded_normed_distances = embedded_distances
    #embedded_normed_distances -= embedded_distances.min()
    #embedded_normed_distances /= embedded_distances.ptp()
    
    if verbose:
        print("Computing earth movers distances between normalized distances ...")
    earth_movers_distance = ot.wasserstein_1d(
        native_distances, embedded_distances)
    
    
    return native_distances, embedded_distances, pearson_correlation, earth_movers_distance






# structure preservation plotting class #
class SP_plot:
    """
    class defining pretty plots of structural evaluation of dimension-reduced embeddings such as PCA, t-SNE, and UMAP
    """

    def __init__(
        self, pre_norm, post_norm, figsize=(4, 4), labels=["Native", "Latent"]
    ):
        """
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        figsize = size of resulting axes
        labels = name of pre- and post-transformation spaces for legend (plot_cell_distances, plot_distributions,
            plot_cumulative_distributions) or axis labels (plot_distance_correlation, joint_plot_distance_correlation)
            as list of two strings. False to exclude labels.
        """
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, figsize=self.figsize)
        self.palette = sns.cubehelix_palette()
        self.cmap = sns.cubehelix_palette(as_cmap=True)
        self.pre = pre_norm
        self.post = post_norm
        self.labels = labels

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine()
        plt.tight_layout()

    def plot_cell_distances(self, legend=True, save_to=None):
        """
        plot all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        plt.plot(self.pre, alpha=0.7, label=self.labels[0], color=self.palette[-1])
        plt.plot(self.post, alpha=0.7, label=self.labels[1], color=self.palette[2])
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distributions(self, legend=True, save_to=None):
        """
        plot probability distributions for all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        sns.distplot(
            self.pre, hist=False, kde=True, label=self.labels[0], color=self.palette[-1]
        )
        sns.distplot(
            self.post, hist=False, kde=True, label=self.labels[1], color=self.palette[2]
        )
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_cumulative_distributions(self, legend=True, save_to=None):
        """
        plot cumulative probability distributions for all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        num_bins = int(len(self.pre) / 100)
        pre_counts, pre_bin_edges = np.histogram(self.pre, bins=num_bins)
        pre_cdf = np.cumsum(pre_counts)
        post_counts, post_bin_edges = np.histogram(self.post, bins=num_bins)
        post_cdf = np.cumsum(post_counts)
        plt.plot(
            pre_bin_edges[1:],
            pre_cdf / pre_cdf[-1],
            label=self.labels[0],
            color=self.palette[-1],
        )
        plt.plot(
            post_bin_edges[1:],
            post_cdf / post_cdf[-1],
            label=self.labels[1],
            color=self.palette[2],
        )
        if legend:
            plt.legend(loc="lower right", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distance_correlation(self, save_to=None):
        """
        plot correlation of all unique cell-cell distances before and after some transformation.
            save_to = path to .png file to save output, or None
        """
        plt.hist2d(x=self.pre, y=self.post, bins=50, cmap=self.cmap)
        plt.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def joint_plot_distance_correlation(self, save_to=None):
        """
        plot correlation of all unique cell-cell distances before and after some transformation.
        includes marginal plots of each distribution.
            save_to = path to .png file to save output, or None
        """
        plt.close()  # close matplotlib figure from __init__() and start over with seaborn.JointGrid()
        self.fig = sns.JointGrid(
            x=self.pre, y=self.post, space=0, height=self.figsize[0]
        )
        self.fig.plot_joint(plt.hist2d, bins=50, cmap=self.cmap)
        sns.kdeplot(
            x=self.pre,
            color=self.palette[-1],
            shade=False,
            bw_method=0.01,
            ax=self.fig.ax_marg_x,
        )
        sns.kdeplot(
            y=self.post,
            color=self.palette[2],
            shade=False,
            bw_method=0.01,
            ax=self.fig.ax_marg_y,
        )
        self.fig.ax_joint.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        plt.tick_params(labelbottom=False, labelleft=False)

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

