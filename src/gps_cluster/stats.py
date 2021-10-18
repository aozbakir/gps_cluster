import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from scipy import linalg
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from scipy.spatial.distance import cdist

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_velo_map(lon, lat, ve, vn):
    fig = plt.figure(figsize=(16,8))
    ax1=fig.add_subplot(121)
    ax1.scatter(ve, vn, edgecolor='blue', facecolor='white', lw=1)
    confidence_ellipse(ve, vn, ax=ax1, n_std=2, 
            facecolor='pink', edgecolor='purple', 
            alpha=.5, zorder=0)
    origin = np.array([[0],[0]])
    mean_x = np.mean(ve)
    mean_y = np.mean(vn)
    
    plt.quiver(*origin, mean_x, mean_y, 
         color=['k'], angles='xy', scale_units='xy', scale=1)

    ax1.set_title('{} horizontal velocities'.format(len(lon.index)))

    widths = [.05]* ve.size #arrow widths

    # -----------------------

    ax2=fig.add_subplot(122,projection=ccrs.Mercator() )
    #ax2=fig.add_subplot(122,projection=ccrs.AlbersEqualArea(central_latitude=25., central_longitude=-80.))
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 16, 'color': 'gray'}
    gl.ylabel_style = {'size': 16, 'color': 'gray'}

    ax2.quiver(lon, lat, ve,vn, 
            alpha=1, color='blue',
            linewidths=widths, transform=ccrs.PlateCarree())
    ax2.coastlines(color='gray')
    ax2.add_feature(cfeature.LAND, color='ivory')
    ax2.set_extent([-124.,-119.2, 36.3, 40.5])

def plot_clusters(X, loc, num_cluster):
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    plt.style.use(mystyle)
    link_matrix= linkage(X, 'centroid') 
    clusters=fcluster(link_matrix, t=num_cluster, criterion='maxclust')

    fig = plt.figure(figsize=(16,8))

    origin = np.array([[0],[0]])

    ax1=fig.add_subplot(121)
    ax1.scatter (X[:,0], X[:,1], c=clusters, s=80, lw=1.,
            cmap='Oranges' , edgecolor='k' , alpha=.6, zorder=1)

    for i in range(1,num_cluster+1):
        confidence_ellipse(
                X[clusters==i,0],
                X[clusters==i,1], 
                ax=ax1, n_std=2, facecolor='lightgray', alpha=.5 ,zorder=0)
#
        mean_x = np.mean(X[clusters==i,0])
        mean_y = np.mean(X[clusters==i,1])
        ax1.annotate(i, xy=(mean_x, mean_y), size=15, 
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9))
        ax1.quiver(*origin, mean_x, mean_y, 
            color=['k'], angles='xy', scale_units='xy', scale=1, zorder=2)
#
    ax2=fig.add_subplot(122,projection=ccrs.Mercator() )

    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=2, color='lightgray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 16, 'color': 'gray'}
    gl.ylabel_style = {'size': 16, 'color': 'gray'}

    ax2.scatter(loc[:,0], loc[:,1],
            c=clusters, s=80, lw=1, 
            cmap='Oranges',
            edgecolor='k', alpha=.7, 
            transform=ccrs.PlateCarree())
    ax2.coastlines(color='gray')
#    ax2.add_feature(cfeature.LAND, alpha=.5, color='lightgray')
#    ax2.set_extent([-124.,-119.2, 36.3, 40.5])


def generate_reference(xp):
    xref=np.random.uniform(xp[:,0].min(),xp[:,0].max(),(len(xp),1))
    yref=np.random.uniform(xp[:,1].min(),xp[:,1].max(),(len(xp),1))
    
    dist_ref=np.matmul(np.concatenate([xref, yref], axis=1), V.T)
    
    return dist_ref

def within_dispersion(max_cluster, link_matrix, data):

    dispersion=np.array([])
    for num_cluster in range(1, max_cluster+1):
        idx=fcluster(link_matrix, t=num_cluster, criterion='maxclust')
        wk=np.array(
            [np.sum(cdist(data[idx==k], data[idx==k].mean(axis=0)[np.newaxis,:])**2)/2
             for k in range(1,num_cluster+1)])
        dispersion=np.append(dispersion, wk.sum())

    return dispersion

def generate_reference(xp, V):
    
    xref=np.random.uniform(xp[:,0].min(),xp[:,0].max(),(len(xp),1))
    yref=np.random.uniform(xp[:,1].min(),xp[:,1].max(),(len(xp),1))
    
    dist_ref=np.matmul(np.concatenate([xref, yref], axis=1), V.T)
    
    return dist_ref

def plot_dendrogram(linkage):
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    fig, ax = plt.subplots(figsize=(25, 10))
    #plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    dendrogram(
        linkage,
        show_leaf_counts=False,
        show_contracted=True,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        #labels=labels,
        )
    ax.xaxis.grid(False)
    plt.show()

def fancy_dendrogram(*args, **kwargs):
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)\n indicates four clusters could be the optimal number', fontsize=14)
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points', fontsize=16,
                             va='top', ha='center')
        plt.axis('off')
        if max_d:
            plt.axhline(y=max_d, c='k', linestyle='--', alpha=.5)
    return ddata

mystyle = {"axes.spines.left" : True,
           "axes.spines.right" : False,
           "axes.spines.bottom" : True,
           "axes.spines.top" : False,
           "axes.grid" : False,
           "xtick.labelsize": 16,
           "ytick.labelsize": 16,
           "xtick.color": "dimgray",
           "ytick.color": "dimgray",
           "lines.linewidth" : 2,
           "legend.fontsize" : 16,
           "figure.titlesize" : 24}

