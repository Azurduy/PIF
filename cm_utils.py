"""
This module contains helper functions 
for cm.py.

author: Florian Heitmann
flowbiker@hotmail.de
"""


import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
from PIL import Image
from minepy import MINE
from xicorrelation import xicorr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import ecdf
from scipy.stats import gaussian_kde
from scipy.stats import rankdata, norm
from sklearn.feature_selection import mutual_info_regression
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kernel_density import EstimatorSettings
from statsmodels.distributions.copula.api import IndependenceCopula

# def get_pseudo_observations(path):
def get_pseudo_observations(df):
   
    headers = [*df]
   
    # calculate pseudo-observations, excluding last two columns (GT and predicted class)
    df_pseudo_obs = pd.DataFrame()
    for header in headers[:-2]:
        col = df[header].to_numpy()
        # statsmodels version
        df_pseudo_obs[header] = ECDF(col, side="right")(col)
       
    return df_pseudo_obs, headers

def scatterplot_3d(x, y, z, labels=None, title=None):
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    color_map = plt.get_cmap('spring')
    scatter_plot = ax.scatter3D(x, y, z,
                                c = z,
                                cmap = color_map)
    plt.title(title)
    if labels != None:
        ax.set_xlabel(labels[0], labelpad=20)
        ax.set_ylabel(labels[1], labelpad=20)
        ax.set_zlabel(labels[2], labelpad=20)
    plt.show()

def scatterplot_3d_multiple(data, labels=None, title=None):
    # max 3 datasets
    fig = plt.figure()
    ax = fig.add_subplot(projection ="3d")
    color_maps = ['autumn', 'winter', 'cool']
    for i, array in enumerate(data):
        x, y, z = array
        ax.scatter3D(x, y, z, c = z,
                     cmap = plt.get_cmap(color_maps[i]) )
    plt.title(title)
    if labels != None:
        ax.set_xlabel(labels[0], labelpad=20)
        ax.set_ylabel(labels[1], labelpad=20)
        ax.set_zlabel(labels[2], labelpad=20)
    plt.show()

# via numerical integration
def numerical_correlation(pseudo_obs1, pseudo_obs2, headers, grid_size=100, plot=False, type="spearman"):
    # based on distance of KDE copula to independence copula
    copula = KDEMultivariate(np.vstack((pseudo_obs1, pseudo_obs2)), var_type="cc")
    # settings = EstimatorSettings(randomize=True, n_res=25, n_sub=50,)
    # copula = KDEMultivariate(np.vstack((pseudo_obs1, pseudo_obs2)), var_type="cc", defaults=settings)
    # # try with scipy
    # copula_scipy = gaussian_kde(np.vstack((pseudo_obs1, pseudo_obs2)), bw_method=0.03)
    n = int(np.sqrt(grid_size)) # grid size = n**2
    x_lin = np.linspace(0+0.5*1/n, 1-0.5*1/n, n)
    y_lin = x_lin
    xs, ys = np.meshgrid(x_lin, y_lin)
    xs, ys = xs.ravel(), ys.ravel()
    # evaluation of KDE copula cdf at meshpoints
    meshpoints = np.vstack((xs, ys))

    # adjust KDE to [0,1]x[0,1] support
    x_zero = np.vstack((xs, np.zeros(len(xs))))
    zero_y = np.vstack((np.zeros(len(ys)), ys))
    copula_cdf_vals_mesh = (copula.cdf(meshpoints)
                           -copula.cdf(x_zero)
                           -copula.cdf(zero_y)
                           + copula.cdf(np.array([0.0,0.0])))
    
    copula_volume_on_support = (copula.cdf(np.array([1.0,1.0]))
                                -copula.cdf(np.array([1.0,0.0]))
                                -copula.cdf(np.array([0.0,1.0]))
                                +copula.cdf(np.array([0.0,0.0])))
    
    # print("copula_volume_on_support: ", copula_volume_on_support)
    copula_cdf_vals_mesh = copula_cdf_vals_mesh/copula_volume_on_support
    # # first version without adjusted KDE
    # copula_cdf_vals_mesh = copula.cdf(meshpoints)



    if plot:
        x, y, z = xs, ys, copula_cdf_vals_mesh
        labels = [headers[0], headers[1], "cdf value"]
        title = "KDE copula cdf scatterplot at meshpoints"
        scatterplot_3d(x, y, z, labels, title)
    pi_cop = IndependenceCopula()
    pi_copula_vals_mesh = pi_cop.cdf(np.dstack((xs, ys))).ravel()
    if plot:
        scatterplot_3d(x, y, pi_copula_vals_mesh, labels, "independence copula")
    if type == "abs_distance":
        # try with other correlation indicator (not necessarily normed)
        rho = 12*np.sum(np.abs(copula_cdf_vals_mesh-pi_copula_vals_mesh))/n**2
    else:
        # spearman
        rho = 12*np.sum(copula_cdf_vals_mesh-pi_copula_vals_mesh)/n**2
    if plot:
        scatterplot_3d_multiple( ((x, y, copula_cdf_vals_mesh.ravel()), (x, y, pi_copula_vals_mesh.ravel())), labels, "KDE and independence copula")
    return rho



def test_correlation_indicators(pseudo_obs1, pseudo_obs2, headers):
    cov = np.cov(np.hstack((pseudo_obs1, pseudo_obs2)))
    var1, var2= np.var(pseudo_obs1), np.var(pseudo_obs2)
    print("covariance: ", cov)
    print("var1, var2: ", var1, var2)
    spearmans_rho_pseudo = cov/(np.sqrt(var1)*np.sqrt(var2))
    print("cov/(np.sqrt(var1)*np.sqrt(var2): ", spearmans_rho_pseudo)
    spearmans_rho_scipy = spearmanr(pseudo_obs1, pseudo_obs2)
    print("spearmans rho scipy: ", spearmans_rho_scipy[0])
    print("kendalls tau scipy:", kendalltau(pseudo_obs1, pseudo_obs2)[0])
    spearmans_rho_num = numerical_correlation(pseudo_obs1, pseudo_obs2, headers, plot=False, type="spearman")
    print("spearmans_rho_numerical: ", spearmans_rho_num)
    abs_dist_num = numerical_correlation(pseudo_obs1, pseudo_obs2, headers, plot=False, type="abs_distance")
    print("absolute distance numerical: ", abs_dist_num)
    print("")

def plot_to_pdf(path):
    # store images as pdf
    # activate scatterplot of pseudo-observations in cm for-loop
    p = PdfPages(path)    
    # get_fignums Return list of existing  
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
    # iterating over the numbers in list 
    for fig in figs:  
        # and saving the files 
        fig.savefig(p, format='pdf')  
    # close the object 
    p.close()
    plt.close()
    # exit()