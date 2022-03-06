import numpy as np
import jax.numpy as jnp
from GRF_perturbations.Modules.Utils import jax_map
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

def compute_SNR_grid(Spectra_grid,Noise_spectral_density):
    """
    The SNR estimated for every spectrum in the 'Spectra_grid'
    SNR=10*log10(Signal_Power/Noise_power)
    Parameters
    ----------
    Spectra_grid: np.ndarray real (N_logA,N_Beta,N_phi,N_k)
        The grid of Anomalies power spectra obtained from Inference_class.Anomalies_Radial_Power_Spectrum
        N_logA,... stand for number of logA,Beta,phi in the grid correspondingly and k is number of
        spatial frequencies defined by Surface_brightness_class.frequencies
    Noise_spectral_density: np.ndarray real (N_k)
        Spectral energy density of noise. For uncorrelated observation noise it is just noise variance for every k
    Returns
    -------
    SNR_grid: np.ndarray real (N_logA,N_Beta)
        Grid of SNR values for every logA,Beta in "Spectra_grid" averaged over random seeds phi
    """
    SNR=10*np.log10(np.mean(Spectra_grid-Noise_spectral_density,axis=-1)/Noise_spectral_density)
    mean_SNR=np.nanmean(SNR,axis=-1)
    return mean_SNR

def Infer_GRF(Data_spectrum,Spectra_grid):
    """
    Infer all the inference-related characteristics of logA,Beta
    i.e. grid of likelihood(logA,Beta), confidence_regions(logA,Beta),
    predicted logA and Beta, confidence_regions
    Parameters
    ----------
    Data_spectrum: np.ndarray real (N_k)
        Power spectrum of observed Einstein ring anomalies obtained from Inference_class.compute_radial_spectrum
    Spectra_grid: np.ndarray real (N_logA,N_Beta,N_phi,N_k)
        The grid of Anomalies power spectra obtained from Inference_class.Anomalies_Radial_Power_Spectrum
        N_logA,... stand for number of logA,Beta,phi in the grid correspondingly and k is number of
        spatial frequencies defined by Surface_brigthness_class.frequencies

    Returns
    -------
        Likelihood_grid: np.ndarray real (N_logA,N_Beta)
            likelihood for hypothesis that 'Data_spectrum' was induced by GRF with logA,Beta
            for every logA,Beta in the 'Spectrum grid'
        Confidence_grid: np.ndarray real (N_logA,N_Beta)
            Integrated likelihood representing confidence regions on logA and Beta
        prediction: [logA_index,Beta_index]
            0.5th quantiles of likelihood(logA) and likelihood(Beta) correspondingly
        logA_conf_intervals: np.ndarray real (3,2)
            likelihood bottom and top quantiles of likelihood(logA) outlining intervals of confidence 1sigma,2sigma,3sigma
        Beta_conf_intervals: np.ndarray real (3,2)
            likelihood bottom and top quantiles of likelihood(Beta) outlining intervals of confidence 1sigma,2sigma,3sigma
    """
    # Statistics of the spectra
    # log(Spectrum) Mean over random seed phi
    MU_tensor = np.log(Spectra_grid).mean(axis=2)
    MU_tensor_ext = np.tile(MU_tensor, Spectra_grid.shape[2])
    MU_tensor_ext = MU_tensor_ext.reshape(Spectra_grid.shape)
    # log(Spectrum) standard deviation over random seed phi
    Sigma_tensor = np.sqrt(np.power(np.log(Spectra_grid) - MU_tensor_ext, 2).sum(axis=2) / (Spectra_grid.shape[2] - 1))


    Data_logspectrum=np.log(Data_spectrum) # Anomalies spectrum that we want to infer logA,Beta for
    Chi_squared=np.mean(np.power((Data_logspectrum-MU_tensor)/Sigma_tensor,2),axis=-1) # negative log-likelihood
    Likelihood_grid=np.exp(-Chi_squared/2)
    Likelihood_grid/=Likelihood_grid.sum() # Normalised grid of likelihoods for every logA,Beta

    prediction, logA_conf_intervals, Beta_conf_intervals=get_confidence_intervals(Likelihood_grid)
    logA_pred_index, Beta_pred_index=prediction
    Confidence_grid=compute_Confidence_grid(Likelihood_grid)

    return Likelihood_grid,Confidence_grid,logA_pred_index,Beta_pred_index,logA_conf_intervals,Beta_conf_intervals


def get_cdf(likelihood):
    """ Get cumulative distribution function from probability density function """
    cdf=jnp.cumsum(likelihood)
    normalised_cdf=cdf/cdf[-1]

    return normalised_cdf

def get_confidence_intervals(Likelihood_grid):
    """
    The function estimates median and 1,2,3sigma confidence intervals for likelihoods of logA and Beta
    Parameters
    ----------
    Likelihood_grid: np.ndarray real (N_logA,N_Beta)
            likelihood for hypothesis that 'Data_spectrum' was induced by GRF with logA,Beta
    Returns
    -------
    prediction: [logA_index,Beta_index]
        0.5th quantiles of likelihood(logA) and likelihood(Beta) correspondingly
    logA_conf_intervals: np.ndarray real (3,2)
        likelihood bottom and top quantiles of likelihood(logA) outlining intervals of confidence 1sigma,2sigma,3sigma
    Beta_conf_intervals: np.ndarray real (3,2)
        likelihood bottom and top quantiles of likelihood(Beta) outlining intervals of confidence 1sigma,2sigma,3sigma
    """

    Beta_cdf=get_cdf(Likelihood_grid.mean(axis=0))
    Beta_pred_index=jnp.argmin(jnp.abs(Beta_cdf-0.5))


    logA_cdf=get_cdf(Likelihood_grid.mean(axis=1))
    logA_pred_index=jnp.argmin(jnp.abs(logA_cdf-0.5))

    percentage_covered=jnp.array([68,95,99.7])

    Beta_conf_intervals=jax_map(lambda percentage: get_confidence_interval_bounds(Beta_cdf,Beta_pred_index,percentage),percentage_covered)
    logA_conf_intervals=jax_map(lambda percentage: get_confidence_interval_bounds(logA_cdf,logA_pred_index,percentage),percentage_covered)

    return [logA_pred_index,Beta_pred_index],logA_conf_intervals,Beta_conf_intervals

def get_confidence_interval_bounds(cumulative_distribution_function,median_index,percentage_covered):
    """get lower and upper confidence bound covering 'percentage_covered'
    of 'cumulative distribution function' """
    #indent=np.minimum(0.5,percentage_covered/200.)
    indent=percentage_covered/200.

    median_cdf=cumulative_distribution_function-cumulative_distribution_function[median_index]
    upper_index=jnp.argmin(jnp.abs(median_cdf-indent))
    lower_index=jnp.argmin(jnp.abs(median_cdf+indent))

    return jnp.array([lower_index,upper_index])

def compute_Confidence_grid(Likelihood_grid):
    """ Compute confidence regions and assign each point of the grid with a value,
     corresponding to the confidence of the maximum confidence region that it belongs to """

    likelihood_isolevels=Likelihood_grid.max()*np.linspace(1,0,101) # Spit on 100 confidence regions
    confidence_levels = np.zeros_like(likelihood_isolevels) # levels of confidence for each region
    Confidence_grid=np.zeros_like(Likelihood_grid)*np.nan # map of confidences

    #Compute cumulative probability enclosed in a contour with constant likelihood
    for i,isolevel in enumerate(likelihood_isolevels):
        above_isolevel_mask=(Likelihood_grid>=isolevel) # mask of everything withing the selected contour
        confidence_levels[i]=np.nansum(Likelihood_grid*above_isolevel_mask) # confidence in that contour

        undefined_confidence_mask=np.isnan(Confidence_grid) # region where values are not yet defined
        confidence_mask=above_isolevel_mask*undefined_confidence_mask # region in which the values should be assigned
        Confidence_grid[np.where(confidence_mask==True)]=confidence_levels[i] # assign the new confidence values

    Confidence_grid/=confidence_levels[-1] # normalise so total probability equals 1

    return Confidence_grid


def plot_Confidence_grid(axis,Beta_array,logA_array,Confidence_grid,SNR_grid,
                         logA_true_index,Beta_true_index,logA_max_likelihood_index,Beta_max_likelihood_index,
                         xticks,yticks,confidence_labels_locations,legend=False,fontsize=18):
    """
    Show confidence regions, maximum likelihood logA and Beta, true logA and Beta, negative SNR region
    Parameters
    ----------
    axis
    Beta_array
    logA_array
    Confidence_grid
    SNR_grid
    logA_true_index
    Beta_true_index
    logA_max_likelihood_index
    Beta_max_likelihood_index
    xticks
    yticks
    confidence_labels_locations
    legend
    fontsize

    Returns
    -------

    """
    #Confidence levels
    #smooth the grid to avoid sharp edged contours
    smooth_confidence_grid=ndimage.gaussian_filter(Confidence_grid,sigma=(2,2))
    # Outline the 1,2,3sigma contours (confidence for 2d gaussian)
    axis.contourf(Beta_array,logA_array,smooth_confidence_grid,[0,0.39347,0.86466,0.988891],colors=['red','indianred','rosybrown','w'],alpha=0.7)
    img=axis.contour(Beta_array,logA_array,smooth_confidence_grid,[0.39347,0.86466,0.988891],colors='k')
    # Label the confidence regions contours
    axis.clabel(img, [0.988891], inline=True, fmt={0.988891: '$3\\sigma$'}, fontsize=15, manual=confidence_labels_locations[0])
    axis.clabel(img, [0.86466], inline=True, fmt={0.86466: '$2\\sigma$'}, fontsize=15, manual=confidence_labels_locations[1])
    axis.clabel(img, [0.39347], inline=True, fmt={0.39347: '$1\\sigma$'}, fontsize=15, manual=confidence_labels_locations[2])

    # Mark max likelihood logA,Beta
    predicted_Point=axis.scatter(Beta_array[Beta_max_likelihood_index],logA_array[logA_max_likelihood_index],label='Max likelihood',marker="o",s=80,color='k',edgecolor='w',linewidth=0.5)
    # Mark ground truth logA,Beta
    true_Point=axis.scatter(Beta_array[logA_true_index],logA_array[Beta_true_index],label='Ground truth',marker="*",s=80,color='k',edgecolor='w',linewidth=0.5)

    #Outline region of negative SNR
    imgSNR=axis.contourf(Beta_array,logA_array,SNR_grid,[SNR_grid.min(),0],colors='grey',alpha=0.5)
    axis.contour(Beta_array,logA_array,SNR_grid,[0],colors='k',linestyles='--')


    axis.set_xticks(xticks)
    axis.set_yticks(yticks)
    axis.set_xlabel(r'$\beta$',fontsize=fontsize)
    axis.set_ylabel(r"${\rm log}(A)$",fontsize=fontsize)

    if legend:
        l=axis.legend([true_Point,predicted_Point,plt.Rectangle((1, 1), 2, 2, fc=imgSNR.collections[0].get_facecolor()[0])],
                      ['Ground truth','Max likelihood',r'$SNR \leq 0$'],loc='upper right',fontsize=15,framealpha=0)
        for text in l.get_texts():
            text.set_color("k")

def marginalize_value(value,precision,marginalization='round'):
    # precision give numbers after the coma in the float number to leave
    if marginalization=='round':
        return round(value*np.power(10,precision))/np.power(10,precision)
    elif marginalization=='floor':
        return math.floor(value * np.power(10, precision)) / np.power(10, precision)
    elif marginalization=='ceil':
        return math.ceil(value * np.power(10, precision)) / np.power(10, precision)
    else:
        raise ValueError('wrong marginalization type, pick one of "round","floor","ceil"')

# TODO some problems with
def plot_Inference_results(fig,ax_row,Surface_brightness,logA_array,Beta_array,Observed_image,Observed_anomalies,GRF_potential,Likelihood_grid,Confidence_grid,SNR_grid,
              logA_true_index,Beta_true_index,confidence_labels_locations=None,
              logA_ind_lim=None,Beta_ind_lim=None,legend_flag=True,titles_flag=True,fontsize=18):
    """

    Parameters
    ----------
    fig
    ax_row
    Surface_brightness
    logA_array
    Beta_array
    Observed_image
    Observed_anomalies
    GRF_potential
    Likelihood_grid
    Confidence_grid
    SNR_grid
    logA_true_index
    Beta_true_index
    confidence_labels_locations
    logA_ind_lim
    Beta_ind_lim
    legend_flag
    titles_flag
    fontsize

    Returns
    -------

    """

    if confidence_labels_locations is None:
        confidence_labels_locations=[[(4,-10)],[(4,-10)],[(4,-10)]]

    if logA_ind_lim is None:
        logA_ind_lim=(0,Likelihood_grid.shape[0])

    if Beta_ind_lim is None:
        Beta_ind_lim = (0, Likelihood_grid.shape[1])

    # Figure of Observed image
    im=ax_row[0].imshow(Observed_image)
    floored_max_Flux=marginalize_value(Observed_image.max(),2,'floor')
    cbar=fig.colorbar(im,ax=ax_row[0],ticks=[0,floored_max_Flux],fraction=0.045)
    cbar.ax.set_yticklabels(['0','{:.1f}'.format(floored_max_Flux)],fontsize=fontsize)
    cbar.ax.set_ylabel('Flux',rotation=90,fontsize=fontsize,labelpad=-20)

    # Figure of GRF gravitational potential inhomogeneities
    im=ax_row[1].imshow(GRF_potential,cmap='Spectral',norm=mpl.colors.TwoSlopeNorm(0))
    colorborders=[marginalize_value(GRF_potential.min(),3,'ceil'),0,marginalize_value(GRF_potential.max(),3,'floor')]
    cbar=fig.colorbar(im,ax=ax_row[1],ticks=colorborders,fraction=0.045)
    cbar.ax.set_ylabel('Potential',rotation=90,fontsize=fontsize,labelpad=-20)
    cbar.ax.set_yticklabels(['{:.3f}'.format(colorborders[0]),'0','{:.3f}'.format(colorborders[2])],fontsize=fontsize)

    # Figure for Observed anomalies
    normalised_masked_anomalies = (Observed_anomalies*Surface_brightness.annulus_mask)/np.sqrt(Surface_brightness.noise_var)
    im = ax_row[2].imshow(normalised_masked_anomalies, cmap='seismic', norm=mpl.colors.TwoSlopeNorm(0))
    colorborders=[marginalize_value(normalised_masked_anomalies.min(),1,'ceil'),0,marginalize_value(normalised_masked_anomalies.max(),1,'floor')]
    cbar=fig.colorbar(im,ax=ax_row[2],ticks=colorborders,fraction=0.045)
    cbar.ax.set_ylabel('Normalised Flux',rotation=90,fontsize=fontsize,labelpad=-20)
    cbar.ax.set_yticklabels(['{:.1f}'.format(colorborders[0])+r'$\sigma$', '0', '{:.1f}'.format(colorborders[2])+r'$\sigma$'],fontsize=fontsize)


    # Image of Confidence grid for logA,Beta
    # It is not necessary to show all the grid, whe can crop it to a part
    Crop_Beta_arr=Beta_array[Beta_ind_lim[0]:Beta_ind_lim[1]]
    Crop_logA_arr=logA_array[logA_ind_lim[0]:logA_ind_lim[1]]
    Crop_Confidence_grid=Confidence_grid[logA_ind_lim[0]:logA_ind_lim[1],Beta_ind_lim[0]:Beta_ind_lim[1]]
    Crop_SNR_grid=SNR_grid[logA_ind_lim[0]:logA_ind_lim[1],Beta_ind_lim[0]:Beta_ind_lim[1]]

    Like_xticks=np.arange(10)
    Like_xticks=Like_xticks[np.where( (Like_xticks>=Crop_Beta_arr[0]) & (Like_xticks<=Crop_Beta_arr[-1]) )[0]]
    Like_yticks=np.arange(0,7)*0.5-10
    Like_yticks=Like_yticks[np.where( (Like_yticks>=Crop_logA_arr[0]) & (Like_yticks<=Crop_logA_arr[-1]) )[0]]

    logA_max_likelihood_index,Beta_max_likelihood_index=np.unravel_index(np.argmax(Likelihood_grid), Likelihood_grid.shape)

    # The function that plots confidence grid, true and predicted points, as well as negative SNR region
    plot_Confidence_grid(ax_row[3],Beta_array,logA_array,Crop_Confidence_grid,Crop_SNR_grid,
                         logA_true_index-logA_ind_lim[0],Beta_true_index-Beta_ind_lim[0],logA_max_likelihood_index-logA_ind_lim[0],Beta_max_likelihood_index-Beta_ind_lim[0],
                         Like_xticks,Like_yticks,confidence_labels_locations,legend_flag,fontsize)

    ax_row[3].set_xticklabels(Like_xticks,fontsize=15)
    ax_row[3].set_yticklabels(Like_yticks,fontsize=15)
    ax_row[3].set_ylabel(r"${\rm log}(A)$",fontsize=fontsize,labelpad=-5)


    ra_at_xy_0,dec_at_xy_0=Surface_brightness.pixel_grid.radec_at_xy_0

    for j in range(3):
        ax_row[j].set_xticks([0,50,100-1])
        ax_row[j].set_xticklabels(np.array([ra_at_xy_0,0,-ra_at_xy_0]).round().astype(int),fontsize=15)
        ax_row[j].set_xlabel('arcsec',fontsize=fontsize)

        ax_row[j].set_yticks([0,50,100-1])
        ax_row[j].set_yticklabels(np.array([dec_at_xy_0,0,-dec_at_xy_0]).round().astype(int),fontsize=15)
        ax_row[j].set_ylabel('arcsec',labelpad=-3,fontsize=fontsize)


    if titles_flag:
        titles=['Einstein ring\nsurface brightness','Gravitational potential\ninhomogeneities',\
                'Anomalies on\nEinstein ring','Confidence regions of\ninhomogeneities parameters']
        for j in range(4):
            ax_row[j].set_title(titles[j],fontsize=fontsize)



