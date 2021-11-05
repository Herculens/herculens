import jax
import numpy as np
import jax.numpy as jnp
import time

from GRF_perturbations.Modules.Jax_Utils import jax_map

from skimage import measure
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

def compute_spectrum(logA,Beta,GRF_seed,get_GRF,simulate_perturbed_image_pure,differentiable_fit_image_pure,compute_radial_spectrum_pure):

    GRF_potential=get_GRF([logA,Beta],GRF_seed)

    #We want noise to be random or at least different for every generated GRF
    #It should complicate computation of gradients, but we want to keep the function pure
    #+1 are needed cause those parameters are great or equal to zero
    noise_seed=jnp.round(jnp.abs(logA*(Beta+1)*(GRF_seed+1)*1e+5)).astype(int)

    simulated_image=simulate_perturbed_image_pure(GRF_potential,noise_seed)
    fit_image=differentiable_fit_image_pure(simulated_image)

    residuals=simulated_image-fit_image
    spectrum=compute_radial_spectrum_pure(residuals)
    return spectrum

#Infer parameters of the distribution assuming the data is distributed LogNormally
def infer_LogNorm_params(Power_spectrum_maxtrix):
    '''https://digitalcommons.fiu.edu/cgi/viewcontent.cgi?article=1677&context=etd'''
    #Moments of distribution
    mean=jnp.mean(Power_spectrum_maxtrix,axis=0)
    variance=jnp.power(jnp.std(Power_spectrum_maxtrix,axis=0),2)
    median=jnp.median(Power_spectrum_maxtrix,axis=0)

    #Parameters of logNormal distribution
    sigma=jnp.sqrt(jnp.log(1+variance/jnp.power(mean,2)))
    mu=jnp.log(mean)-jnp.power(sigma,2)/2
    gamma=jnp.maximum(median-jnp.exp(mu),0)

    return gamma,mu,sigma

def Spectra_Loss(model_spectra,data_logSpectrum,uncertainties_logSpectrum):

    #Infer mu of model's LogNormal distribution
    mean=jnp.mean(model_spectra,axis=0)
    variance=jnp.power(jnp.std(model_spectra,axis=0),2)
    sigma_model=jnp.sqrt(jnp.log(1+variance/jnp.power(mean,2)))
    mu_model=jnp.log(mean)-jnp.power(sigma_model,2)/2

    #-LogLikelihood for LogNormal distribution
    chi_sq=jnp.mean(jnp.power((data_logSpectrum-mu_model)/uncertainties_logSpectrum,2))

    return chi_sq

def distribution_Noise_LogSpectrum(noise_var,compute_radial_spectrum_pure):
    Noise_spectra=jax_map(compute_radial_spectrum_pure,np.array([np.random.normal(0,np.sqrt(noise_var)) for i in range(10000)]))
    gamma,mu,sigma=infer_LogNorm_params(Noise_spectra)
    return mu,sigma


def GRF_Loss(GRF_params,GRF_seeds,compute_spectrum_pure,Spectra_Loss_pure):

    get_model_spectra=jax.jit(lambda GRF_seed: compute_spectrum_pure(GRF_params[0],GRF_params[1],GRF_seed))
    model_spectra=jax_map(get_model_spectra,GRF_seeds)

    Loss=Spectra_Loss_pure(model_spectra)
    return Loss

def compute_SNR_grid(Spectra_grid,Noise_spectral_density):
    SNR=10*np.log10(np.mean(Spectra_grid-Noise_spectral_density,axis=-1)/Noise_spectral_density)
    mean_SNR=np.nanmean(SNR,axis=-1)
    return mean_SNR

def compute_Loss_grid(Spectra_grid,Spectra_Loss_pure):
    Loss_grid=np.zeros((Spectra_grid.shape[0],Spectra_grid.shape[1]))

    for i,Spectra_Beta_row in enumerate(Spectra_grid):
        for j,Spectra_Phase_row in enumerate(Spectra_Beta_row):
            #Spectrum=Spectra_Phase_row.mean(axis=0)
            Loss_grid[i,j]=Spectra_Loss_pure(Spectra_Phase_row)

    pred_logA_index=np.where(Loss_grid==Loss_grid.min())[0].item()
    pred_Beta_index=np.where(Loss_grid==Loss_grid.min())[1].item()

    return Loss_grid,pred_logA_index,pred_Beta_index

def compute_Confidence_grid(likelihood):
    isolevels=np.linspace(1,0,101)
    contours=[]


    confidence_grid=np.zeros_like(likelihood)*np.nan

    #Compute isolevel contours
    for i,isolevel in enumerate(isolevels):
        contours+=[np.array(measure.find_contours(likelihood,isolevel))]

    confidence_levels=np.zeros_like(isolevels)

    #Compute cumulative probability enclosed in them
    for i,isolevel in enumerate(isolevels):
        above_isolevel_mask=(likelihood>=isolevel)
        confidence_levels[i]=np.nansum(likelihood*above_isolevel_mask)

        undefined_confidence_mask=np.isnan(confidence_grid)

        #Mask the region to be assigned with confidence
        confidence_mask=above_isolevel_mask*undefined_confidence_mask
        #Change selected pixels to given confidence
        confidence_grid[np.where(confidence_mask==True)]=confidence_levels[i]


    confidence_grid/=confidence_levels[-1]

    return confidence_grid

def Inference_pipeline(data_resid_spectrum,Spectra_grid,fitted_logA_index,fitted_Beta_index):

    print('Estimating uncertainties')
    start_time=time.time()
    gamma_data,mu_data,sigma_data=infer_LogNorm_params(Spectra_grid[fitted_logA_index,fitted_Beta_index])
    end_time=time.time()
    print('Uncertainties estimation took {:.1f} seconds'.format(end_time-start_time))

    logSpec_data=np.log(data_resid_spectrum)
    Spectra_Loss_pure=lambda model_spectra: Spectra_Loss(model_spectra,logSpec_data,sigma_data)

    print('Computing Loss grid')
    start_time=time.time()
    Loss_grid=np.zeros((Spectra_grid.shape[0],Spectra_grid.shape[1]))
    for i,Spectra_Beta_row in enumerate(Spectra_grid):
        for j,Spectra_Phase_row in enumerate(Spectra_Beta_row):
            Loss_grid[i,j]=Spectra_Loss_pure(Spectra_Phase_row)
    end_time=time.time()
    print('Loss grid computation took {:.1f} seconds'.format(end_time-start_time))

    pred_logA_index=np.where(Loss_grid==Loss_grid.min())[0].item()
    pred_Beta_index=np.where(Loss_grid==Loss_grid.min())[1].item()
    likelihood=np.exp(-Loss_grid/2)

    print('Computing Confidence grid')
    start_time=time.time()
    Confidence_grid=compute_Confidence_grid(likelihood)
    end_time=time.time()
    print('Confidence grid computation took {:.1f} seconds'.format(end_time-start_time))

    return likelihood,Confidence_grid,pred_logA_index,pred_Beta_index



def plot_likelihood(axis,Beta_array,logA_array,confidence_grid,SNR,true_logA_index,true_Beta_index,pred_logA_index,pred_Beta_index,xticks,yticks,legend=False,fontsize=18):

    #Confidence levels
    #smooth the grid to avoid sharp edged contours
    smooth_confidence_grid=ndimage.gaussian_filter(confidence_grid,sigma=(2,2))
    img=axis.contourf(Beta_array,logA_array,smooth_confidence_grid,[0,0.39347,0.86466,0.988891],colors=['red','indianred','rosybrown','w'],alpha=0.7)

    img=axis.contour(Beta_array,logA_array,smooth_confidence_grid,[0.39347,0.86466,0.988891],colors='k')


    fmt = {}
    strs = [r'$1\sigma$',r'$2\sigma$',r'$3\sigma$']
    for l,s in zip( img.levels, strs ):
        fmt[l] = s

    manual_locations=[(5,20)]
    axis.clabel(img,[0.988891],inline=True,fmt={0.988891: '$3\\sigma$'},fontsize=15,manual=manual_locations)
    manual_locations=[(5,20)]
    axis.clabel(img,[0.86466],inline=True,fmt={0.86466: '$2\\sigma$'},fontsize=15,manual=manual_locations)
    manual_locations=[(5,20)]
    axis.clabel(img,[0.39347],inline=True,fmt={0.39347: '$1\\sigma$'},fontsize=15,manual=manual_locations)

    #Prediction and truth

    predPoint=axis.scatter(Beta_array[pred_Beta_index],logA_array[pred_logA_index],label='Predicted value',marker="o",s=80,color='k')
    truePoint=axis.scatter(Beta_array[true_Beta_index],logA_array[true_logA_index],label='True value',marker="*",s=80,color='k')

    #SNR constraint

    imgSNR=axis.contourf(Beta_array,logA_array,SNR,[SNR.min(),0],colors='grey',alpha=0.5)


    img=axis.contour(Beta_array,logA_array,SNR,[0],colors='k',linestyles='--')


    axis.set_xticks(xticks)
    axis.set_yticks(yticks)
    axis.set_xlabel(r'$\beta$',fontsize=fontsize)
    axis.set_ylabel(r"${\rm log}(A)$",fontsize=fontsize)

    #axis.set_xticklabels([1,2,3,4,5])
    #.xticks([1,2,3,4,5])
    #plt.yticks([-9,-8.5,-8,-8.5,-7.5,-7])

    if legend:
        l=axis.legend([truePoint,predPoint,plt.Rectangle((1, 1), 2, 2, fc=imgSNR.collections[0].get_facecolor()[0])],['True value','Predicted value','SNR constraint'],loc='upper left',fontsize=15,framealpha=0)
        for text in l.get_texts():
            text.set_color("k")
