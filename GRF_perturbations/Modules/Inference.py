import jax
import numpy as np
import jax.numpy as jnp
import time

from GRF_perturbations.Modules.Jax_Utils import jax_map,gradient_descent
from GRF_perturbations.Modules.GRF_generation import get_k_grid,get_Fourier_phase,get_jaxified_GRF,nonsingular_Power_spectrum
from GRF_perturbations.Modules.Image_processing import model_loss_function,Radial_profile,compute_radial_spectrum
from GRF_perturbations.Modules.Data_generation import Observation_conditions_class

from skimage import measure
import scipy.ndimage as ndimage
from scipy.optimize import minimize as scipy_minimize

import matplotlib.pyplot as plt

class Inference_class:

    def __init__(self,Observation_conditions: Observation_conditions_class,\
                 model_kwargs=None,GRF_seeds_number=None,
                 SL_fitting_max_iter=None,SL_fitting_learning_rate=None):

        self.model_kwargs=model_kwargs
        self.GRF_seeds_number=GRF_seeds_number
        self.SL_max_iter=SL_fitting_max_iter
        self.SL_learning_rate=SL_fitting_learning_rate
        self.Observation_conditions=Observation_conditions

        if self.model_kwargs is None:
            self.model_kwargs=self.Observation_conditions.kwargs_data

        if self.GRF_seeds_number is None:
            self.GRF_seeds_number=100

        if self.SL_max_iter is None:
            self.SL_max_iter=1000

        if self.SL_learning_rate is None:
            self.SL_learning_rate=5e-4

        print('Precomputing Fourier phases')
        #Precompute Fourier phases, because jax doesn't tolerate random inside pure functions
        self.Fourier_phase_tensor=np.zeros((self.GRF_seeds_number,self.Observation_conditions.pixel_number,self.Observation_conditions.pixel_number),dtype=complex)
        for i in range(self.GRF_seeds_number):
            self.Fourier_phase_tensor[i]=get_Fourier_phase(self.Observation_conditions.pixel_number,seed=i)

        #Return seed to the fixed one
        np.random.seed(42)

        print('Precompiling source-lens loss,gradient,hessian')
        simulate_unperturbed_image=self.Observation_conditions.unperturbed_image_getter
        simulate_unperturbed_image_pure= lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)

        self.image_loss= jax.jit(lambda args,image: model_loss_function(args,image,simulate_unperturbed_image_pure,\
                                                                        self.Observation_conditions.noise_var,self.Observation_conditions.parameters))
        self.image_loss_gradient=jax.grad(self.image_loss)
        self.image_loss_hessian=jax.jacfwd(jax.jit(jax.jacrev(self.image_loss)))

        #During the first use jax does XLA compilation and builds the graph. It takes a lot of time, but should be doe once
        #Given the graph it computes everything in milliseconds
        args=self.Observation_conditions.parameters.kwargs2args(self.Observation_conditions.kwargs_data)
        image=jnp.zeros((self.Observation_conditions.pixel_number,self.Observation_conditions.pixel_number))
        print('Precomputing loss')
        _=self.image_loss(args,image)
        print('Precomputing loss gradient')
        _=self.image_loss_gradient(args,image)
        print('Precomputing loss hessian')
        _=self.image_loss_hessian(args,image)
        print('Inference class is ready')

    @property
    def frequency_grids(self):
        #Grid of Fourier frequencies. nonsingular_grid doesn't have zero for Power_spectrum to not diverge
        k_grid,nonsingular_k_grid=get_k_grid(self.Observation_conditions.pixel_number, self.Observation_conditions.pixel_scale)
        return k_grid,nonsingular_k_grid

    def GRF_getters(self,from_index=True):

        k_grid,nonsingular_k_grid=self.frequency_grids

        def get_GRF_from_Phase_index(GRF_params,GRF_seed_index):

            #Check that it can be used as array index -size<=index<size
            #assert isinstance(GRF_seed_index,int)
            #assert ((GRF_seed_index>=-self.GRF_seeds_number)) and (GRF_seed_index<self.GRF_seeds_number)

            return get_jaxified_GRF(GRF_params,nonsingular_k_grid,self.Fourier_phase_tensor[GRF_seed_index])

        def get_GRF_from_Phase_Matrix(GRF_params,Fourier_phase):

            #Check that it can be used as array index -size<=index<size
            #assert isinstance(GRF_seed_index,int)
            #assert ((GRF_seed_index>=-self.GRF_seeds_number)) and (GRF_seed_index<self.GRF_seeds_number)

            return get_jaxified_GRF(GRF_params,nonsingular_k_grid,Fourier_phase)

        if from_index:
            return get_GRF_from_Phase_index
        else:
            return get_GRF_from_Phase_Matrix

    def alpha_kappa_variance(self,GRF_params):
        k_grid,nonsingular_k_grid=self.frequency_grids
        k_vector=np.fft.fftfreq(self.Observation_conditions.pixel_number,self.Observation_conditions.pixel_scale)
        kx,ky=np.meshgrid(k_vector,k_vector)

        zero_mean_mask=np.ones_like(nonsingular_k_grid)
        zero_mean_mask[0,0]=0

        potential_PS=nonsingular_Power_spectrum(GRF_params,nonsingular_k_grid)*zero_mean_mask

        #d(psi)/dx in Fourier space
        alphax_PS=(2*np.pi*kx)**2*potential_PS
        #d(psi)/dy in Fourier space
        alphay_PS=(2*np.pi*ky)**2*potential_PS

        #laplacian/2 in Fourier space
        kappa_PS=((2*np.pi*ky)**4+(2*np.pi*kx)**4)*potential_PS/4

        return alphax_PS.sum(),alphay_PS.sum(),kappa_PS.sum()



    def scipy_fit_image(self,image,method='trust-krylov',initial_values=None):


        if initial_values is None:
            initial_values=self.Observation_conditions.parameters.initial_values()

        image_loss=lambda args: self.image_loss(args,image)
        image_loss_grad= lambda args: self.image_loss_gradient(args,image)
        image_loss_hess= lambda args: self.image_loss_hessian(args,image)

        res = scipy_minimize(image_loss, initial_values,jac=image_loss_grad,hess=image_loss_hess, method=method)

        return res.x

    def differentiable_fit_image(self,image):

        model_loss_grad= lambda args: self.image_loss_gradient(args,image)

        args_guess=self.Observation_conditions.parameters.kwargs2args(self.model_kwargs)

        #Gradiend descent is but a recursion. Here its depth-limited and differentiable version
        args_fit=gradient_descent(model_loss_grad,args_guess,self.SL_max_iter,self.SL_learning_rate)

        return args_fit

    def Radial_profile(self,image):
        return Radial_profile(image,(self.Observation_conditions.pixel_number,self.Observation_conditions.pixel_number))

    def compute_radial_spectrum(self,image):
        k_grid,_=self.frequency_grids
        return compute_radial_spectrum(image,self.Observation_conditions.annulus_mask,k_grid,self.Observation_conditions.frequencies)

    def Residual_spectrum_for_GRF(self,GRF_params,Fourier_phase,Noise=True):
        get_GRF=self.GRF_getters(from_index=False)

        GRF_potential=get_GRF(GRF_params,Fourier_phase)
        #We want noise to be random or at least different for every generated GRF
        #It should complicate computation of gradients, but we want to keep the function pure
        #+1 are needed cause those parameters are great or equal to zero
        noise_seed=jnp.round(jnp.abs(GRF_params[0]*(GRF_params[1]+1)*(Fourier_phase[0,1].real*(1e+3)+1)*1e+5)).astype(int)

        simulate_perturbed_image=self.Observation_conditions.perturbed_image_getter
        simulated_image=simulate_perturbed_image(GRF_potential,self.Observation_conditions.kwargs_data,Noise,noise_seed)

        args_fit=self.differentiable_fit_image(simulated_image)

        simulate_unperturbed_image=self.Observation_conditions.unperturbed_image_getter
        fit_image=simulate_unperturbed_image(self.Observation_conditions.parameters.args2kwargs(args_fit),Noise_flag=False)

        residuals=simulated_image-fit_image
        spectrum=self.compute_radial_spectrum(residuals)
        return spectrum

    def GRF_Loss(self,GRF_params,GRF_seeds_number,Spectra_Loss_pure,Noise_flag):

        Fourier_phases=self.Fourier_phase_tensor[:GRF_seeds_number]
        get_model_spectra=jax.jit(lambda Fourier_phase: self.Residual_spectrum_for_GRF(GRF_params,Fourier_phase,Noise_flag))
        model_spectra=jax_map(get_model_spectra,Fourier_phases)

        Loss=Spectra_Loss_pure(model_spectra)
        return Loss




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




def compute_SNR_grid(Spectra_grid,Noise_spectral_density):
    SNR=10*np.log10(np.mean(Spectra_grid-Noise_spectral_density,axis=-1)/Noise_spectral_density)
    mean_SNR=np.nanmean(SNR,axis=-1)
    return mean_SNR


def compute_Confidence_grid(likelihood):
    isolevels=likelihood.max()*np.linspace(1,0,101)
    contours=[]


    confidence_grid=np.zeros_like(likelihood)*np.nan

    #Compute isolevel contours
    #for i,isolevel in enumerate(isolevels):
    #contours+=[np.array(measure.find_contours(likelihood,isolevel))]

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

def compute_Loss_grid(Spectra_grid,Spectra_Loss_pure):

    #map function over row of Betas_array and GRF_seeds
    def loop_over_Betas(Spectra_Beta_row):
        return jax_map(Spectra_Loss_pure,Spectra_Beta_row)

    Loss_grid=jax_map(loop_over_Betas,Spectra_grid)

    return Loss_grid


def Inference_pipeline(data_resid_spectrum,MU_tensor,Sigma_tensor):


    logSpec_data=np.log(data_resid_spectrum)
    Loss_grid=jnp.mean(jnp.power((logSpec_data-MU_tensor)/Sigma_tensor,2),axis=-1)


    likelihood=np.exp(-Loss_grid/2)

    res_matrix=get_conf_intervals(likelihood)
    pred_logA_index,pred_Beta_index=res_matrix[0]
    logA_conf_regions=res_matrix[1:4]
    Beta_conf_regions=res_matrix[4:]

    Confidence_grid=compute_Confidence_grid(likelihood)

    return likelihood,Confidence_grid,pred_logA_index,pred_Beta_index,logA_conf_regions,Beta_conf_regions

def Inference_pipeline_old(data_resid_spectrum,Spectra_grid,fitted_logA_index,fitted_Beta_index,report_timings=True):

    #Estimating uncertainties
    start_time=time.time()
    gamma_data,mu_data,sigma_data=infer_LogNorm_params(Spectra_grid[fitted_logA_index,fitted_Beta_index])
    time_uncertainties=time.time()-start_time


    logSpec_data=np.log(data_resid_spectrum)
    Spectra_Loss_pure=lambda model_spectra: Spectra_Loss(model_spectra,logSpec_data,sigma_data)

    #Computing Loss grid
    start_time=time.time()
    '''
    Loss_grid=np.zeros((Spectra_grid.shape[0],Spectra_grid.shape[1]))
    for i,Spectra_Beta_row in enumerate(Spectra_grid):
        for j,Spectra_Phase_row in enumerate(Spectra_Beta_row):
            Loss_grid[i,j]=Spectra_Loss_pure(Spectra_Phase_row)
    '''
    Loss_grid=compute_Loss_grid(Spectra_grid,Spectra_Loss_pure)
    time_Loss_grid=time.time()-start_time


    #pred_logA_index=np.where(Loss_grid==Loss_grid.min())[0].item()
    #pred_Beta_index=np.where(Loss_grid==Loss_grid.min())[1].item()
    likelihood=np.exp(-Loss_grid/2)

    res_matrix=get_conf_intervals(likelihood)
    pred_logA_index,pred_Beta_index=res_matrix[0]
    logA_conf_regions=res_matrix[1:4]
    Beta_conf_regions=res_matrix[4:]
    #pred_logA_index,pred_Beta_index,logA_conf_regions,Beta_conf_regions=get_conf_intervals(likelihood)

    #Computing Confidence grid
    start_time=time.time()
    Confidence_grid=compute_Confidence_grid(likelihood)
    time_Confidence_grid=time.time()-start_time


    if report_timings:
        print('Uncertainties estimation took {:.1f} second§s'.format(time_uncertainties))
        print('Loss grid computation took {:.1f} seconds'.format(time_Loss_grid))
        print('Confidence grid computation took {:.1f} seconds'.format(time_Confidence_grid))

    return likelihood,Confidence_grid,pred_logA_index,pred_Beta_index

def get_cdf(likelihood):
    cdf=jnp.cumsum(likelihood)
    normalised_cdf=cdf/cdf[-1]

    return normalised_cdf

def get_prediction(cdf,pred_index,precentage_covered):
    #indent=np.minimum(0.5,percentage_covered/200.)
    indent=precentage_covered/200.

    upper_index=jnp.argmin(jnp.abs(cdf-cdf[pred_index]-indent))
    lower_index=jnp.argmin(jnp.abs(cdf-cdf[pred_index]+indent))

    return jnp.array([lower_index,upper_index])

def get_conf_intervals(likelihood):

    Beta_cdf=get_cdf(likelihood.mean(axis=0))
    Beta_pred_index=jnp.argmin(jnp.abs(Beta_cdf-0.5))


    logA_cdf=get_cdf(likelihood.mean(axis=1))
    logA_pred_index=jnp.argmin(jnp.abs(logA_cdf-0.5))

    percentage_covered=jnp.array([68,95,99.7])

    Beta_conf_regions=jax_map(lambda percentage: get_prediction(Beta_cdf,Beta_pred_index,percentage),percentage_covered)
    logA_conf_regions=jax_map(lambda percentage: get_prediction(logA_cdf,logA_pred_index,percentage),percentage_covered)


    res_matrix=jnp.array([[logA_pred_index,Beta_pred_index]])
    res_matrix=jnp.append(res_matrix,logA_conf_regions,axis=0)
    res_matrix=jnp.append(res_matrix,Beta_conf_regions,axis=0)

    return res_matrix

def get_likelihood(data_spectrum,MU_tensor,Sigma_tensor):

    logSpec_data = jnp.log(data_spectrum)
    Loss_grid = jnp.mean(jnp.power((logSpec_data - MU_tensor) / Sigma_tensor, 2), axis=-1)
    likelihood=jnp.exp(-Loss_grid/2)

    return likelihood


def plot_likelihood(axis,Beta_array,logA_array,confidence_grid,SNR,true_logA_index,true_Beta_index,pred_logA_index,pred_Beta_index,xticks,yticks,manual_locations,legend=False,fontsize=18):

    #Confidence levels
    #smooth the grid to avoid sharp edged contours
    smooth_confidence_grid=ndimage.gaussian_filter(confidence_grid,sigma=(2,2))
    img=axis.contourf(Beta_array,logA_array,smooth_confidence_grid,[0,0.39347,0.86466,0.988891],colors=['red','indianred','rosybrown','w'],alpha=0.7)

    img=axis.contour(Beta_array,logA_array,smooth_confidence_grid,[0.39347,0.86466,0.988891],colors='k')


    fmt = {}
    strs = [r'$1\sigma$',r'$2\sigma$',r'$3\sigma$']
    for l,s in zip( img.levels, strs ):
        fmt[l] = s

    #manual_locations=[(5,20)]
    axis.clabel(img,[0.988891],inline=True,fmt={0.988891: '$3\\sigma$'},fontsize=15,manual=manual_locations[0])
    #manual_locations=[(5,20)]
    axis.clabel(img,[0.86466],inline=True,fmt={0.86466: '$2\\sigma$'},fontsize=15,manual=manual_locations[1])
    #manual_locations=[(5,20)]
    axis.clabel(img,[0.39347],inline=True,fmt={0.39347: '$1\\sigma$'},fontsize=15,manual=manual_locations[2])

    #Prediction and truth

    predPoint=axis.scatter(Beta_array[pred_Beta_index],logA_array[pred_logA_index],label='Max likelihood',marker="o",s=80,color='k',edgecolor='w',linewidth=0.5)
    truePoint=axis.scatter(Beta_array[true_Beta_index],logA_array[true_logA_index],label='Ground truth',marker="*",s=80,color='k',edgecolor='w',linewidth=0.5)

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
        l=axis.legend([truePoint,predPoint,plt.Rectangle((1, 1), 2, 2, fc=imgSNR.collections[0].get_facecolor()[0])],['Ground truth','Max likelihood',r'$SNR \leq 0$'],loc='upper right',fontsize=15,framealpha=0)
        for text in l.get_texts():
            text.set_color("k")


