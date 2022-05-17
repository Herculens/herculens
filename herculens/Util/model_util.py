import copy
import numpy as np
import jax.numpy as jnp
import jax
import time
from scipy import signal
from tqdm import tqdm
import warnings
from scipy.ndimage import morphology
from scipy import ndimage

from herculens.Util import linear_util, jax_util
from herculens.VKL_operator import vkl_operator as vkl_util



def mask_from_source_area(lens_image, parameters):
    src_idx = lens_image.SourceModel.pixelated_index
    kwargs_param_mask = copy.deepcopy(parameters.current_values(as_kwargs=True))
    pixels = kwargs_param_mask['kwargs_source'][src_idx]['pixels']
    pixels = np.zeros_like(pixels)
    pixels[3:-3, 3:-3] = 1.  # so the source plane is ones except in a small margin all around
    kwargs_param_mask['kwargs_source'][src_idx]['pixels'] = jnp.array(pixels)
    model_mask = lens_image.source_surface_brightness(kwargs_param_mask['kwargs_source'],
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    model_mask = morphology.binary_opening(model_mask, iterations=10)
    model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask


def mask_from_lensed_source(lens_image, parameters=None, source_model=None,
                            threshold=0.1, smoothing=0):
    # imports are here to avoid issues with circular imports
    from herculens.LensImage.lens_image import LensImage
    from herculens.LightModel.light_model import LightModel

    if parameters is None and source_model is None:
        raise ValueError("You must provide an the source model "
                         "if no parameters are provided.")
    if parameters is not None:
        kwargs_param = parameters.current_values(as_kwargs=True)
        source_model = lens_image.source_surface_brightness(kwargs_param['kwargs_source'], de_lensed=True, unconvolved=True)
    source_model = np.array(source_model)
    if smoothing > 0:
        source_model = ndimage.gaussian_filter(source_model, sigma=smoothing)
    binary_source = source_model / source_model.max()
    binary_source[binary_source < threshold] = 0.
    binary_source[binary_source >= threshold] = 1.
    grid = copy.deepcopy(lens_image.Grid)
    grid.remove_model_grid('source')
    lens_image_pixel = LensImage(grid, lens_image.PSF, 
                                 noise_class=lens_image.Noise,
                                 lens_model_class=lens_image.LensModel,
                                 source_model_class=LightModel(['PIXELATED']),
                                 lens_light_model_class=lens_image.LensLightModel,
                                 kwargs_numerics=lens_image._kwargs_numerics)
    kwargs_param_mask = copy.deepcopy(kwargs_param)
    kwargs_param_mask['kwargs_source'] = [{'pixels': jnp.array(binary_source)}]
    model_mask = lens_image_pixel.source_surface_brightness(kwargs_param_mask['kwargs_source'], 
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    #model_mask = morphology.binary_opening(model_mask, iterations=10)
    #model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask, binary_source


def pixelated_region_from_sersic(kwargs_sersic, use_major_axis=False,
                                 min_width=1.0, min_height=1.0, scaling=1.0):
    # imports are here to avoid issues with circular imports
    from herculens.Util import param_util

    # TODO: support arbitrary smooth source profile
    c_x = kwargs_sersic['center_x']
    c_y = kwargs_sersic['center_y']
    r_eff = kwargs_sersic['R_sersic']
    if 'e1' in kwargs_sersic:
        e1 = kwargs_sersic['e1']
        e2 = kwargs_sersic['e2']
    else:
        e1 = e2 = 0.
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    a = r_eff / np.sqrt(q)  # semi-major axis
    r = r_eff * np.sqrt(q)  # product average radius
    print(f"r_eff={r_eff:.2f}, r={r:.2f}, a={a:.2f}")
    diameter = 2*a if use_major_axis else 2*r
    width = diameter * np.abs(np.cos(phi)) * scaling
    height = diameter * np.abs(np.sin(phi)) * scaling
    width = max(min_width, width)
    height = max(min_height, height)
    # the following dict is consistent with arguments of PixelGrid.create_model_grid()
    kwargs_pixelated_grid = {
        'grid_center': [float(c_x), float(c_y)],
        'grid_shape': [float(width), float(height)],
    }
    return kwargs_pixelated_grid


def halo_sensitivity_map(macro_lens_image, macro_parameters, data, 
                         halo_profile='SIS', kwargs_numerics=None,
                         init_mass_proxy_param=0., norm_mass_proxy_param=1.,
                         x_grid=None, y_grid=None):
    """
    init_mass_proxy_param is the value of the mass parameter (e.g. theta_E)
    that is used for evaluated the gradient of the loss function.
    """
    # imports are here to avoid issues with circular imports
    from herculens.LightModel.light_model import LightModel
    from herculens.LensModel.lens_model import LensModel
    from herculens.LensImage.lens_image import LensImage
    from herculens.Parameters.parameters import Parameters
    from herculens.Inference.loss import Loss
    from herculens.Util import util

    if halo_profile == 'POINT_MASS':
        kwargs_halo_fixed = {}
        kwargs_halo_init = {'theta_E': init_mass_proxy_param, 'center_x': 0., 'center_y': 0.}
    elif halo_profile == 'PIXELATED_DIRAC':
        kwargs_halo_fixed = {}
        kwargs_halo_init = {'psi': init_mass_proxy_param, 'center_x': 0., 'center_y': 0.}
    elif halo_profile == 'SIS':
        halo_profile = 'SIE'  # because no standalone SIS profile in Herculens so far
        kwargs_halo_fixed = {'e1': 0., 'e2': 0.}
        kwargs_halo_init = {'theta_E': init_mass_proxy_param, 'center_x': 0., 'center_y': 0.}
    else:
        raise NotImplementedError(f"Halo profile '{halo_profile}' is not yet supported.")
    
    halo_lens_model_list = [halo_profile] + macro_lens_image.LensModel.lens_model_list
    halo_lens_model = LensModel(halo_lens_model_list)

    grid = copy.deepcopy(macro_lens_image.Grid)
    #grid.remove_model_grid('lens')
    psf = copy.deepcopy(macro_lens_image.PSF)
    noise = copy.deepcopy(macro_lens_image.Noise)
    halo_lens_image = LensImage(grid, psf, noise_class=noise,
                                lens_model_class=halo_lens_model,
                                source_model_class=macro_lens_image.SourceModel,
                                lens_light_model_class=macro_lens_image.LensLightModel,
                                kwargs_numerics=kwargs_numerics)

    kwargs_macro = macro_parameters.current_values(as_kwargs=True)
    kwargs_fixed = {
        #'kwargs_lens': [kwargs_halo_fixed] + [{} for _ in range(len(kwargs_macro['kwargs_lens']))], # + kwargs_macro['kwargs_lens'],
        'kwargs_lens': [kwargs_halo_fixed] + kwargs_macro['kwargs_lens'],
        'kwargs_source': kwargs_macro['kwargs_source'],
        'kwargs_lens_light': kwargs_macro['kwargs_lens_light'],
    }
    kwargs_init = {
        'kwargs_lens': [kwargs_halo_init] + kwargs_macro['kwargs_lens'],
        'kwargs_source': [{} for _ in range(len(kwargs_macro['kwargs_source']))],
        'kwargs_lens_light': [{} for _ in range(len(kwargs_macro['kwargs_lens_light']))],
    }
    halo_parameters = Parameters(halo_lens_image, kwargs_init, kwargs_fixed)
    print("num. params:", halo_parameters.num_parameters)
    print("init. params:", halo_parameters.initial_values())

    # create the loss to minimize
    halo_loss = Loss(data, halo_lens_image, halo_parameters, likelihood_type='chi2')

    #p_macro = copy.deepcopy(macro_parameters.current_values(as_kwargs=False)).tolist()

    # define the function that computes sensitivity at a given pixel (x, y)
    @jax.jit
    def sensitivity_at_pixel(x, y):
        #p = [init_mass_proxy_param, x, y] + p_macro
        p = [init_mass_proxy_param, x, y]
        grad_loss_mass = jax.grad(halo_loss)(p)
        partial_deriv_mass_proxy = grad_loss_mass[0]
        return partial_deriv_mass_proxy / norm_mass_proxy_param

    # efficiently compute sensitivity on the data grid
    if halo_lens_image.ImageNumerics.grid_supersampling_factor > 1:
        # note: those are 1D arrays
        x_grid, y_grid = halo_lens_image.ImageNumerics.coordinates_evaluate
    elif x_grid is None or y_grid is None:
        x_grid, y_grid = halo_lens_image.Grid.pixel_coordinates
    
    # evaluate the sensitivity over the coordinates grid
    sensitivity_map = jnp.vectorize(sensitivity_at_pixel)(x_grid, y_grid).block_until_ready()

    # convert to numpy array and reshape
    sensitivity_map = np.array(sensitivity_map)
    if len(sensitivity_map.shape) == 1:
        sensitivity_map = util.array2image(sensitivity_map)

    # get the coordinates where the sensitivity is the highest
    from skimage import feature
    peak_indices_2d = feature.peak_local_max(-np.clip(sensitivity_map, a_min=None, a_max=0))
    
    x_coords, y_coords = halo_lens_image.Grid.pixel_axes
    x_minima = x_coords[peak_indices_2d[:, 1]]
    y_minima = y_coords[peak_indices_2d[:, 0]]
    z_minima = sensitivity_map[peak_indices_2d[:, 1], peak_indices_2d[:, 0]]

    return sensitivity_map, (x_minima, y_minima, z_minima)


def pixel_pot_noise_map(lens_image, kwargs_res, k_src=None, cut=1e-5):
    """EMPIRICAL noise map based on the ivnerse of the sqrt of the source model"""
    # imports are here to avoid issues with circular imports
    from herculens.Util import image_util

    ls_0 = lens_image.source_surface_brightness(kwargs_res['kwargs_source'],
                                                kwargs_lens=kwargs_res['kwargs_lens'],
                                                de_lensed=False, unconvolved=False,
                                                k=k_src)
    ls_0 = np.array(ls_0)
    #print(ls_0.min(), ls_0.max())

    #scaled_ls_0 = ls_0
    scaled_ls_0 = ls_0**(1/2.)
    #scaled_ls_0 = ls_0**(1/3.)

    potential_noise_map = np.zeros_like(ls_0)
    potential_noise_map[ls_0 > cut] = 1. / scaled_ls_0[ls_0 > cut]
    potential_noise_map[ls_0 <= cut] = potential_noise_map.max()

    # normalize by data noise
    noise_map = np.sqrt(lens_image.Noise.C_D)  # TODO: replace by .C_D_model()
    #print(noise_map.mean())
    potential_noise_map *= noise_map

    # rescaled to potential grid
    x_in, y_in = lens_image.Grid.pixel_axes
    x_out, y_out = lens_image.Grid.model_pixel_axes('lens')
    potential_noise_map = image_util.re_size_array(x_in, y_in, potential_noise_map, x_out, y_out)
    
    return potential_noise_map


def pixel_pot_noise_map_deriv(lens_image, kwargs_res, k_src=None, cut=1e-5, 
                              use_model_covariance=True,
                              wavelet_type_list=['starlet', 'battle-lemarie-3']):
    """EMPIRICAL noise map (although inspired by Koopmans+05) as the inverse of the blurred source derivative"""
    # imports are here to avoid issues with circular imports
    from herculens.Util.jax_util import BicubicInterpolator as Interpolator
    from herculens.Util import image_util, util
    # TODO: fix the inconsitent use of either Interpolator or re_size_array method

    # data coordinates
    x_grid, y_grid = lens_image.Grid.pixel_coordinates

    # numerics grid, for intermediate computation on a higher resolution grid
    x_grid_num, y_grid_num = lens_image.ImageNumerics.coordinates_evaluate
    x_grid_num = util.array2image(x_grid_num)
    y_grid_num = util.array2image(y_grid_num)
    x_coords_num, y_coords_num = x_grid_num[0, :], y_grid_num[:, 0]
    s_0 = lens_image.SourceModel.surface_brightness(x_grid_num, y_grid_num, 
                                                    kwargs_res['kwargs_source'], 
                                                    k=k_src)
    interp_source = Interpolator(y_coords_num, x_coords_num, s_0)
    grad_s_x_srcplane = interp_source(y_grid_num, x_grid_num, dy=1)
    grad_s_y_srcplane = interp_source(y_grid_num, x_grid_num, dx=1)
    # compute its derivatives *on source plane*
    grad_s_x_srcplane = interp_source(y_grid_num, x_grid_num, dy=1)
    grad_s_y_srcplane = interp_source(y_grid_num, x_grid_num, dx=1)
    # setup the Interpolator to read on data pixels
    interp_grad_s_x = Interpolator(y_coords_num, x_coords_num, grad_s_x_srcplane)
    interp_grad_s_y = Interpolator(y_coords_num, x_coords_num, grad_s_y_srcplane)
    # use the lens equation to ray shoot the coordinates of the data grid
    x_src, y_src = lens_image.LensModel.ray_shooting(
        x_grid, y_grid, kwargs_res['kwargs_lens'])
    # evaluate the resulting arrays on that grid
    grad_s_x = interp_grad_s_x(y_src, x_src)
    grad_s_y = interp_grad_s_y(y_src, x_src)
    # proper flux units
    pixel_area = lens_image.Grid.pixel_area
    grad_s_x = np.array(grad_s_x) * pixel_area
    grad_s_y = np.array(grad_s_y) * pixel_area
    grad_s = np.hypot(grad_s_x, grad_s_y)

    # convolve with PSF
    grad_s = lens_image.ImageNumerics.convolution_class.convolution2d(grad_s)

    potential_noise_map = np.zeros_like(grad_s)
    potential_noise_map[grad_s > cut] = 1. / grad_s[grad_s > cut]
    potential_noise_map[grad_s <= cut] = potential_noise_map.max()

    # normalize by data noise
    if use_model_covariance:
        if lens_image.Noise.exposure_map is None:
            warnings.warn("Exposure map is None, model variance might not be computed properly.")
        model_source_only = lens_image.model(**kwargs_res, lens_light_add=False)
        C_D_model = lens_image.Noise.covariance_matrix(model_source_only, 
                                                       lens_image.Noise.background_rms, 
                                                       lens_image.Noise.exposure_map)
        noise_map = np.sqrt(C_D_model)
    else:
        noise_map = np.sqrt(lens_image.Noise.C_D)
    potential_noise_map *= noise_map

    # rebin to potential grid
    x_in, y_in = lens_image.Grid.pixel_axes
    x_out, y_out = lens_image.Grid.model_pixel_axes('lens')
    potential_noise_map = image_util.re_size_array(x_in, y_in, potential_noise_map, x_out, y_out)

    # rescale to each wavelet scale
    psi_wt_std_list = []
    nx_psi, ny_psi = potential_noise_map.shape
    for wavelet_type in wavelet_type_list:
        if wavelet_type in ['battle-lemarie-1', 'battle-lemarie-3']:
            num_scales = 1  # we only care about the first scale for this one
        else:
            num_scales = int(np.log2(min(nx_psi, ny_psi)))
        wavelet = jax_util.WaveletTransform(num_scales, wavelet_type=wavelet_type, second_gen=False)
        wavelet_norms = wavelet.scale_norms  # ignore coarsest scale
        psi_wt_std = []
        for wavelet_norm in wavelet_norms:
            psi_wt_std.append(wavelet_norm * potential_noise_map)
        psi_wt_std_list.append(np.array(psi_wt_std))

    return psi_wt_std_list


def data_noise_to_wavelet_potential(lens_image, kwargs_res, k_src=None,
                                    wavelet_type_list=['starlet', 'battle-lemarie-3'], 
                                    starlet_second_gen=False,
                                    method='MC', num_samples=10000, seed=None, 
                                    ignore_poisson_noise=False, ignore_lens_light_flux=False,
                                    model_var_map=None, verbose=False):
    lens_model = lens_image.LensModel
    kwargs_lens = kwargs_res['kwargs_lens']
    source_model = lens_image.SourceModel
    kwargs_source = kwargs_res['kwargs_source']

    # get model data variance
    model = lens_image.model(**kwargs_res, lens_light_add=(not ignore_lens_light_flux))
    if ignore_poisson_noise:
        data_var_map = lens_image.Noise.background_rms**2 * np.ones_like(model)
    else:
        data_var_map = lens_image.Noise.C_D_model(model, force_recompute=True)
    if model_var_map is not None:
        var_map = data_var_map + model_var_map  # add variances
    else:
        var_map = data_var_map
    # import matplotlib.pyplot as plt
    # plt.imshow(var_map)
    # plt.colorbar()
    # plt.show()
    # raise
    var_map = np.array(var_map)  # cast to std numpy array otherwise computations are slowed down
    std_map = np.sqrt(var_map)
    var_d = var_map.flatten()
    std_d = std_map.flatten()

    # extract coordinates grid, in image plane, ray-shot to source plane, and for the pixelated potential
    x_grid_d, y_grid_d = lens_image.Grid.pixel_coordinates
    x_grid_rs, y_grid_rs = lens_model.ray_shooting(x_grid_d, y_grid_d, kwargs_lens)
    x_grid_psi, y_grid_psi = lens_image.Grid.model_pixel_coordinates('lens')

    # number of pixels and wavelet scales
    nx_d, ny_d = x_grid_d.shape
    nx_psi, ny_psi = x_grid_psi.shape
    
    # compute derivatives of the source light at ray-shot coordinates
    source_deriv_x, source_deriv_y = source_model.spatial_derivatives(x_grid_rs, 
                                                                      y_grid_rs, 
                                                                      kwargs_source, k=k_src)
    source_deriv_x *= lens_image.Grid.pixel_area  # correct flux units
    source_deriv_y *= lens_image.Grid.pixel_area  # correct flux units

    # reshape all quantities for building the operator
    data_x = x_grid_d.flatten()
    data_y = np.flip(y_grid_d, axis=0).flatten()  # WARNING: y-coords must be flipped vertically!
    dpsi_x = x_grid_psi.flatten()
    dpsi_y = np.flip(y_grid_psi, axis=0).flatten()  # WARNING: y-coords must be flipped vertically!
    dpsi_dx = abs(x_grid_psi[0, 0] - x_grid_psi[0, 1])
    dpsi_dy = abs(y_grid_psi[0, 0] - y_grid_psi[1, 0])
    dpsi_xmin = x_grid_psi[0, :].min() - dpsi_dx/2.
    dpsi_ymax = y_grid_psi[:, 0].max() + dpsi_dy/2.
    dpsi_Nx, dpsi_Ny = nx_psi, ny_psi
    source0_dx = source_deriv_x.flatten()
    source0_dy = source_deriv_y.flatten()

    # get the DsD operator (see Koopmans+05)
    start = time.time()
    DsDpsi_matrix = vkl_util.vkl_operator(np.array(data_x), np.array(data_y),
                                          dpsi_xmin, dpsi_dx, dpsi_Nx, np.array(dpsi_x), 
                                          dpsi_ymax, dpsi_dy, dpsi_Ny, np.array(dpsi_y),
                                          np.array(source0_dx), np.array(source0_dy))
    if verbose: print("compute DsDpsi:", time.time()-start)

    # get the blurring operator for PSF convolutions
    start = time.time()
    psf_kernel_2d = np.array(lens_image.PSF.kernel_point_source)
    B_matrix = linear_util.build_convolution_matrix(psf_kernel_2d, (nx_d, ny_d))
    if verbose: print("compute B:", time.time()-start)

    # D operator
    start = time.time()
    D_matrix  = - B_matrix.dot(DsDpsi_matrix)
    DT_matrix = D_matrix.T
    if verbose: print("construct D:", time.time()-start)

    # wavelet transform Phi^T operators
    PhiT_operator_list = []
    num_scales_list = []
    for wavelet_type in wavelet_type_list:
        if wavelet_type in ['battle-lemarie-1', 'battle-lemarie-3']:
            num_scales = 1  # we only care about the first scale for this one
        else:
            num_scales = int(np.log2(min(nx_psi, ny_psi)))
        wavelet = jax_util.WaveletTransform(num_scales, 
                                            wavelet_type=wavelet_type, 
                                            second_gen=starlet_second_gen)
        PhiT_operator_list.append(wavelet.decompose)
        num_scales_list.append(num_scales)


    if method == 'MC':

        # initialize random generator seed
        np.random.seed(seed)

        # draw samples from the data covariance matrix
        # cov_d = np.diag(var_d)
        # mu = np.zeros(nx_d*ny_d)  # zero mean
        # noise_reals = np.random.multivariate_normal(mu, cov_d, size=num_samples)
        
        # apply the operators for each realization of the noise
        psi_wt_std_list = []
        for wavelet_type, PhiT_operator in zip(wavelet_type_list, PhiT_operator_list):

            start = time.time()
            psi_wt_reals = []
            for i in range(num_samples):
                # noise_i = noise_reals[i, :]
                noise_i = std_d * np.random.randn(*std_d.shape)  # draw a noise realization
                psi_i = DT_matrix.dot(noise_i)
                psi_wt_i = PhiT_operator( psi_i.reshape(nx_psi, ny_psi) )
                psi_wt_reals.append(psi_wt_i)
            psi_wt_reals = np.array(psi_wt_reals) # --> shape = (num_samples, num_scales, nx_psi, ny_psi)
            if verbose: print(f"loop over MC samples for wavelet '{wavelet_type}':", time.time()-start)

            # compute the variance per wavelet scale per potential pixel over all the samples
            psi_wt_var = np.var(psi_wt_reals, axis=0)
            # check
            if np.any(psi_wt_var < 0.):
                raise ValueError("Negative variance terms!")

            # convert to standard deviation
            psi_wt_std = np.sqrt(psi_wt_var)

            psi_wt_std_list.append(psi_wt_std)


    elif method == 'SLIT':

        psi_wt_std_list = []
        for PhiT_operator, num_scales in zip(PhiT_operator_list, num_scales_list):

            # following the same recipe as SLIT(ronomy):

            # first term of
            B_noise = std_map * np.sqrt(np.sum(psf_kernel_2d.T**2))
            B_noise = B_noise.flatten()
            DsDpsiB_noise = (DsDpsi_matrix.T).dot(B_noise)
            DsDpsiB_noise = DsDpsiB_noise.reshape(nx_psi, ny_psi)

            dirac = np.zeros((nx_psi, ny_psi))
            if nx_psi % 2 == 0:
                warnings.warn("Expect the potential noise maps to be shifted by one pixel (psi grid has even size!).")
            dirac[nx_psi//2, ny_psi//2] = 1
            dirac_wt = PhiT_operator(dirac)

            psi_wt_std = []
            for k in range(num_scales+1):
                psi_wt_std2_k = signal.fftconvolve(DsDpsiB_noise**2, dirac_wt[k]**2, mode='same')
                psi_wt_std_k = np.sqrt(psi_wt_std2_k)
                psi_wt_std.append(psi_wt_std_k)
            psi_wt_std = np.array(psi_wt_std) # --> shape = (num_scales, nx_psi, ny_psi)

            psi_wt_std_list.append(psi_wt_std)

    else:
        raise ValueError(f"Method '{method}' for noise propagation is not supported.")
    
    return psi_wt_std_list

def estimate_model_covariance(lens_image, parameters, samples, return_cross_covariance=False):
    model_samples = []
    for sample in samples:
        kwargs_sample = parameters.args2kwargs(sample)
        model_sample = lens_image.model(**kwargs_sample)
        model_map_shape = model_sample.shape
        model_samples.append(model_sample.flatten())
    model_samples = np.array(model_samples)
    
    # variance map (as a 2D image)
    model_var_map = np.var(model_samples, axis=0).reshape(*model_map_shape)
    
    # full (auto) covariance matrix from the model samples
    model_cov = np.cov(model_samples, rowvar=False)

    # computation of the cross-covariance matrix from model and data samples
    if return_cross_covariance:
        data_mean_proxy = lens_image.model(**parameters.best_fit_values(as_kwargs=True))   # best-fit model as a proxy for the unknown data 'mean'
        data_var = lens_image.Noise.C_D_model(data_mean_proxy, force_recompute=True)
        data_mean_proxy = data_mean_proxy.flatten()
        data_cov = np.diag(data_var.flatten())
        data_samples = draw_samples_from_covariance(data_mean_proxy, data_cov, num_samples=len(model_samples))
        data_vector  = np.mean(data_samples - data_samples.mean(axis=0), axis=0)
        model_vector = np.mean(model_samples - model_samples.mean(axis=0), axis=0)
        print("D", data_vector.shape)
        print("M", model_vector.shape)
        data_model_cross_cov = np.outer(data_vector, model_vector)
        return model_var_map, model_cov, data_model_cross_cov
    else:
        return model_var_map, model_cov

def draw_samples_from_covariance(mean, covariance, num_samples=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.multivariate_normal(mean, covariance, size=num_samples)
    return samples
