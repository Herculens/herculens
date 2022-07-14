# Classes and functions to use with JAX
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


from functools import partial
from copy import deepcopy
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.special import gammaln
from jax.scipy.signal import convolve2d
from jax.scipy.stats import norm
from jax.lax import conv_general_dilated, conv_dimension_numbers


def unjaxify_kwargs(kwargs_params):
    """
    Utility to convert all JAX's device arrays contained in a model kwargs 
    to standard floating point or numpy arrays.
    """
    kwargs_params_new = deepcopy(kwargs_params)
    for model_key, model_kwargs in kwargs_params.items():
        for profile_idx, profile_kwargs in enumerate(model_kwargs):
            for param_key, param_value in profile_kwargs.items():
                if not isinstance(param_value, (float, int)):
                    if param_value.size == 1:
                        kwargs_params_new[model_key][profile_idx][param_key] = float(param_value)
                    else:
                        kwargs_params_new[model_key][profile_idx][param_key] = np.array(param_value)
    return kwargs_params_new


def R_omega(z, t, q, nmax):
    """Angular dependency of the deflection angle in the EPL lens profile.

    The computation follows Eqs. (22)-(29) in Tessore & Metcalf (2015), where
    z = R * e^(i * phi) is a position vector in the lens plane,
    t = gamma - 1 is the logarithmic slope of the profile, and
    q is the axis ratio.

    This iterative implementation is necessary, since the usual hypergeometric
    function `hyp2f1` provided in `scipy.special` has not yet been implemented
    in an autodiff way in JAX.

    Note that the value returned includes an extra factor R multiplying Eq. (23)
    for omega(phi).

    """
    # Set the maximum number of iterations
    # nmax = 10
    
    # Compute constant factors
    f = (1. - q) / (1. + q)
    ei2phi = z / z.conjugate()
    # Set the first term of the series
    omega_i = z  # jnp.array(np.copy(z))  # Avoid overwriting z ?
    partial_sum = omega_i

    for i in range(1, nmax):
        # Iteration-dependent factor
        ratio = (2. * i - (2. - t)) / (2. * i + (2 - t))
        # Current Omega term proportional to the previous term
        omega_i = -f * ratio * ei2phi * omega_i
        # Update the partial sum
        partial_sum += omega_i
    return partial_sum



class special(object):
    @staticmethod
    @jit
    def gamma(x):
        """Gamma function.

        This function is necessary in many lens mass models, but JAX does not
        currently have an implementation in jax.scipy.special. Values of the
        function are computed simply as the exponential of the logarithm of the
        gamma function (which has been implemented in jax), taking the sign
        of negative inputs properly into account.

        Note that even when just-in-time compiled, this function is much
        slower than its original scipy counterpart.

        Parameters
        ----------
        x : array_like
            Real-valued argument.

        Returns
        -------
        scalar or ndarray
            Values of the Gamma function.
        """
        # Properly account for the sign of negative x values
        sign_condition = (x > 0) | (jnp.ceil(x) % 2 != 0) | (x % 2 == 0)
        sign = 2 * jnp.asarray(sign_condition, dtype=float) - 1
        return sign * jnp.exp(gammaln(x))


class GaussianFilter(object):
    """JAX-friendly Gaussian filter."""
    def __init__(self, sigma, truncate=4.0):
        """Convolve an image by a gaussian filter.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian kernel.
        truncate : float, optional
            Truncate the filter at this many standard deviations.
            Default is 4.0.

        Note
        ----
        Reproduces `scipy.ndimage.gaussian_filter` with high accuracy.

        """
        self.kernel = self.gaussian_kernel(sigma, truncate)

    def gaussian_kernel(self, sigma, truncate):
        # Determine the kernel pixel size (rounded up to an odd int)
        self.radius = int(jnp.ceil(2 * truncate * sigma)) // 2
        npix = self.radius * 2 + 1  # always at least 1

        # Return the identity if sigma is not a positive number
        if sigma <= 0:
            return jnp.array([[1.]])

        # Compute the kernel
        x, y = jnp.indices((npix, npix))  # pixel coordinates
        kernel = norm.pdf(jnp.hypot(x - self.radius, y - self.radius) / sigma)
        kernel /= kernel.sum()

        return kernel

    @partial(jit, static_argnums=(0,))
    def __call__(self, image):
        """Jit-compiled convolution an image by a gaussian filter.

        Parameters
        ----------
        image : array_like
            Image to filter.
        """
        # Convolve
        # pad_mode = ['constant', 'edge'][mode == 'nearest']
        # image_padded = jnp.pad(image, pad_width=radius, mode=pad_mode)
        image_padded = jnp.pad(image, pad_width=self.radius, mode='edge')
        return convolve2d(image_padded, self.kernel, mode='valid')


class WaveletTransform(object):
    """
    Class that handles wavelet transform using JAX, using the 'a trous' algorithm

    Parameters
    ----------
    nscales : number of scales in the decomposition
    self._type : supported types are 'starlet', 'battle-lemarie-1', 'battle-lemarie-3'

    """
    def __init__(self, nscales, wavelet_type='starlet', second_gen=False):
        self._n_scales = nscales
        self._second_gen = second_gen
        if wavelet_type == 'starlet':
            self._h = jnp.array([1, 4, 6, 4, 1]) / 16.
            self._fac = 2
        elif wavelet_type == 'battle-lemarie-1':
            self._h = jnp.array([-0.000122686, -0.000224296, 0.000511636, 
                        0.000923371, -0.002201945, -0.003883261, 0.009990599, 
                        0.016974805, -0.051945337, -0.06910102, 0.39729643, 
                        0.817645956, 0.39729643, -0.06910102, -0.051945337, 
                        0.016974805, 0.009990599, -0.003883261, -0.002201945,
                        0.000923371, 0.000511636, -0.000224296, -0.000122686])
            self._h /= 1.4140825479999999  # sum of coefficients above
            self._fac = 11
        elif wavelet_type == 'battle-lemarie-3':
            self._h = jnp.array([0.000146098, -0.000232304, -0.000285414, 
                          0.000462093, 0.000559952, -0.000927187, -0.001103748, 
                          0.00188212, 0.002186714, -0.003882426, -0.00435384, 
                          0.008201477, 0.008685294, -0.017982291, -0.017176331, 
                          0.042068328, 0.032080869, -0.110036987, -0.050201753, 
                          0.433923147, 0.766130398, 0.433923147, -0.050201753, 
                          -0.110036987, 0.032080869, 0.042068328, -0.017176331, 
                          -0.017982291, 0.008685294, 0.008201477, -0.00435384, 
                          -0.003882426, 0.002186714, 0.00188212, -0.001103748, 
                          -0.000927187, 0.000559952, 0.000462093, -0.000285414, 
                          -0.000232304, 0.000146098])
            self._h /= 1.4141580200000003  # sum of coefficients above
            self._fac = 20
        else:
            raise ValueError(f"'{wavelet_type}' starlet transform is not supported")

    @property
    def scale_norms(self):
        if not hasattr(self, '_norms'):
            npix_dirac = 2**(self._n_scales + 2)
            dirac = jnp.diag((jnp.arange(npix_dirac) == int(npix_dirac / 2)).astype(float))
            wt_dirac = self.decompose(dirac)
            self._norms = jnp.sqrt(jnp.sum(wt_dirac**2, axis=(1, 2,)))
        return self._norms

    @partial(jit, static_argnums=(0,))
    def decompose(self, image):
        """Decompose an image into the chosen wavelet basis"""
        if self._second_gen is True:
            return self._decompose_2nd_gen(image)
        else:
            return self._decompose_1st_gen(image)

    @partial(jit, static_argnums=(0,))
    def reconstruct(self, coeffs):
        """Reconstruct an image from wavelet decomposition coefficients"""
        if self._second_gen is True:
            return self._reconstruct_2nd_gen(coeffs)
        else:
            return self._reconstruct_1st_gen(coeffs)

    def _decompose_1st_gen(self, image):
        # Validate input
        assert self._n_scales >= 0, "nscales must be a non-negative integer"
        if self._n_scales == 0:
            return image

        # Preparations
        image = jnp.expand_dims(image, (0, 3))
        kernel = jnp.expand_dims(jnp.outer(self._h, self._h), (2, 3))
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        dn = conv_dimension_numbers(image.shape, kernel.shape, dimension_numbers)

        # Compute the first scale
        c0 = image
        padded = jnp.pad(c0, ((0, 0), (self._fac, self._fac), (self._fac, self._fac), (0, 0)), mode='edge')
        c1 = conv_general_dilated(padded, kernel,
                                  window_strides=(1, 1),
                                  padding='VALID',
                                  rhs_dilation=(1, 1),
                                  dimension_numbers=dn)
        w0 = (c0 - c1)[0,:,:,0]  # Wavelet coefficients
        result = jnp.expand_dims(w0, 0)
        cj = c1

        # Compute the remaining scales
        for ii in range(1, self._n_scales):
            b = self._fac**(ii + 1)  # padding pixels
            padded = jnp.pad(cj, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
            cj1 = conv_general_dilated(padded, kernel,
                                       window_strides=(1, 1),
                                       padding='VALID',
                                       rhs_dilation=(self._fac**ii, self._fac**ii),
                                       dimension_numbers=dn)
            # wavelet coefficients
            wj = (cj - cj1)[0,:,:,0]
            result = jnp.concatenate((result, jnp.expand_dims(wj, 0)), axis=0)
            cj = cj1

        # Append final coarse scale
        result = jnp.concatenate((result, cj[:,:,:,0]), axis=0)
        return result

    def _decompose_2nd_gen(self, image):
        # Validate input
        assert self._n_scales >= 0, "nscales must be a non-negative integer"
        if self._n_scales == 0:
            return image

        # Preparations
        image = jnp.expand_dims(image, (0, 3))
        kernel = jnp.expand_dims(jnp.outer(self._h, self._h), (2, 3))
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        dn = conv_dimension_numbers(image.shape, kernel.shape, dimension_numbers)

        # Compute the first scale
        c0 = image
        padded = jnp.pad(c0, ((0, 0), (self._fac, self._fac), (self._fac, self._fac), (0, 0)), mode='edge')
        c1 = conv_general_dilated(padded, kernel,
                                  window_strides=(1, 1),
                                  padding='VALID',
                                  rhs_dilation=(1, 1),
                                  dimension_numbers=dn)
        padded = jnp.pad(c1, ((0, 0), (self._fac, self._fac), (self._fac, self._fac), (0, 0)), mode='edge')
        c1p = conv_general_dilated(padded, kernel,
                                  window_strides=(1, 1),
                                  padding='VALID',
                                  rhs_dilation=(1, 1),
                                  dimension_numbers=dn)
        w0 = (c0 - c1p)[0,:,:,0]  # Wavelet coefficients
        result = jnp.expand_dims(w0, 0)
        cj = c1

        # Compute the remaining scales
        for ii in range(1, self._n_scales):
            b = self._fac**(ii + 1)  # padding pixels
            padded = jnp.pad(cj, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
            cj1 = conv_general_dilated(padded, kernel,
                                       window_strides=(1, 1),
                                       padding='VALID',
                                       rhs_dilation=(self._fac**ii, self._fac**ii),
                                       dimension_numbers=dn)
            padded = jnp.pad(cj1, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
            cj1p = conv_general_dilated(padded, kernel,
                                       window_strides=(1, 1),
                                       padding='VALID',
                                       rhs_dilation=(self._fac**ii, self._fac**ii),
                                       dimension_numbers=dn)
            # wavelet coefficients
            wj = (cj - cj1p)[0,:,:,0]
            result = jnp.concatenate((result, jnp.expand_dims(wj, 0)), axis=0)
            cj = cj1

        # Append final coarse scale
        result = jnp.concatenate((result, cj[:,:,:,0]), axis=0)
        return result

    def _reconstruct_1st_gen(self, coeffs):
        return jnp.sum(coeffs, axis=0)

    def _reconstruct_2nd_gen(self, coeffs):
        # Validate input
        assert coeffs.shape[0] == self._n_scales+1, "Wavelet coefficients are not consistent with number of scales"
        if self._n_scales == 0:
            return coeffs[0, :, :]

        # Preparations
        image_shape = (1, coeffs.shape[1], coeffs.shape[2], 1)
        kernel = jnp.expand_dims(jnp.outer(self._h, self._h), (2, 3))
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        dn = conv_dimension_numbers(image_shape, kernel.shape, dimension_numbers)

        # Start with the last scale 'J-1'
        cJ = jnp.expand_dims(coeffs[self._n_scales, :, :], (0, 3))
        b = self._fac**(self._n_scales-1 + 1)  # padding pixels
        padded = jnp.pad(cJ, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
        cJp = conv_general_dilated(padded, kernel,
                                    window_strides=(1, 1),
                                    padding='VALID',
                                    rhs_dilation=(self._fac**(self._n_scales-1), self._fac**(self._n_scales-1)),
                                    dimension_numbers=dn)
        wJ = jnp.expand_dims(coeffs[self._n_scales-1, :, :], (0, 3))
        cj = cJp + wJ

        # Compute the remaining scales
        for ii in range(self._n_scales-2, -1, -1):
            cj1 = cj

            b = self._fac**(ii + 1)  # padding pixels
            padded = jnp.pad(cj1, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
            cj1p = conv_general_dilated(padded, kernel,
                                       window_strides=(1, 1),
                                       padding='VALID',
                                       rhs_dilation=(self._fac**ii, self._fac**ii),
                                       dimension_numbers=dn)
            wj1 = jnp.expand_dims(coeffs[ii, :, :], (0, 3))
            cj = cj1p + wj1

        result = cj[0, :, :, 0]
        return result


class BilinearInterpolator(object):
    """Bilinear interpolation of a 2D field.

    Functionality is modelled after scipy.interpolate.RectBivariateSpline
    when `kx` and `ky` are both equal to 1. Results match the scipy version when
    interpolated values lie within the x and y domain (boundaries included).
    Returned values can be significantly different outside the natural domain,
    as the scipy version does not extrapolate. Evaluation of this jax version
    is MUCH SLOWER as well.

    """
    def __init__(self, x, y, z, allow_extrapolation=True):
        self.z = jnp.array(z)  # z
        if np.all(np.diff(x) >= 0):  # check if sorted in increasing order
            self.x = jnp.array(x)
        else:
            self.x = jnp.array(np.sort(x))
            self.z = jnp.flip(self.z, axis=0)
        if np.all(np.diff(y) >= 0):  # check if sorted in increasing order
            self.y = jnp.array(y)
        else:
            self.y = jnp.array(np.sort(y))
            self.z = jnp.flip(self.z, axis=1)
        self._extrapol_bool = allow_extrapolation

    def __call__(self, x, y, dx=0, dy=0):
        """Vectorized evaluation of the interpolation or its derivatives.

        Parameters
        ----------
        x, y : array_like
            Position(s) at which to evaluate the interpolation.
        dx, dy : int, either 0 or 1
            If 1, return the first partial derivative of the interpolation
            with respect to that coordinate. Only one of (dx, dy) should be
            nonzero at a time.

        """
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)

        error_msg_type = "dx and dy must be integers"
        error_msg_value = "dx and dy must only be either 0 or 1"
        assert isinstance(dx, int) and isinstance(dy, int), error_msg_type
        assert dx in (0, 1) and dy in (0, 1), error_msg_value
        if dx == 1: dy = 0

        return vmap(self._evaluate, in_axes=(0, 0, None, None))(x, y, dx, dy)

    # @partial(jit, static_argnums=(0,))
    def _compute_coeffs(self, x, y):
        # Find the pixel that the point (x, y) falls in
        # x_ind = jnp.digitize(x, self.x_padded) - 1
        # y_ind = jnp.digitize(y, self.y_padded) - 1
        x_ind = jnp.searchsorted(self.x, x, side='right') - 1
        x_ind = jnp.clip(x_ind, a_min=0, a_max=(len(self.x) - 2))
        y_ind = jnp.searchsorted(self.y, y, side='right') - 1
        y_ind = jnp.clip(y_ind, a_min=0, a_max=(len(self.y) - 2))

        # Determine the coordinates and dimensions of this pixel
        x1 = self.x[x_ind]
        x2 = self.x[x_ind + 1]
        y1 = self.y[y_ind]
        y2 = self.y[y_ind + 1]
        area = (x2 - x1) * (y2 - y1)

        # Compute function values at the four corners
        # Edge padding is implicitly constant
        v11 = self.z[x_ind, y_ind]
        v12 = self.z[x_ind, y_ind + 1]
        v21 = self.z[x_ind + 1, y_ind]
        v22 = self.z[x_ind + 1, y_ind + 1]

        # Compute the coefficients
        a0_ = v11 * x2 * y2 - v12 * x2 * y1 - v21 * x1 * y2 + v22 * x1 * y1
        a1_ = -v11 * y2 + v12 * y1 + v21 * y2 - v22 * y1
        a2_ = -v11 * x2 + v12 * x2 + v21 * x1 - v22 * x1
        a3_ = v11 - v12 - v21 + v22

        return a0_ / area, a1_ / area, a2_ / area, a3_ / area

    def _evaluate(self, x, y, dx=0, dy=0):
        """Single-point evaluation of the interpolation."""
        a0, a1, a2, a3 = self._compute_coeffs(x, y)
        if (dx, dy) == (0, 0):
            result = a0 + a1 * x + a2 * y + a3 * x * y
        elif (dx, dy) == (1, 0):
            result = a1 + a3 * y
        else:
            result = a2 + a3 * x
        # if extrapolation is not allowed, then we mask out values outside the original bounding box
        result = lax.cond(self._extrapol_bool, 
                          lambda _: result, 
                          lambda _: result * (x >= self.x[0]) * (x <= self.x[-1]) * (y >= self.y[0]) * (y <= self.y[-1]), 
                          operand=None)
        return result


class BicubicInterpolator(object):
    """Bicubic interpolation of a 2D field.

    Functionality is modelled after scipy.interpolate.RectBivariateSpline
    when `kx` and `ky` are both equal to 3.

    """
    def __init__(self, x, y, z, zx=None, zy=None, zxy=None, allow_extrapolation=True):
        self.z = jnp.array(z)
        if np.all(np.diff(x) >= 0):  # check if sorted in increasing order
            self.x = jnp.array(x)
        else:
            self.x = jnp.array(np.sort(x))
            self.z = jnp.flip(self.z, axis=1)
        if np.all(np.diff(y) >= 0):  # check if sorted in increasing order
            self.y = jnp.array(y)
        else:
            self.y = jnp.array(np.sort(y))
            self.z = jnp.flip(self.z, axis=0)

        # Assume uniform coordinate spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Compute approximate partial derivatives if not provided
        if zx is None:
            self.zx = jnp.gradient(z, axis=0) / self.dx
        else:
            self.zx = zy
        if zy is None:
            self.zy = jnp.gradient(z, axis=1) / self.dy
        else:
            self.zy = zx
        if zxy is None:
            self.zxy = jnp.gradient(self.zx, axis=1) / self.dy
        else:
            self.zxy = zxy

        # Prepare coefficients for function evaluations
        self._A = jnp.array([[1., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [-3., 3., -2., -1.],
                            [2., -2., 1., 1.]])
        self._B = jnp.array([[1., 0., -3., 2.],
                            [0., 0., 3., -2.],
                            [0., 1., -2., 1.],
                            [0., 0., -1., 1.]])
        row0 = [self.z[:-1,:-1], self.z[:-1,1:], self.dy * self.zy[:-1,:-1], self.dy * self.zy[:-1,1:]]
        row1 = [self.z[1:,:-1], self.z[1:,1:], self.dy * self.zy[1:,:-1], self.dy * self.zy[1:,1:]]
        row2 = self.dx * jnp.array([self.zx[:-1,:-1], self.zx[:-1,1:],
                                   self.dy * self.zxy[:-1,:-1], self.dy * self.zxy[:-1,1:]])
        row3 = self.dx * jnp.array([self.zx[1:,:-1], self.zx[1:,1:],
                                   self.dy * self.zxy[1:,:-1], self.dy * self.zxy[1:,1:]])
        self._m = jnp.array([row0, row1, row2, row3])

        self._m = jnp.transpose(self._m, axes=(2, 3, 0, 1))

        self._extrapol_bool = allow_extrapolation

    def __call__(self, x, y, dx=0, dy=0):
        """Vectorized evaluation of the interpolation or its derivatives.

        Parameters
        ----------
        x, y : array_like
            Position(s) at which to evaluate the interpolation.
        dx, dy : int, either 0, 1, or 2
            Return the nth partial derivative of the interpolation
            with respect to the specified coordinate. Only one of (dx, dy)
            should be nonzero at a time.

        """
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        if x.ndim == 1:
            vmap_call = vmap(self._evaluate, in_axes=(0, 0, None, None))
        elif x.ndim == 2:
            vmap_call = vmap(vmap(self._evaluate, in_axes=(0, 0, None, None)),
                             in_axes=(0, 0, None, None))
        return vmap_call(x, y, dx, dy)

    def _evaluate(self, x, y, dx=0, dy=0):
        """Evaluate the interpolation at a single point."""
        # Determine which pixel (i, j) the point (x, y) falls in
        i = jnp.maximum(0, jnp.searchsorted(self.x, x) - 1)
        j = jnp.maximum(0, jnp.searchsorted(self.y, y) - 1)

        # Rescale coordinates into (0, 1)
        u = (x - self.x[i]) / self.dx
        v = (y - self.y[j]) / self.dy

        # Compute interpolation coefficients
        a = jnp.dot(self._A, jnp.dot(self._m[i, j], self._B))

        if dx == 0:
            uu = jnp.asarray([1., u, u**2, u**3])
        if dx == 1:
            uu = jnp.asarray([0., 1., 2. * u, 3. * u**2]) / self.dx
        if dx == 2:
            uu = jnp.asarray([0., 0., 2., 6. * u]) / self.dx**2
        if dy == 0:
            vv = jnp.asarray([1., v, v**2, v**3])
        if dy == 1:
            vv = jnp.asarray([0., 1., 2. * v, 3. * v**2]) / self.dy
        if dy == 2:
            vv = jnp.asarray([0., 0., 2., 6. * v]) / self.dy**2
        result = jnp.dot(uu, jnp.dot(a, vv))

        # if extrapolation is not allowed, then we mask out values outside the original bounding box
        result = lax.cond(self._extrapol_bool, 
                          lambda _: result, 
                          lambda _: result * (x >= self.x[0]) * (x <= self.x[-1]) * (y >= self.y[0]) * (y <= self.y[-1]), 
                          operand=None)
        return result
        