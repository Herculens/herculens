import functools
import jax.numpy as np
from jax import jit, vmap
from jax.scipy.special import gammaln
from jax.scipy.signal import convolve2d
from jax.scipy.stats import norm
from jax.lax import conv_general_dilated, conv_dimension_numbers


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
        sign_condition = (x > 0) | (np.ceil(x) % 2 != 0) | (x % 2 == 0)
        sign = 2 * np.asarray(sign_condition, dtype=float) - 1
        return sign * np.exp(gammaln(x))


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
        self.radius = int(np.ceil(2 * truncate * sigma)) // 2
        npix = self.radius * 2 + 1  # always at least 1

        # Return the identity if sigma is not a positive number
        if sigma <= 0:
            return np.array([[1.]])

        # Compute the kernel
        x, y = np.indices((npix, npix))  # pixel coordinates
        kernel = norm.pdf(np.hypot(x - self.radius, y - self.radius) / sigma)
        kernel /= kernel.sum()

        return kernel

    @functools.partial(jit, static_argnums=(0,))
    def __call__(self, image):
        """Jit-compiled convolution an image by a gaussian filter.

        Parameters
        ----------
        image : array_like
            Image to filter.
        """
        # Convolve
        # pad_mode = ['constant', 'edge'][mode == 'nearest']
        # image_padded = np.pad(image, pad_width=radius, mode=pad_mode)
        image_padded = np.pad(image, pad_width=self.radius, mode='edge')
        return convolve2d(image_padded, self.kernel, mode='valid')


@functools.partial(jit, static_argnums=(1,))
def starlet2d(image, nscales):
    """JAX-based starlet transform of an image.

    Parameters
    ----------
    image : TODO
    nscales : TODO

    Returns
    -------
    TODO

    """
    # Validate input
    assert nscales >= 0, "nscales must be a non-negative integer"
    if nscales == 0:
        return image

    # Preparations
    shape = image.shape
    image = np.expand_dims(image, (0, 3))
    h = np.array([1, 4, 6, 4, 1]) / 16.
    kernel = np.expand_dims(np.outer(h, h), (2, 3))
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dn = conv_dimension_numbers(image.shape, kernel.shape, dimension_numbers)

    # Compute the first scale
    c0 = image
    padded = np.pad(c0, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='edge')
    c1 = conv_general_dilated(padded, kernel,
                              window_strides=(1, 1),
                              padding='VALID',
                              rhs_dilation=(1, 1),
                              dimension_numbers=dn)
    w0 = (c0 - c1)[0,:,:,0]  # Wavelet coefficients
    result = np.expand_dims(w0, 0)
    cj = c1

    # Compute the remaining scales
    for ii in range(1, nscales):
        b = 2**(ii + 1)  # padding pixels
        padded = np.pad(cj, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
        cj1 = conv_general_dilated(padded, kernel,
                                   window_strides=(1, 1),
                                   padding='VALID',
                                   rhs_dilation=(2**ii, 2**ii),
                                   dimension_numbers=dn)
        # wavelet coefficients
        wj = (cj - cj1)[0,:,:,0]
        result = np.concatenate((result, np.expand_dims(wj, 0)), axis=0)
        cj = cj1

    # Append final coarse scale
    result = np.concatenate((result, cj[:,:,:,0]), axis=0)

    return result

@functools.partial(jit, static_argnums=(1,))
def battlelemarie2d(image, nscales):
    """JAX-based starlet transform of an image.

    Parameters
    ----------
    image : TODO
    nscales : TODO

    Returns
    -------
    TODO

    """
    # Validate input
    assert nscales >= 0, "nscales must be a non-negative integer"
    if nscales == 0:
        return image

    # Preparations
    shape = image.shape
    image = np.expand_dims(image, (0, 3))
    h = np.array([-0.000122686, -0.000224296, 0.000511636, 
                    0.000923371, -0.002201945, -0.003883261, 0.009990599, 
                    0.016974805, -0.051945337, -0.06910102, 0.39729643, 
                    0.817645956, 0.39729643, -0.06910102, -0.051945337, 
                    0.016974805, 0.009990599, -0.003883261, -0.002201945,
                    0.000923371, 0.000511636, -0.000224296, -0.000122686])
    h /= h.sum()
    kernel = np.expand_dims(np.outer(h, h), (2, 3))
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dn = conv_dimension_numbers(image.shape, kernel.shape, dimension_numbers)

    # Compute the first scale
    c0 = image
    padded = np.pad(c0, ((0, 0), (11, 11), (11, 11), (0, 0)), mode='edge')
    c1 = conv_general_dilated(padded, kernel,
                              window_strides=(1, 1),
                              padding='VALID',
                              rhs_dilation=(1, 1),
                              dimension_numbers=dn)
    w0 = (c0 - c1)[0,:,:,0]  # Wavelet coefficients
    result = np.expand_dims(w0, 0)
    cj = c1

    # Compute the remaining scales
    for ii in range(1, nscales):
        b = 11**(ii + 1)  # padding pixels
        padded = np.pad(cj, ((0, 0), (b, b), (b, b), (0, 0)), mode='edge')
        cj1 = conv_general_dilated(padded, kernel,
                                   window_strides=(1, 1),
                                   padding='VALID',
                                   rhs_dilation=(11**ii, 11**ii),
                                   dimension_numbers=dn)
        # wavelet coefficients
        wj = (cj - cj1)[0,:,:,0]
        result = np.concatenate((result, np.expand_dims(wj, 0)), axis=0)
        cj = cj1

    # Append final coarse scale
    result = np.concatenate((result, cj[:,:,:,0]), axis=0)

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
    def __init__(self, x_coords, y_coords, z):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z = z

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
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        error_msg_type = "dx and dy must be integers"
        error_msg_value = "dx and dy must only be either 0 or 1"
        assert isinstance(dx, int) and isinstance(dy, int), error_msg_type
        assert dx in (0, 1) and dy in (0, 1), error_msg_value
        if dx == 1: dy = 0

        return vmap(self._evaluate, in_axes=(0, 0, None, None))(x, y, dx, dy)

    # @functools.partial(jit, static_argnums=(0,))
    def _compute_coeffs(self, x, y):
        # Find the pixel that the point (x, y) falls in
        # x_ind = np.digitize(x, self.x_padded) - 1
        # y_ind = np.digitize(y, self.y_padded) - 1
        x_ind = np.searchsorted(self.x_coords, x, side='right') - 1
        x_ind = np.clip(x_ind, a_min=0, a_max=(len(self.x_coords) - 2))
        y_ind = np.searchsorted(self.y_coords, y, side='right') - 1
        y_ind = np.clip(y_ind, a_min=0, a_max=(len(self.y_coords) - 2))

        # Determine the coordinates and dimensions of this pixel
        x1 = self.x_coords[x_ind]
        x2 = self.x_coords[x_ind + 1]
        y1 = self.y_coords[y_ind]
        y2 = self.y_coords[y_ind + 1]
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

        return result


class BicubicInterpolator(object):
    """Bicubic interpolation of a 2D field.

    Functionality is modelled after scipy.interpolate.RectBivariateSpline
    when `kx` and `ky` are both equal to 3.

    """
    def __init__(self, x, y, z, zx=None, zy=None, zxy=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        # Assume uniform coordinate spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Compute approximate partial derivatives if not provided
        if zx is None:
            self.zx = np.gradient(z, axis=0) / self.dx
        else:
            self.zx = zy
        if zy is None:
            self.zy = np.gradient(z, axis=1) / self.dy
        else:
            self.zy = zx
        if zxy is None:
            self.zxy = np.gradient(self.zx, axis=1) / self.dy
        else:
            self.zxy = zxy

        # Prepare coefficients for function evaluations
        self._A = np.array([[1., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [-3., 3., -2., -1.],
                            [2., -2., 1., 1.]])
        self._B = np.array([[1., 0., -3., 2.],
                            [0., 0., 3., -2.],
                            [0., 1., -2., 1.],
                            [0., 0., -1., 1.]])
        row0 = [self.z[:-1,:-1], self.z[:-1,1:], self.dy * self.zy[:-1,:-1], self.dy * self.zy[:-1,1:]]
        row1 = [self.z[1:,:-1], self.z[1:,1:], self.dy * self.zy[1:,:-1], self.dy * self.zy[1:,1:]]
        row2 = self.dx * np.array([self.zx[:-1,:-1], self.zx[:-1,1:],
                                   self.dy * self.zxy[:-1,:-1], self.dy * self.zxy[:-1,1:]])
        row3 = self.dx * np.array([self.zx[1:,:-1], self.zx[1:,1:],
                                   self.dy * self.zxy[1:,:-1], self.dy * self.zxy[1:,1:]])
        self._m = np.array([row0, row1, row2, row3])

        self._m = np.transpose(self._m, axes=(2, 3, 0, 1))

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
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.ndim == 1:
            vmap_call = vmap(self._evaluate, in_axes=(0, 0, None, None))
        elif x.ndim == 2:
            vmap_call = vmap(vmap(self._evaluate, in_axes=(0, 0, None, None)),
                             in_axes=(0, 0, None, None))
        return vmap_call(x, y, dx, dy)

    def _evaluate(self, x, y, dx=0, dy=0):
        """Evaluate the interpolation at a single point."""
        # Determine which pixel (i, j) the point (x, y) falls in
        i = np.maximum(0, np.searchsorted(self.x, x) - 1)
        j = np.maximum(0, np.searchsorted(self.y, y) - 1)

        # Rescale coordinates into (0, 1)
        u = (x - self.x[i]) / self.dx
        v = (y - self.y[j]) / self.dy

        # Compute interpolation coefficients
        a = np.dot(self._A, np.dot(self._m[i, j], self._B))

        if dx == 0:
            uu = np.asarray([1., u, u**2, u**3])
        if dx == 1:
            uu = np.asarray([0., 1., 2. * u, 3. * u**2]) / self.dx
        if dx == 2:
            uu = np.asarray([0., 0., 2., 6. * u]) / self.dx**2
        if dy == 0:
            vv = np.asarray([1., v, v**2, v**3])
        if dy == 1:
            vv = np.asarray([0., 1., 2. * v, 3. * v**2]) / self.dy
        if dy == 2:
            vv = np.asarray([0., 0., 2., 6. * v]) / self.dy**2

        return np.dot(uu, np.dot(a, vv))
