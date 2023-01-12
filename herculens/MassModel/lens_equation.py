__author__ = 'austinpeel'

import jax.numpy as jnp
from jax import jit

class LensEquationSolver(object):
    def __init__(self, mass_model):
        """Solver for the multiple lensed image positions of a source point.

        Parameters
        ----------
        mass_model : `herculens.MassModel.mass_model.MassModel`
            Instance of MassModel defining the lens potential and deflection
            angles.

        """
        self.mass_model = mass_model

        # self.shoot_rays = jax.jit(self._shoot_rays)
        self.shoot_rays = self._shoot_rays

    def _shoot_rays(self, x, y, kwargs_lens):
        return self.mass_model.ray_shooting(x, y, kwargs_lens)

    def triangulate(self, pixel_grid):
        """Triangulate a pixel grid.

        Parameters
        ----------
        pixel_grid : `herculens.Coordinates.pixel_grid.PixelGrid`
            Instance of PixelGrid defining the initial ray-shooting positions.
            This can in general be at lower resolution than the data for faster
            computations.

        Returns
        -------
        out : jax array of shape (2 * N, 3, 2)
            Vertices of 2 * N triangles, where each pixel is divided in half
            along the diagonal.

        """
        xgrid, ygrid = pixel_grid.pixel_coordinates
        xcoords, ycoords = xgrid.flatten(), ygrid.flatten()
        delta = 0.5 * pixel_grid.pixel_width

        # Coordinates of the four corners of each pixel
        x_LL, y_LL = xcoords - delta, ycoords - delta
        x_LR, y_LR = xcoords + delta, ycoords - delta
        x_UL, y_UL = xcoords - delta, ycoords + delta
        x_UR, y_UR = xcoords + delta, ycoords + delta
        t1 = jnp.array([[x_LL, y_LL], [x_LR, y_LR], [x_UL, y_UL]]).transpose(2, 0, 1)
        t2 = jnp.array([[x_LR, y_LR], [x_UR, y_UR], [x_UL, y_UL]]).transpose(2, 0, 1)

        # Interleave arrays so that the two triangles corresponding to a pixel are adjacent
        triangles = jnp.column_stack((t1, t2))

        return triangles.reshape(2 * pixel_grid.num_pixel, 3, 2)

    def source_plane_triangles(self, image_triangles, kwargs_lens):
        """Source plane triangles corresponding to image plane counterparts.

        Parameters
        ----------
        image_triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.
        kwargs_lens : dict
            Parameters defining the mass model.

        """
        # Unpack into (x, y) triangle vertex arrays
        n = len(image_triangles)
        theta1, theta2 = image_triangles.transpose((2, 0, 1)).reshape((2, 3 * n))

        # Shoot vertices to the source plane
        beta1, beta2 = self.shoot_rays(theta1, theta2, kwargs_lens)

        # Repack into an array of triangle vertices
        return jnp.vstack([beta1, beta2]).reshape((2, n, 3)).transpose((1, 2, 0))

    def indices_containing_point(self, triangles, point):
        """Determine whether a point lies within a triangle.

        Points lying along a triangle's edges are not considered
        to be within in it.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles.
        point : jax array of shape (2,)
            Point to test.

        Returns
        -------
        bool : jax array of shape (N,)
            Whether each triangle contains the input point.

        """
        # Distances between each vertex and the input point
        delta = triangles - jnp.atleast_1d(point)

        sign1 = jnp.sign(jnp.cross(delta[:, 0], delta[:, 1]))
        sign2 = jnp.sign(jnp.cross(delta[:, 1], delta[:, 2]))
        sign3 = jnp.sign(jnp.cross(delta[:, 2], delta[:, 0]))
        return jnp.abs(sign1 + sign2 + sign3) == 3

    def scale_triangles(self, triangles, scale_factor):
        """Scale triangles about their centroids.

        scale_factor : float
            Factor by which each triangle's area is scaled.

        """
        c = self.centroids(triangles)
        c = jnp.repeat(jnp.expand_dims(c, 1), repeats=3, axis=1)
        return c + scale_factor**0.5 * (triangles - c)

    def subdivide_triangles(self, triangles, niter=1):
        """Divide a set of triangles into 4 congruent triangles.

        Parameters
        ----------
        triangles : TODO
            ...
        niter : int
            ...

        """
        v1, v2, v3 = triangles.transpose(1, 0, 2)
        v4 = 0.5 * (v1 + v2)
        v5 = 0.5 * (v2 + v3)
        v6 = 0.5 * (v3 + v1)
        t1 = [v1, v4, v6]
        t2 = [v4, v2, v5]
        t3 = [v6, v4, v5]
        t4 = [v6, v5, v3]
        subtriangles = jnp.column_stack((t1, t2, t3, t4)).transpose(1, 0, 2)

        for k in range(1, niter):
            v1, v2, v3 = subtriangles.transpose(1, 0, 2)
            v4 = 0.5 * (v1 + v2)
            v5 = 0.5 * (v2 + v3)
            v6 = 0.5 * (v3 + v1)
            t1 = [v1, v4, v6]
            t2 = [v4, v2, v5]
            t3 = [v6, v4, v5]
            t4 = [v6, v5, v3]
            subtriangles = jnp.column_stack((t1, t2, t3, t4)).transpose(1, 0, 2)

        return subtriangles.reshape(4**niter * len(triangles), 3, 2)

    def centroids(self, triangles):
        """The centroid positions of triangles."""
        return triangles.sum(axis=1) / 3.

    def signed_areas(self, triangles):
        """The signed area of triangles."""
        side1 = triangles[:, 1] - triangles[:, 0]
        side2 = triangles[:, 2] - triangles[:, 1]
        return 0.5 * jnp.cross(side1, side2)

    def solve(self, image_plane, beta, kwargs_lens, niter=5, scale_factor=2, nsubdivisions=1):
        """Solve the lens equation.

        Parameters
        ----------
        image_plane : `herculens.Coordinates.pixel_grid.PixelGrid`
            Instance of PixelGrid defining the initial ray-shooting positions.
            This can in general be at lower resolution than the data.
        beta : jax array of shape (2,)
            Position of a point source in the source plane.
        kwargs_lens : dict
            Parameters defining the mass model.
        niter : int
            ...
        scale_factor : float
            ...
        nsubdivisions : int
            ...

        """
        # Triangulate the image plane
        img_triangles = self.triangulate(image_plane)

        # Compute source plane images of the image plane triangles
        src_triangles = self.source_plane_triangles(img_triangles, kwargs_lens)

        # Retain only those image plane triangles whose source image contains the point source
        inds = self.indices_containing_point(src_triangles, beta)
        img_selection = img_triangles[jnp.where(inds, size=5)]

        for k in range(niter):
            # Scale up triangles
            img_selection = self.scale_triangles(img_selection, scale_factor)

            # Subdivide each triangle into 4
            img_selection = self.subdivide_triangles(img_selection, nsubdivisions)

            # Ray-shoot subtriangles to the source plane
            src_selection = self.source_plane_triangles(img_selection, kwargs_lens)

            # Select corresponding image plane triangles containing the source point
            inds = self.indices_containing_point(src_selection, beta)
            img_selection = img_selection[jnp.where(inds, size=5)]

        # Ray-shoot the final image plane triangles to the source plane
        src_selection = self.source_plane_triangles(img_selection, kwargs_lens)

        return self.centroids(img_selection), self.centroids(src_selection)
