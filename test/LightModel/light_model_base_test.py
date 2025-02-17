from contextlib import AbstractContextManager
from typing import Any
import unittest

import herculens as hcl
from herculens.LightModel.light_model_base import LightModelBase

class TestLightModelBase(unittest.TestCase):
    def setUp(self):
        self.grid_class = hcl.PixelGrid(nx=10, ny=10)
        # Create a sample profile list
        self.profile_list_base_strings = ['SERSIC_ELLIPSE', 'GAUSSIAN_ELLIPSE']
        self.profile_list_base_instances = [hcl.SersicElliptic(), hcl.GaussianEllipseLight()]
        self.profile_list_pixelated = [hcl.PixelatedLight()]
        self.profile_list_pixelated_adapt = [hcl.PixelatedLight(adaptive_grid=True)]
        self.kwargs_pixelated = {'num_pixels': 10}

    def test_constructor(self):
        # Test the constructor with profile strings
        light_model_base = LightModelBase(
            self.profile_list_base_strings,
        )
        self.assertEqual(light_model_base._num_func, 2)
        self.assertEqual(light_model_base._model_list, self.profile_list_base_strings)

        # Test the constructor with profile instances
        light_model_base = LightModelBase(
            self.profile_list_base_instances,
        )
        self.assertEqual(light_model_base._num_func, 2)
        self.assertEqual(light_model_base._model_list, self.profile_list_base_instances)

        # Test the constructor with pixelated profile
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertEqual(light_model_base._num_func, 1)

    # Test incorrect input types
    def test_incorrect_input_types(self):
        with self.assertRaises(TypeError):
            light_model_base = LightModelBase(
                'SERSIC_ELLIPSE',
            )
        with self.assertRaises(TypeError):
            light_model_base = LightModelBase(
                1,
                self.kwargs_pixelated,
            )
        with self.assertRaises(TypeError):
            light_model_base = LightModelBase(
                hcl.SersicElliptic(),
                self.kwargs_pixelated,
            )

    def test_is_light_profile_class(self):
        self.assertTrue(LightModelBase.is_light_profile_class(hcl.SersicElliptic))
        self.assertFalse(LightModelBase.is_light_profile_class('INVALID_PROFILE'))

    # Test the get_class_from_string method
    def test_get_class_from_string_SERSIC(self):
        profile_class = LightModelBase.get_class_from_string('SERSIC_ELLIPSE')
        self.assertTrue(isinstance(profile_class, hcl.SersicElliptic))
        profile_class = LightModelBase.get_class_from_string('SERSIC')
        self.assertTrue(isinstance(profile_class, hcl.Sersic))
        profile_class = LightModelBase.get_class_from_string('SERSIC_SUPERELLIPSE')
        self.assertTrue(isinstance(profile_class, hcl.SersicElliptic))
        # profile_class = LightModelBase.get_class_from_string('SHAPELETS')  # requires gigalens
        # self.assertTrue(isinstance(profile_class, hcl.Shapelets))

    def test_get_class_from_string_PIXELATED(self):
        profile_class = LightModelBase.get_class_from_string('PIXELATED')
        self.assertTrue(isinstance(profile_class, hcl.PixelatedLight))

    def test_get_class_from_string_invalid_profile(self):
        with self.assertRaises(ValueError):
            LightModelBase.get_class_from_string('INVALID_PROFILE')

    # Test if model has pixels
    def test_has_pixels(self):
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertTrue(light_model_base.has_pixels)

        light_model_base = LightModelBase(
            self.profile_list_base_instances,
        )
        self.assertFalse(light_model_base.has_pixels)

    # Test features valid with a pixelated profile
    def test_pixelated_profile_index(self):
        # when no pixelated profile is present
        light_model_base = LightModelBase(
            self.profile_list_base_instances,
        )
        self.assertIsNone(light_model_base.pixelated_index)
        
        # when a pixelated profile is present
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertEqual(light_model_base.pixelated_index, 0)
        # test we can access the pixel grid settings
        self.assertEqual(light_model_base.pixel_grid_settings, self.kwargs_pixelated)

        # let's set the grid instance
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        light_model_base.set_pixel_grid(
            self.grid_class.create_model_grid(**light_model_base.pixel_grid_settings),
            data_pixel_area=self.grid_class.pixel_area
        )
        self.assertIsNotNone(light_model_base.pixel_grid)

        # test we can access the coordinates
        self.assertIsNotNone(light_model_base.pixelated_coordinates)
        # test we can access the shape of the pixel grid
        self.assertIsNotNone(light_model_base.pixelated_shape)

    def test_bool_list(self):
        light_model_base = LightModelBase(
            self.profile_list_base_instances,
        )
        self.assertEqual(light_model_base._bool_list(0), [True, False])
        self.assertEqual(light_model_base._bool_list(1), [False, True])

    def test_num_amplitudes_list(self):
        light_model_base = LightModelBase(
            self.profile_list_base_instances,
        )
        self.assertEqual(light_model_base.num_amplitudes_list, [1, 1])

    def test_pixel_is_adaptive(self):
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertFalse(light_model_base.pixel_is_adaptive)
        light_model_base = LightModelBase(
            self.profile_list_pixelated_adapt,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertTrue(light_model_base.pixel_is_adaptive)

    def test_pixel_grid(self):
        light_model_base = LightModelBase(
            self.profile_list_base_instances
        )
        self.assertIsNone(light_model_base.pixel_grid)
        light_model_base = LightModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        light_model_base.set_pixel_grid(
            self.grid_class.create_model_grid(**light_model_base.pixel_grid_settings),
            data_pixel_area=self.grid_class.pixel_area
        )
        self.assertTrue(isinstance(light_model_base.pixel_grid, hcl.PixelGrid))
