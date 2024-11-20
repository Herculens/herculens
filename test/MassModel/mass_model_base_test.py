from contextlib import AbstractContextManager
from typing import Any
import unittest

import herculens as hcl
from herculens.MassModel.mass_model_base import MassModelBase

class TestMassModelBase(unittest.TestCase):
    def setUp(self):
        # Create a sample profile list
        self.profile_list_base_strings = ['EPL', 'SHEAR']
        self.profile_list_base_instances = [hcl.EPL(), hcl.Shear()]
        self.profile_list_pixelated = [hcl.PixelatedPotential()]
        self.kwargs_pixelated = {'num_pixels': 10}

    def test_constructor(self):
        # Test the constructor with profile strings
        mass_model_base = MassModelBase(
            self.profile_list_base_strings,
        )
        self.assertEqual(mass_model_base._num_func, 2)
        self.assertEqual(mass_model_base._model_list, self.profile_list_base_strings)

        # Test the constructor with profile instances
        mass_model_base = MassModelBase(
            self.profile_list_base_instances,
        )
        self.assertEqual(mass_model_base._num_func, 2)
        self.assertEqual(mass_model_base._model_list, self.profile_list_base_instances)

        # Test the constructor with pixelated profile
        mass_model_base = MassModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertEqual(mass_model_base._num_func, 1)

    # Test incorrect input types
    def test_incorrect_input_types(self):
        with self.assertRaises(TypeError):
            mass_model_base = MassModelBase(
                'EPL',
            )
        with self.assertRaises(TypeError):
            mass_model_base = MassModelBase(
                1,
                self.kwargs_pixelated,
            )
        with self.assertRaises(TypeError):
            mass_model_base = MassModelBase(
                hcl.EPL(),
                self.kwargs_pixelated,
            )

    def test_is_mass_profile_class(self):
        self.assertTrue(MassModelBase.is_mass_profile_class(hcl.EPL))
        self.assertFalse(MassModelBase.is_mass_profile_class('INVALID_PROFILE'))

    # Test the get_class_from_string method
    def test_get_class_from_string_EPL(self):
        profile_class = MassModelBase.get_class_from_string('EPL', no_complex_numbers=True)
        self.assertTrue(isinstance(profile_class, hcl.EPL))

    def test_get_class_from_string_PIXELATED(self):
        profile_class = MassModelBase.get_class_from_string('PIXELATED', pixel_derivative_type= 'bicubic', pixel_interpol='autodiff')
        self.assertTrue(isinstance(profile_class, hcl.PixelatedPotential))

    def test_get_class_from_string_PIXELATED_FIXED(self):
        profile_class = MassModelBase.get_class_from_string('PIXELATED_FIXED', kwargs_pixel_grid_fixed={'func_pixel_grid': None})
        self.assertTrue(isinstance(profile_class, hcl.PixelatedFixed))

    def test_get_class_from_string_invalid_profile(self):
        with self.assertRaises(ValueError):
            MassModelBase.get_class_from_string('INVALID_PROFILE')

    # Test if model has pixels
    def test_has_pixels(self):
        mass_model_base = MassModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertTrue(mass_model_base.has_pixels)

        mass_model_base = MassModelBase(
            self.profile_list_base_instances,
        )
        self.assertFalse(mass_model_base.has_pixels)

    # Test features valid with a pixelated profile
    def test_pixelated_profile_index(self):
        # when no pixelated profile is present
        mass_model_base = MassModelBase(
            self.profile_list_base_instances,
        )
        self.assertIsNone(mass_model_base.pixelated_index)
        
        # when a pixelated profile is present
        mass_model_base = MassModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        self.assertEqual(mass_model_base.pixelated_index, 0)
        # test we can access the pixel grid settings
        self.assertEqual(mass_model_base.pixel_grid_settings, self.kwargs_pixelated)

        # let's set the grid instance
        grid_class = hcl.PixelGrid(nx=10, ny=10)
        mass_model_base = MassModelBase(
            self.profile_list_pixelated,
            kwargs_pixelated=self.kwargs_pixelated,
        )
        mass_model_base.set_pixel_grid(grid_class.create_model_grid(**mass_model_base.pixel_grid_settings))
        self.assertIsNotNone(mass_model_base.pixel_grid)

        # test we can access the coordinates
        self.assertIsNotNone(mass_model_base.pixelated_coordinates)
        # test we can access the shape of the pixel grid
        self.assertIsNotNone(mass_model_base.pixelated_shape)

    def test_bool_list(self):
        mass_model_base = MassModelBase(
            self.profile_list_base_instances,
        )
        self.assertEqual(mass_model_base._bool_list(0), [True, False])
        self.assertEqual(mass_model_base._bool_list(1), [False, True])
