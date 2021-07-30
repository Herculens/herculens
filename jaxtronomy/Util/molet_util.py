import os
import re
import json
import numpy as np
from astropy.io import fits
from jaxtronomy.Coordinates.pixel_grid import PixelGrid


def read_json(input_path):
    with open(input_path,'r') as f:
        input_str = f.read()
        input_str = re.sub(re.compile("/\*.*?\*/",re.DOTALL),"",input_str)
        input_str = re.sub(re.compile("//.*?\n" ),"",input_str)
        json_in   = json.loads(input_str)
    return json_in

def get_pixel_grid_class(molet_dir, input_file, instrument_name, intrument_index=0):
    """utility method for getting the PixelGrid class from MOLET settings"""
    # load the settings
    input_settings = read_json(os.path.join(molet_dir, input_file))
    instru_settings = read_json(os.path.join(molet_dir, 'instrument_modules', instrument_name, 'specs.json'))
    # sanity check for instrument consistency
    assert input_settings['instruments'][intrument_index]['name'] == instrument_name
    # load required settings values
    fov_xmin = float(input_settings['instruments'][intrument_index]['field-of-view_xmin'])
    fov_xmax = float(input_settings['instruments'][intrument_index]['field-of-view_xmax'])
    fov_ymin = float(input_settings['instruments'][intrument_index]['field-of-view_ymin'])
    fov_ymax = float(input_settings['instruments'][intrument_index]['field-of-view_ymax'])
    #data_snr = float(input_settings['instruments']['sn'])
    resolution = float(instru_settings['resolution'])
    # the following follows VKL conventions for defining the coordinates grid
    width  = fov_xmax - fov_xmin
    height = fov_ymax - fov_ymin
    step_x = resolution
    step_y = resolution
    Nx = int(width / step_x)
    Ny = int(height / step_y)
    ra_at_xy_0 = -width/2. + step_x/2.0
    dec_at_xy_0 = -height/2. + step_y/2.0
    # here we assume pixels are square
    transform_pix2angle = resolution * np.eye(2)
    kwargs_pixel = {'nx': Nx, 'ny': Ny,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
                    'transform_pix2angle': transform_pix2angle}
    return PixelGrid(**kwargs_pixel)

def get_simulated_data(molet_dir, input_file, instrument_name):
    data_path = os.path.join(molet_dir, os.path.dirname(input_file), 'output', f'OBS_{instrument_name}.fits')
    return fits.getdata(data_path).astype(float)
