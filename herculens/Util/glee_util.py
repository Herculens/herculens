# Utility class to load GLEE files and convert to Herculens syntax

__author__ = 'aymgal'

import numpy as np
from pprint import pprint


class GLEEReader(object):
    """Load and parse a GLEE config file.

    NOTE: As of now, not all the config file can be parsed.
    Things that are not parsed are, for instance:
    - a:X,X,X statements
    - step:X,X statements
    - min:X and max:X
    - a large part of the esources properties
    - cosmology

    Parameters
    ----------
    config_file_path : str
        Path to the config file.
    verbose : bool
        Whether or not to print warning messages. Default to False.
    """

    _num_param_for_profile = {
        # TODO: add other mass/light profiles
        'dpie': 7,
        # 'piemd': 7,
        'shear': 2,
    }
    _profile_names = list(_num_param_for_profile.keys())

    def __init__(self, config_file_path, verbose=False):
        self._config_path = config_file_path
        self._verbose = verbose
        self._parsed = False

    def print_summary(self):
        if not self._parsed:
            self._raise_parser_run_error()
        print("="*60)
        print("Parsed GLEE model:")
        print("-"*18)
        print("> Number of lenses:", self.num_lenses)
        print("> Number of point-like sources:", self.num_point_like_sources)
        print(f"  leading to {self.num_point_like_images} point-like multiple images")
        print("> Number of extended sources:", self.num_extended_sources)
        print("="*60)

    @property
    def num_lenses(self):
        return len(self.lens_parameters)

    @property
    def num_point_like_sources(self):
        return len(self.point_like_source_parameters)

    @property
    def num_point_like_images(self):
        params = self.point_like_source_parameters
        return sum([len(p['x_img']) for p in params])

    @property
    def num_extended_sources(self):
        return len(self.extended_source_parameters)
        
    @property
    def lens_redshifts(self):
        redshifts = [p['z'] for p in self.lens_parameters]
        redshifts_priors = [p['z'] for p in self.lens_priors]
        return redshifts, redshifts_priors
    
    @property
    def lens_profiles(self):
        if not hasattr(self, '_lens_profiles'):
            self._raise_parser_run_error()
        return self._lens_profiles
    
    @property
    def lens_parameters(self):
        if not hasattr(self, '_lens_params'):
            self._raise_parser_run_error()
        return self._lens_params

    @property
    def lens_priors(self):
        if not hasattr(self, '_lens_priors'):
            self._raise_parser_run_error()
        return self._lens_priors

    @property
    def lens_labels(self):
        if not hasattr(self, '_lens_labels'):
            self._raise_parser_run_error()
        return self._lens_labels

    @property
    def point_like_source_redshifts(self):
        redshifts = [p['z'] for p in self.point_like_source_parameters]
        redshifts_priors = [p['z'] for p in self.point_like_source_priors]
        return redshifts, redshifts_priors
    
    @property
    def point_like_source_parameters(self):
        if not hasattr(self, '_ptl_src_params'):
            self._raise_parser_run_error()
        return self._ptl_src_params

    @property
    def point_like_source_priors(self):
        if not hasattr(self, '_ptl_src_priors'):
            self._raise_parser_run_error()
        return self._ptl_src_priors

    @property
    def point_like_source_labels(self):
        if not hasattr(self, '_ptl_src_labels'):
            self._raise_parser_run_error()
        return self._ptl_src_labels

    @property
    def point_like_source_errors(self):
        if not hasattr(self, '_ptl_src_errors'):
            self._raise_parser_run_error()
        return self._ptl_src_errors
    
    def point_like_source_group_indices(self):
        """Return a list of lists containing indices of point-like sources
        which have the redshift, order by increasing redshift.
        For instance: [
            [redshift1_index1, redshift1_index2],  # redshift 1
            [redshift2_index1],  # redshift 2
            [redshift3_index1, redshift3_index2, redshift3_index3],  # redshift 3
            ...
        ]

        Returns
        -------
        list
            List lists of integers
        """
        redshifts = np.array(self.point_like_source_redshifts[0])
        iter_indices = np.argsort(redshifts)
        group_indices = [
            [iter_indices[0]]
        ]
        unique_redshifts = [redshifts[iter_indices[0]]]
        for i in iter_indices[1:]:
            z_current = redshifts[i]
            z_prev = redshifts[i-1]
            if z_current != z_prev:
                group_indices.append([])
                unique_redshifts.append(z_current)
            group_indices[-1].append(i)
        return unique_redshifts, group_indices
    
    @property
    def extended_source_redshifts(self):
        redshifts = [p['z'] for p in self.extended_source_parameters]
        redshifts_priors = [p['z'] for p in self.extended_source_priors]
        return redshifts, redshifts_priors
    
    @property
    def extended_source_parameters(self):
        if not hasattr(self, '_ext_src_params'):
            self._raise_parser_run_error()
        return self._ext_src_params

    @property
    def extended_source_priors(self):
        if not hasattr(self, '_ext_src_priors'):
            self._raise_parser_run_error()
        return self._ext_src_priors

    @property
    def extended_source_settings(self):
        if not hasattr(self, '_ext_src_settings'):
            self._raise_parser_run_error()
        return self._ext_src_settings

    @property
    def extended_source_ref_coordinates(self):
        return [s['refcoord'] for s in self.extended_source_settings]

    def parse_config(self):
        lines = self.read_and_separate_lines(self._config_path)
        # next we go through the lines and group them by lens/source/esource blocks
        model_component_blocks = self._extract_blocks(
            lines, ['lenses_vary', 'sources', 'esources'],
        )
        # parse the lens block
        self._lens_profiles, self._lens_params, self._lens_priors, self._lens_labels \
            = self._parse_lens_model(model_component_blocks['lenses_vary'])
        # parse the point-like source block
        self._ptl_src_params, self._ptl_src_priors, self._ptl_src_labels, self._ptl_src_errors \
            = self._parse_point_like_source_model(model_component_blocks['sources'])
        # parse the extended source block
        self._ext_src_params, self._ext_src_priors, self._ext_src_settings \
            = self._parse_extended_sources_block(model_component_blocks['esources'])
        # if successful so far, then we say it's parsed
        self._parsed = True

    def _parse_lens_model(self, block):
        header = block[0]
        content = block[1:]
        component_type = header[0]
        num_components = int(header[1])
        model_profile_blocks, profile_types = self._extract_blocks(
            content, self._profile_names, with_duplicates=True,
        )
        assert num_components == len(profile_types), f"Inconsistent number of lens profiles ('{component_type}')"
        return self._parse_all_lens_profiles(profile_types, model_profile_blocks)

    def _parse_all_lens_profiles(self, types, blocks):
        params_list = []
        priors_list = []
        labels_list = []
        for i, (type_, block) in enumerate(zip(types, blocks)):
            header = block[0]
            content = block[1:]
            assert self.get_line_item(header) == type_, f"Inconsistent lens profile type ({i})"
            kwargs_params, kwargs_priors, kwargs_labels = self._parse_lens_profile(i, content)
            params_list.append(kwargs_params)
            priors_list.append(kwargs_priors)
            labels_list.append(kwargs_labels)
        return types, params_list, priors_list, labels_list
    
    def _parse_lens_profile(self, i, lines):
        redshift, redshift_prior, redshift_label = self._parse_redshift(i, lines[0])
        kwargs_params, kwargs_priors, kwargs_labels = self._parse_lens_parameters(lines[1:])
        # add the redshift to the rest of the parameters
        kwargs_params['z'] = redshift
        kwargs_priors['z'] = redshift_prior
        kwargs_labels['z'] = redshift_label
        return kwargs_params, kwargs_priors, kwargs_labels

    def _parse_lens_parameters(self, lines):
        kwargs_params = {}
        kwargs_priors = {}
        kwargs_labels = {}
        for line in lines:
            if self.line_is_empty(line):
                continue
            value = float(self.get_line_item(line, pos=0))
            name = self.get_line_item(line, pos=1).replace('#', '')
            prior = self._parse_prior(line)
            label = self._parse_label(line)
            kwargs_params[name] = value
            kwargs_priors[name] = prior
            kwargs_labels[name] = label
        return kwargs_params, kwargs_priors, kwargs_labels

    def _parse_point_like_source_model(self, block):
        header = block[0]
        content = block[1:]
        component_type = header[0]
        num_components = int(header[1])
        model_component_blocks, _ = self._extract_blocks(
            content, ['z'], with_duplicates=True,  # we extract the block that starts with z
        )
        assert (num_components == len(model_component_blocks), 
                f"Inconsistent number of point-like sources ('{component_type}')")
        return self._parse_all_point_like_sources(model_component_blocks)

    def _parse_all_point_like_sources(self, blocks):
        params_list = []
        priors_list = []
        labels_list = []
        errors_list = []
        for i, block in enumerate(blocks):
            kwargs_params, kwargs_priors, kwargs_errors, kwargs_labels \
                = self._parse_point_like_source(i, block)
            params_list.append(kwargs_params)
            priors_list.append(kwargs_priors)
            labels_list.append(kwargs_labels)
            errors_list.append(kwargs_errors)
        return params_list, priors_list, labels_list, errors_list

    def _parse_point_like_source(self, i, lines):
        redshift, redshift_prior, redshift_label = self._parse_redshift(i, lines[0])
        kwargs_params, kwargs_priors, kwargs_errors = self._parse_ptl_src_parameters(i, lines[1:])
        # add the redshift to the rest of the parameters
        kwargs_params['z'] = redshift
        kwargs_priors['z'] = redshift_prior
        kwargs_errors['z'] = None
        kwargs_labels = {'z': redshift_label}
        return kwargs_params, kwargs_priors, kwargs_errors, kwargs_labels

    def _parse_ptl_src_parameters(self, i, lines):
        # type of point-like source weighting (?)
        assert self.get_line_item(lines[0], pos=0) == 'source'  # sanity check
        src_type = self.get_line_item(lines[0], pos=1)
        # source plane position coordinates
        assert self.get_line_item(lines[1], pos=0) == 'srcx'  # sanity check
        x_src = float(self.get_line_item(lines[1], pos=1))
        prior_x_src = self._parse_prior(lines[1])
        assert self.get_line_item(lines[2], pos=0) == 'srcy'  # sanity check
        y_src = float(self.get_line_item(lines[2], pos=1))
        prior_y_src = self._parse_prior(lines[2])
        # number of multiple images
        num_images = int(self.get_line_item(lines[3]))
        # image plane positions coordinates
        x_img, y_img = [], []
        err_img = []
        for line in lines[4:]:
            if self.line_is_empty(line):
                continue
            x_img.append(float(self.get_line_item(line, pos=0)))
            y_img.append(float(self.get_line_item(line, pos=1)))
            err_img.append(self._parse_error(line))
        assert len(x_img) == len(y_img) == num_images, f"Inconsistent number of multiple images for point-like source {i}."
        kwargs_params = {
            'x_src': x_src,
            'y_src': y_src,
            'x_img': x_img,
            'y_img': y_img,
        }
        kwargs_priors = {
            'x_src': prior_x_src,
            'y_src': prior_y_src,
        }
        kwargs_errors = {
            'x_img': err_img,
            'y_img': err_img,
        }
        return kwargs_params, kwargs_priors, kwargs_errors
    
    def _parse_extended_sources_block(self, block):
        header = block[0]
        content = block[1:]
        component_type = header[0]
        num_components = int(header[1])
        model_component_blocks, _ = self._extract_blocks(
            content, ['z'], with_duplicates=True,  # we extract the block that starts with z
        )
        assert num_components == len(model_component_blocks), f"Inconsistent number of extended sources ('{component_type}')"
        return self._parse_all_extended_sources(model_component_blocks)

    def _parse_all_extended_sources(self, blocks):
        params_list = []
        priors_list = []
        settings_list = []
        for i, block in enumerate(blocks):
            kwargs_params, kwargs_priors, kwargs_settings \
                = self._parse_extended_source(i, block)
            params_list.append(kwargs_params)
            priors_list.append(kwargs_priors)
            settings_list.append(kwargs_settings)
        return params_list, priors_list, settings_list

    def _parse_extended_source(self, i, lines):
        redshift, redshift_prior, _ = self._parse_redshift(i, lines[0])
        kwargs_settings = self._parse_extended_source_settings(i, lines[1:])
        # add the redshift to the rest of the parameters
        kwargs_params = {'z': redshift}
        kwargs_priors = {'z': redshift_prior}
        return kwargs_params, kwargs_priors, kwargs_settings

    def _parse_extended_source_settings(self, i, lines):
        kwargs_settings = {}
        for line in lines:
            setting_name = self.get_line_item(line, pos=0)
            if setting_name is None or setting_name == 'esource_end':
                break
            elif setting_name == 'esource_refcoord':
                ref_x = float(self.get_line_item(line, pos=1))
                ref_y = float(self.get_line_item(line, pos=2))
                setting_value = (ref_x, ref_y)
            else:
                setting_value = self.get_line_item(line, pos=1)
            setting_name = setting_name.replace('esource_', '')  # remove the 'esource' prefix
            kwargs_settings[setting_name] = setting_value
        return kwargs_settings
    
    def _parse_redshift(self, i, line):
        assert self.get_line_item(line) == 'z'
        z = self.get_line_item(line, pos=1)  # redshift value is the next one
        prior = self._parse_prior(line)
        label = self._parse_label(line)
        return float(z), prior, label

    def _parse_prior(self, line):
        # TODO: replace pos=2 with a list of start_with
        prior_item = self.get_line_item(line, pos=2).split(':')
        prior_type = prior_item[0]
        if prior_type in ('exact', 'noprior'):
            val1, val2 = None, None
        elif prior_type == 'flat':
            bounds = prior_item[1].split(',')
            val1 = float(bounds[0])
            val2 = float(bounds[1])
        elif prior_type == 'link':
            link_to_label = prior_item[1]
            val1 = link_to_label
            val2 = None
        else:
            raise ValueError(f"Prior type '{prior_type}' is not supported")
        return (prior_type, val1, val2)

    def _parse_error(self, line):
        error_item = self.get_line_item(line, start_with='error')
        if error_item is None:
            return None
        error_item = error_item.split(':')
        error_type = error_item[0]
        error_value = float(error_item[1])
        assert error_type == 'error'  # sanity check
        return error_value

    def _parse_label(self, line):
        label_item = self.get_line_item(line, start_with='label')
        if label_item is None:
            return None
        label = label_item.split(':')[1]
        return label
    
    def _extract_blocks(self, lines, separator_strings, with_duplicates=False):
        num_lines = len(lines)
        lines.append([[None]])  # just for practicality, adds a line with just None
        if with_duplicates:
            # if there are several blocks with the same separator, we use two lists
            # to allow for duplicates
            block_keys = []
            blocks = []
        else:
            # in this case we use a dictionary as we group all blocks
            # that belong to the same separator
            blocks = {k: [] for k in separator_strings}
        current_block_type = None
        block_is_done = True
        for i in range(num_lines):
            line_items = lines[i]
            first_item = self.get_line_item(line_items)
            next_first_item = self.get_line_item(lines[i+1])
            if first_item in separator_strings:
                current_block_type = first_item
                block_is_done = False
                if with_duplicates:
                    block_keys.append(current_block_type)
                    blocks.append([])
            if not block_is_done:
                if with_duplicates:
                    blocks[-1].append(line_items)
                else:
                    blocks[current_block_type].append(line_items)
            if next_first_item in separator_strings:
                block_is_done = True
        if with_duplicates:
            return blocks, block_keys
        return blocks

    @staticmethod
    def get_line_item(line, pos=0, start_with=None):
        """
        Given a 'line' defined as a list of strings, get the next (if pos=0) 
        non-empty (i.e. different from ''), or any one after if pos>0.
        """
        if start_with is not None and not isinstance(start_with, (str, tuple, list)):
            raise ValueError("`start_with` can only be None, a string or a list/tuple of strings.")
        i = 0
        for item in line:
            if start_with is None:
                if item != '':
                    if pos == i:
                        return item
                    i += 1
            elif isinstance(start_with, str):
                if start_with in item:
                    return item
            else:
                for start_with_sgl in start_with:
                    if start_with_sgl in item:
                        return item
        return None  # if nothing was found

    @staticmethod
    def line_is_empty(line):
        return len(line) == 0 or all([l == '' for l in line])

    @staticmethod
    def read_and_separate_lines(file_path):
        with open(file_path) as f:
            lines = f.readlines()
        # remove \n and split each line into individual items
        lines = [l.replace('\n', '').split(' ') for l in lines]
        return lines

    def _print(self, txt):
        if self._verbose is True:
            print(txt)
    
    def _raise_parser_run_error(self):
        raise RuntimeError("You must run parse_config() beforehand.")
