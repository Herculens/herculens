# Utility class to load GLEE files and convert to Herculens syntax

__author__ = 'aymgal'


from pprint import pprint


class GLEEReader(object):
    """Classes that load, parse and convert a GLEE config file 
    to a Herculens model.

    Parameters
    ----------
    config_file_path : str
        Path to the config file.
    """

    _num_param_for_profile = {
        'dpie': 7,
        'shear': 2,
    }
    _profile_names = list(_num_param_for_profile.keys())

    def __init__(self, config_file_path):
        self._config_path = config_file_path

    def parse_config(self):
        lines = self.read_and_separate_lines(self._config_path)
        # next we go through the lines and group them by lens/source/esource blocks
        model_component_blocks = self._extract_blocks(
            lines, ['lenses_vary', 'sources', 'esources'],
        )
        self._parse_lenses(model_component_blocks['lenses_vary'])
        # self._parse_point_like_sources_block(ptl_source_block)
        # self._parse_extended_sources_block(ext_source_block)

    def _parse_lenses(self, block):
        header = block[0]
        content = block[1:]
        component_type = header[0]
        num_components = int(header[1])
        model_profile_blocks, profile_types = self._extract_blocks(
            content, self._profile_names, with_duplicates=True,
        )
        assert num_components == len(profile_types), f"Inconsistent number of lens profiles ('{component_type}')"
        self._parse_lens_profiles(profile_types, model_profile_blocks)

    def _parse_lens_profiles(self, types, blocks):
        for i, (type, block) in enumerate(zip(types, blocks)):
            header = block[0]
            content = block[1:]
            assert self.first_line_item(header) == type, f"Inconsistent lens profile type ({i})"

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
            first_item = self.first_line_item(line_items)
            next_first_item = self.first_line_item(lines[i+1])
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
    def first_line_item(line):
        for l in line:
            if l != '':
                return l

    @staticmethod
    def read_and_separate_lines(file_path):
        with open(file_path) as f:
            lines = f.readlines()
        # remove \n and split each line into individual items
        lines = [l.replace('\n', '').split(' ') for l in lines]
        return lines
    