# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2020, SLITronomy developers and contributors

__author__ = 'aymgal'


import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_minimize_history(parameters, opt_extra_fields, max_num_params=6):
    # tood: implement support for multi-start optimization
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    ax = axes[0]
    ax.plot(range(len(opt_extra_fields['loss_history'])), opt_extra_fields['loss_history'])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax = axes[1]
    if 'param_history' in opt_extra_fields:
        param_history = np.array(opt_extra_fields['param_history'])
        for i in range(min(len(parameters.names), max_num_params)):
            ax.plot(range(len(opt_extra_fields['loss_history'])), 
                    (param_history[:, i] - param_history[-1, i]) / param_history[-1, i], 
                    label=parameters.symbols[i])
        ax.set_ylabel("Parameter trace")
        ax.set_xlabel("Iteration")
        ax.legend(loc='upper right')
    else:
        warnings.warn("No `'param_history'` found in the extra fields (use `return_param_history=True` in Optimizer).")
        ax.axis('off')
    fig.tight_layout()
    return fig

def std_colorbar(mappable, label=None, fontsize=12, label_kwargs={}, **colorbar_kwargs):
    cb = plt.colorbar(mappable, **colorbar_kwargs)
    if label is not None:
        colorbar_kwargs.pop('label', None)
        cb.set_label(label, fontsize=fontsize, **label_kwargs)
    return cb

def std_colorbar_residuals(mappable, res_map, vmin, vmax, label=None, fontsize=12, 
                           label_kwargs={}, **colorbar_kwargs):
    if res_map.min() < vmin and res_map.max() > vmax:
        cb_extend = 'both'
    elif res_map.min() < vmin:
        cb_extend = 'min'
    elif res_map.max() > vmax:
        cb_extend = 'max'
    else:
        cb_extend = 'neither'
    colorbar_kwargs.update({'extend': cb_extend})
    return std_colorbar(mappable, label=label, fontsize=fontsize, 
                        label_kwargs=label_kwargs, **colorbar_kwargs)

def nice_colorbar(mappable, position='right', pad=0.1, size='5%', label=None, fontsize=12, 
                  invisible=False, max_nbins=None,
                  divider_kwargs={}, colorbar_kwargs={}, label_kwargs={}):
    divider_kwargs.update({'position': position, 'pad': pad, 'size': size})
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**divider_kwargs)
    if invisible:
        cax.axis('off')
        return None
    cb = plt.colorbar(mappable, cax=cax, **colorbar_kwargs)
    if label is not None:
        colorbar_kwargs.pop('label', None)
        cb.set_label(label, fontsize=fontsize, **label_kwargs)
    if position == 'top':
        cax.xaxis.set_ticks_position('top')
    if max_nbins is not None:
        cb.locator = ticker.MaxNLocator(nbins=max_nbins)
        cb.update_ticks()
    return cb

def nice_colorbar_residuals(mappable, res_map, vmin, vmax, position='right', pad=0.1, size='5%', 
                            invisible=False, label=None, fontsize=12,
                            divider_kwargs={}, colorbar_kwargs={}, label_kwargs={}):
    if res_map.min() < vmin and res_map.max() > vmax:
        cb_extend = 'both'
    elif res_map.min() < vmin:
        cb_extend = 'min'
    elif res_map.max() > vmax:
        cb_extend = 'max'
    else:
        cb_extend = 'neither'
    colorbar_kwargs.update({'extend': cb_extend})
    return nice_colorbar(mappable, position=position, pad=pad, size=size, label=label, fontsize=fontsize,
                  invisible=invisible, colorbar_kwargs=colorbar_kwargs, label_kwargs=label_kwargs,
                  divider_kwargs=divider_kwargs)
