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

def contour_with_legend(ax, array, kwargs_contour={}, kwargs_legend={}, logscale=True):
    """
    Add contours to an axis with a legend that actually works.
    Works with a single array or a list of arrays.
    """
    if (isinstance(array, np.ndarray) and array.ndim == 3) or (isinstance(array, (list, tuple)) and len(array) > 1):
        if isinstance(kwargs_contour, dict):
            kwargs_contour_list = [kwargs_contour] * array.shape[0]
        else:
            kwargs_contour_list = kwargs_contour
        if len(kwargs_contour_list) != len(array):
            raise ValueError("The number of contour kwargs must match the number of arrays.")
        array_list = array
    else:
        array_list = [array]
        kwargs_contour_list = [kwargs_contour]
    legend_handles = []
    for i, (arr, kw) in enumerate(zip(array_list, kwargs_contour_list)):
        label = kw.pop('label', f"Contour {i}")
        arr_transf = np.log10(arr) if logscale else arr
        ax.contour(
            arr_transf, **kw,
        )
        legend_handles.append(
            plt.Line2D(
                [0], [0], 
                linestyle=kw.get('linestyles', '-'),
                linewidth=kw.get('linewidths', 1),
                color=kw.get('colors', 'black'), 
                label=label,
            )
        )
    ax.legend(
        handles=legend_handles,
        **kwargs_legend,
    )

# def clip(data, nsigma):
#     """
#     Iteratively removes data until all is within nsigma of the median, then returns the median and std
#     author: Cameron Lemon
#     """
#     lennewdata = 0
#     lenolddata = data.size
#     while lenolddata>lennewdata:
#         lenolddata = data.size
#         data = data[np.where((data<np.nanmedian(data)+nsigma*np.nanstd(data))&(data>np.nanmedian(data)-nsigma*np.nanstd(data)))]
#         lennewdata = data.size
#     return np.median(data), np.std(data)
# 18 h 08
# bg, sigma = clip(data, 4.)
# lenspixels = data[data>bg+3*sigma]
# ax.imshow(data, origin='lower', cmap=cm, norm=LogNorm(vmax=np.nanpercentile(lenspixels, 95.), vmin=bg))
