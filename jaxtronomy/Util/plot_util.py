import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
                  invisible=False, divider_kwargs={}, colorbar_kwargs={}, label_kwargs={}):
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
    return cb

def nice_colorbar_residuals(mappable, res_map, vmin, vmax, position='right', pad=0.1, size='5%', 
                            invisible=False, label=None, fontsize=16):
    if res_map.min() < vmin and res_map.max() > vmax:
        cb_extend = 'both'
    elif res_map.min() < vmin:
        cb_extend = 'min'
    elif res_map.max() > vmax:
        cb_extend = 'max'
    else:
        cb_extend = 'neither'
    nice_colorbar(mappable, position=position, pad=pad, size=size, label=label, fontsize=fontsize,
                  invisible=invisible, colorbar_kwargs={'extend': cb_extend})