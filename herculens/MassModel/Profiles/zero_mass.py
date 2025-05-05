# Defines en dual pseudo isothermal elliptical (dPIE) mass profile
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.tree_util import register_pytree_node


__all__ = [
    'ZeroMass', 
]


class ZeroMass(object):
    """
    Dummy profile that does not have any mass associated with it.
    """
    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}
    fixed_default = {}
    
    @partial(jit, static_argnums=(0,))
    def function(self, x, y, **kwargs):
        return jnp.zeros_like(x)

    @partial(jit, static_argnums=(0,))
    def derivatives(self, x, y, **kwargs):
        return jnp.zeros_like(x), jnp.zeros_like(y)
    
    @partial(jit, static_argnums=(0,))
    def hessian(self, x, y, **kwargs):
        return jnp.zeros_like(x), jnp.zeros_like(y), jnp.zeros_like(y)
    
def flatten_func(self):
    children = ()
    aux_data = {}
    return (children, aux_data)

def unflatten_func(aux_data, children):
    return ZeroMass()

register_pytree_node(ZeroMass, flatten_func, unflatten_func)
