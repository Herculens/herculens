from jaxtronomy.LensModel.Profiles.base_profile import LensProfileBase
from jaxtronomy.Util.models import Net, get_grad
import numpy as np
import jax.numpy as jnp


class NetworkPotential(LensProfileBase):
    param_names = ['state']

    def __init__(self):
        """Lensing potential modlled by a network trained on a fixed coordinate grid."""
        super(NetworkPotential, self).__init__()

    def function(self, x, y, state):
        """Evaluation of the lensing potential modelled by the network.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential.
        state : FreezedDictionary
            Networks parameters and training hyperparaeters

        """
        # Due to matching scipy's interpolation, we need to switch x and y
        # coordinates as well as transpose
        mgrid = np.stack([y,x], axis=-1)
        mgrid = mgrid.reshape(-1, 2)
        mgrid=jnp.float32(mgrid)
        a = state.unfreeze()
        l = [a[i]['kernel'].shape[1] for i in list(a.keys())]
        pot = Net(l).apply({'params': state},mgrid)
        return np.reshape(pot,(len(x),len(y)))

    def derivatives(self, x, y, state):
        """Spatial first derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential.
        state : FreezedDictionary
            Networks parameters and training hyperparaeters

        """
        mgrid = np.stack([y,x], axis=-1)
        mgrid = mgrid.reshape(-1, 2)
        mgrid=jnp.float32(mgrid)
        a = state.unfreeze()
        l = [a[i]['kernel'].shape[1] for i in list(a.keys())]
        pot_d= get_grad(tuple(l),state,mgrid)
        return np.reshape(jnp.take(pot_d,1,axis=1),x.shape), np.reshape(jnp.take(pot_d,0,axis=1),y.shape)


