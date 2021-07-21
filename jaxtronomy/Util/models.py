from typing import Sequence
import jax
import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn 
from flax.training import train_state  
import optax        



class Net(nn.Module):
    features: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.swish(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def create_train_state(rng, learning_rate, input_size, hidden_sizes):
    """
    instantiates network and optimizers 

    :rng: random key
    :learning_rate: optimizer's learning rate
    :input_size: input data dimension
    :hidden_sizes: number of nodes per hidden layer

    """
    net = Net(hidden_sizes)
    params = net.init(rng, jnp.ones([1, input_size]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)

@jax.jit
def mse_loss(*,logits,labels,noise_add):
      return 0.5*jnp.sum(jnp.square(labels - logits)/noise_add)


@jax.partial(jax.jit, static_argnums=(0,)) 
def get_grad(hidd,params_,image_):
    def forward_step(image_n):
        logits = Net(hidd).apply({'params': params_}, image_n)
        return jnp.reshape(logits, ())
    grad_fn = vmap(jax.grad(forward_step))
    grads = grad_fn(image_)
    return grads