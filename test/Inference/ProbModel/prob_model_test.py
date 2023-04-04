# Testing modeling workflows
# 
# Copyright (c) 2023, herculens developers and contributors


import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from herculens.Inference.ProbModel.numpyro import NumpyroModel


def test_num_parameters_numpyro():

    class MyProbModel(NumpyroModel):
    
        def model(self):
            # Parameters of the source
            prior_source = [
              {
                  'amp': numpyro.sample('source_amp', dist.LogNormal(1.0, 0.1)),
             'R_sersic': numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.1, 0.05, low=0.02)), 
             'n_sersic': numpyro.sample('source_n', dist.Uniform(1., 3.)), 
             'e1': numpyro.sample('source_e1', dist.TruncatedNormal(0.1, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('source_e2', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': numpyro.sample('source_center_x', dist.Normal(0.05, 0.05)), 
            'center_y': numpyro.sample('source_center_y', dist.Normal(0.1, 0.05))}
            ]

            # Parameters of the lens
            cx = numpyro.sample('lens_center_x', dist.Normal(0., 0.05))
            cy = numpyro.sample('lens_center_y', dist.Normal(0., 0.05))
            prior_lens = [
                # power-law
            {
                'theta_E': numpyro.sample('lens_theta_E', dist.Normal(1.5, 0.1)),
                'gamma': numpyro.sample('lens_gamma', dist.Normal(2.1, 0.05)),
             'e1': numpyro.sample('lens_e1', dist.TruncatedNormal(0.1, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('lens_e2', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': cx, 
             'center_y': cy},
                # external shear, with fixed origin
            {'gamma1': numpyro.sample('lens_gamma1', dist.TruncatedNormal(-0.03, 0.05, low=-0.3, high=0.3)), 
             'gamma2': numpyro.sample('lens_gamma2', dist.TruncatedNormal(0.02, 0.05, low=-0.3, high=0.3)), 
             'ra_0': 0.0, 'dec_0': 0.0}
            ]

            # Parameters of the lens light, with center relative the lens mass
            prior_lens_light = [
            {'amp': numpyro.sample('light_amp', dist.LogNormal(1.0, 0.1)) , 
             'R_sersic': numpyro.sample('light_R_sersic', dist.Normal(1.3, 0.05)), 
             'n_sersic': numpyro.sample('light_n', dist.Uniform(2., 4.)), 
             'e1': numpyro.sample('light_e1', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('light_e2', dist.TruncatedNormal(0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': numpyro.sample('light_center_x', dist.Normal(cx, 0.01)), 
             'center_y': numpyro.sample('light_center_y', dist.Normal(cy, 0.01))}
            ]
            
    prob_model = MyProbModel()

    assert prob_model.num_parameters == 22
