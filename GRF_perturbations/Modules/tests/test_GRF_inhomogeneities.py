import unittest

from GRF_perturbations.Modules.GRF_inhomogeneities_class import *
import scipy as sc


class test_GRF_inhomogeneities(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.GRF_class = GRF_inhomogeneities_class(100, 0.08, 100)

    def test_box_muller(self):
        BM_variates = np.zeros((1000, 2))
        for i in range(1000):
            np.random.seed(i)
            BM_variates[i] = Box_Muller_transform()

        np.random.seed(42)

        # Should be independent
        p_value = sc.stats.spearmanr(BM_variates[:, 0], BM_variates[:, 1])[1]
        self.assertGreater(p_value, 0.5)

        self.assertAlmostEqual(BM_variates[:, 0].std(), 1., delta=0.1, msg='Std is not 1')
        self.assertAlmostEqual(BM_variates[:, 1].std(), 1., delta=0.1, msg='Std is not 1')

        self.assertAlmostEqual(BM_variates[:, 0].mean(), 0., delta=0.1, msg='Mean is not 0')
        self.assertAlmostEqual(BM_variates[:, 1].mean(), 0., delta=0.1, msg='Mean is not 0')


    def test_sample_unit_fourier_image(self):
        # Consider only uncorrected points
        mask_full_variates = np.ones((100, 100), dtype=bool)
        mask_full_variates[0, 0] = False
        mask_full_variates[0, 50] = False
        mask_full_variates[50, 0] = False
        mask_full_variates[50, 50] = False

        # Real and Image should be standard normal
        Fourier_values = (np.sqrt(2)) * self.GRF_class.tensor_unit_Fourier_images[:, mask_full_variates]
        self.assertTrue(np.allclose(Fourier_values.real.std(axis=1), 1., rtol=0.05), msg='Std is not 1')
        self.assertTrue(np.allclose(Fourier_values.imag.std(axis=1), 1., rtol=0.05), msg='Std is not 1')

    def test_potential(self):

        logA_array=[-9.,-8.,-7.]
        Beta_array=[0,2,4]

        theoretical_Variances=np.zeros((len(logA_array),len(Beta_array),100))
        generated_Variances=np.zeros((len(logA_array),len(Beta_array),100))

        for i,logA in enumerate(logA_array):
            for j,Beta in enumerate(Beta_array):
                #100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i,j]=np.repeat(self.GRF_class.field_variance([logA,Beta],field='potential'),100)
                #100 variances for each logA,Beta from actually sampled field
                Potentials = np.array([self.GRF_class.potential([logA,Beta],self.GRF_class.tensor_unit_Fourier_images[seed])\
                                       for seed in range(100)])
                generated_Variances[i,j]=Potentials.var(axis=(-1,-2))


        #Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances),np.log(theoretical_Variances),rtol=0.25))

    def test_alpha(self):

        logA_array = [-9., -8., -7.]
        Beta_array = [0, 2, 4]

        theoretical_Variances = np.zeros((len(logA_array), len(Beta_array), 100, 2))
        generated_Variances = np.zeros((len(logA_array), len(Beta_array), 100, 2))

        for i, logA in enumerate(logA_array):
            for j, Beta in enumerate(Beta_array):
                # 100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i, j,:,0] = np.repeat(self.GRF_class.field_variance([logA, Beta], field='alpha_y'),100)
                theoretical_Variances[i, j,:,1] = np.repeat(self.GRF_class.field_variance([logA, Beta], field='alpha_x'),100)

                # 100 variances for each logA,Beta from actually sampled field
                alphas_y = np.array([self.GRF_class.alpha([logA, Beta], self.GRF_class.tensor_unit_Fourier_images[seed],\
                                                        direction='y') for seed in range(100)])
                alphas_x = np.array([self.GRF_class.alpha([logA, Beta], self.GRF_class.tensor_unit_Fourier_images[seed], \
                                                          direction='x') for seed in range(100)])

                generated_Variances[i, j,:,0] = alphas_y.var(axis=(-1, -2))
                generated_Variances[i, j, :, 1] = alphas_x.var(axis=(-1, -2))

        # Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances), np.log(theoretical_Variances), rtol=0.25))

    def test_kappa(self):

        logA_array=[-9.,-8.,-7.]
        Beta_array=[0,2,4]

        theoretical_Variances=np.zeros((len(logA_array),len(Beta_array),100))
        generated_Variances=np.zeros((len(logA_array),len(Beta_array),100))

        for i,logA in enumerate(logA_array):
            for j,Beta in enumerate(Beta_array):
                #100 variances for each logA,Beta from Parseval's theorem
                theoretical_Variances[i,j]=np.repeat(self.GRF_class.field_variance([logA,Beta],field='kappa'),100)
                #100 variances for each logA,Beta from actually sampled field
                kappas = np.array([self.GRF_class.kappa([logA,Beta],self.GRF_class.tensor_unit_Fourier_images[seed])\
                                       for seed in range(100)])
                generated_Variances[i,j]=kappas.var(axis=(-1,-2))

        # Test that logVariances match
        self.assertTrue(np.allclose(np.log(generated_Variances), np.log(theoretical_Variances), rtol=0.25))

if __name__ == '__main__':
    unittest.main()
