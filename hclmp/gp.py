"""
This submodule contains code for creating and handling Gaussian Processes.
"""

__author__ = 'Kevin Tran'
__email__ = 'kevin.tran@tri.global'

import gpytorch
import torch


class MultiTaskSVGP(gpytorch.models.ApproximateGP):
    """
    Multi-task/objective, sparse, variational Gaussian process. This code is
    modified from the `MultitaskGPModel` object in GpyTorch's example notebook,
    `SVGP_Multitask_GP_Regression.ipynb`.
    """

    def __init__(self,
                 num_tasks: int,
                 num_latents: int = 1,
                 n_inducing_pts: int = 16,
                 device: torch.device = None,
                 ):
        """
        Initialize this multi-objective SVGP

        Args:
            num_tasks (int): The number of tasks/objectives for this model to
            be learning/predicting

            num_latents (int): This model assumes that each of the
            tasks/objectives/outputs are linear combinations of some latent GP
            functions. This argument allows you to choose the number of these
            latent functions.

            n_inducing_pts (int): The number of inducing points to use for the
            kernel approximation. Higher numbers yield more accurate results,
            but they come at higher memory costs.

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GP.
        """

        if device is None:
            device = (torch.device('cuda') if torch.cuda.is_available() else
                      torch.device('cpu'))

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, n_inducing_pts, 1).to(device)

        # We have to mark the CholeskyVariationalDistribution as batch so that
        # we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than
        # a batch output
        base_var_strat = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        lmc_var_strat = gpytorch.variational.LMCVariationalStrategy(
            base_var_strat,
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(lmc_var_strat)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]))
        kernel = gpytorch.kernels.MaternKernel(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=torch.Size([num_latents]))

    def forward(self, x):

        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
