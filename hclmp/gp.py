"""
This submodule contains code for creating and handling Gaussian Processes.
"""

__author__ = 'Kevin Tran'
__email__ = 'kevin.tran@tri.global'

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data
from tqdm.auto import tqdm

import hclmp.core
import hclmp.graph_encoder
import hclmp.model


class PenultDataset(torch.utils.data.Dataset):
    """
    Dataset whose inputs are the outputs from the penultimate layer of an
    H-CMLP model. The outputs of this dataset are the labels used during
    training of the H-CMLP model.
    """

    def __init__(self,
                 model: hclmp.model.Hclmp,
                 comp_dataset: hclmp.graph_encoder.CompositionData,
                 device: torch.device = None,
                 ):
        """
        Initialize this dataset

        Args:
            model (hclmp.model.Hclmp): Instance of a trained H-CMLP model

            comp_dataset (hclmp.graph_encoder.CompositionData): The dataset
            used to train the given model

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GP.
        """

        self.model = model
        self.comp_dataset = comp_dataset

        if device is None:
            device = (torch.device('cuda') if torch.cuda.is_available() else
                      torch.device('cpu'))
        self.device = device

        self._get_penult()

    def __len__(self):
        return len(self.comp_dataset)

    def __getitem__(self, idx):

        _, targets, _, _, _ = self.comp_dataset[idx]
        inputs = self.pen_values[idx]
        return inputs, targets

    def _get_penult(self):
        """
        Feed the composition dataset to the trained H-CLMP model to get the
        output of the penultimate layer in H-CLMP.
        """

        # Initialize predictions
        loader = torch.utils.data.DataLoader(
            self.comp_dataset,
            batch_size=2**9,  # Make this smaller if you hit memory issues
            shuffle=False,
            collate_fn=hclmp.graph_encoder.collate_batch)

        self.model.eval()
        pen_values = []

        # Calculate penultimate output
        with torch.no_grad():
            batch_iterator = tqdm(loader, desc='Generating penultimate values')
            for x_batch, _, generated_feature_batch, _, _ in batch_iterator:
                x_batch = tuple(feature.to(self.device) for feature in x_batch)
                penult = self.model.get_penultimate_output(
                    generated_feature_batch, *x_batch)

                # Format/store the data
                pen_values.append(penult.cpu().numpy())
        pen_values_stacked = np.concatenate(pen_values, axis=0)
        self.pen_values = torch.tensor(pen_values_stacked).to(self.device)


class MultiTaskSVGP(gpytorch.models.ApproximateGP):
    """
    Multi-task/objective, sparse, variational Gaussian process. This code is
    modified from the `MultitaskGPModel` object in GpyTorch's example notebook,
    `SVGP_Multitask_GP_Regression.ipynb`.
    """

    def __init__(self,
                 num_tasks: int,
                 num_features: int,
                 num_latents: int = 1,
                 num_inducing_points: int = 16,
                 device: torch.device = None,
                 ):
        """
        Initialize this multi-objective SVGP

        Args:
            num_tasks (int): The number of tasks/objectives for this model to
            be learning/predicting

            num_features (int): The number of inputs/features for this GP

            num_latents (int): This model assumes that each of the
            tasks/objectives/outputs are linear combinations of some latent GP
            functions. This argument allows you to choose the number of these
            latent functions.

            num_inducing_points (int): The number of inducing points to use for
            the kernel approximation. Higher numbers yield more accurate
            results, but they come at higher memory costs.

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GP.
        """

        if device is None:
            device = (torch.device('cuda') if torch.cuda.is_available() else
                      torch.device('cpu'))
        self.device = device

        # Let's use a different set of inducing points for each latent function
        self.inducing_points = torch.rand(
            num_latents, num_inducing_points, num_features).to(self.device)

        # We have to mark the CholeskyVariationalDistribution as batch so that
        # we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing_points,
            batch_shape=torch.Size([num_latents]),
        ).to(self.device)

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than
        # a batch output
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=self.inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True,
        ).to(self.device)
        lmc_var_strat = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        ).to(self.device)

        super().__init__(lmc_var_strat)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]))
        kernel = gpytorch.kernels.MaternKernel(
            batch_shape=torch.Size([num_latents])).to(self.device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=torch.Size([num_latents])).to(self.device)

    def forward(self, x):

        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_on_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        optimizer: torch.optim.Optimizer,
        n_epochs: int = 20,
        device: torch.device = None
    ):
        """
        Train this multitask SVGP

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader
            containing the training dataset to use

            likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): The
            likelihood object that should be used for this GP. The `num_tasks`
            argument used to instantiate this likelihood should probably be
            equal to the number of tasks that this GP is meant to perform.

            optimizer (torch.optim.Optimizer): The optimizer to use to find the
            model parameters

            n_epochs (int): The number of epochs that should be used to train
            this model

            device (torch.device): The device (e.g., CPU or GPU) that should be
            used during training. If `None` will default to cuda (if
            available).
        """

        # Auto-detection of the device to use (if necessary)
        if device is None:
            cuda_available = bool(torch.cuda.device_count())
            device = torch.device('cuda') if cuda_available else torch.device('cpu')

        # If we're in a notebook, we'll use a dynamic plot of the loss.  But if
        # we're in a normal python thread, then we'll just use tqdm and print
        # statements.
        notebook = hclmp.core.is_notebook()
        if notebook:
            fig = plt.figure()
            ax = fig.subplots(1, 1)
            from IPython.display import display, clear_output

        # Initialize training
        self.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood=likelihood,
                                            model=self,
                                            num_data=len(data_loader.dataset))
        losses = []
        minibatches = []

        epoch_iter = tqdm(range(n_epochs), total=n_epochs, desc='Epochs')
        for epoch in epoch_iter:

            # Divide training of each epoch into batches
            minibatch_iter = tqdm(data_loader, desc='Minibatch')
            for x_batch, y_batch in minibatch_iter:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Train on this one batch
                optimizer.zero_grad()
                output = self(x_batch)
                loss = -mll(output, y_batch)
                epoch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

                # Display the loss via dynamically updating plot
                losses.append(float(loss))
                minibatches.append(len(losses))
                if notebook:
                    ax.clear()
                    ax.set_xlim([0, n_epochs * len(data_loader)])
                    min_loss = min(0, min(losses))
                    max_loss = max(losses)
                    ax.set_ylim([min_loss, max_loss])
                    ax.set_xlabel('Minibatch #')
                    ax.set_ylabel('Loss')
                    sns.lineplot(x=minibatches, y=losses, ax=ax)
                    display(fig)
                    clear_output(wait=True)
