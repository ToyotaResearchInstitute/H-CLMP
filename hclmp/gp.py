"""
This submodule contains code for creating and handling Gaussian Processes.
"""

__author__ = 'Kevin Tran'
__email__ = 'kevin.tran@tri.global'

import os
from textwrap import dedent

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
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
                 flatten: bool = False,
                 device: torch.device = None,
                 ):
        """
        Initialize this dataset

        Args:
            model (hclmp.model.Hclmp): Instance of a trained H-CMLP model

            comp_dataset (hclmp.graph_encoder.CompositionData): The dataset
            used to train the given model

            flatten (bool): Whether to flatten the 2-D array of penultimate
            values into a 1-D vector

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GP.
        """

        self.model = model
        self.comp_dataset = comp_dataset
        self.flatten = flatten

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

        if self.flatten:
            self.pen_values = self.pen_values.flatten(start_dim=-2)


class SparseGP(gpytorch.models.ExactGP):
    """
    A Gaussian Process meant to be trained in batches using a set of inducing
    points for the covariance rather the entire training set.
    """

    def __init__(self,
                 train_inputs: torch.tensor,
                 train_targets: torch.tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 mean_module: gpytorch.means.Mean,
                 base_covar_module: gpytorch.kernels.Kernel,
                 num_inducing_points: int = 512,
                 device: torch.device = None,
                 ):
        """

        Args:
            train_inputs (torch.tensor): The features to initialize this GP

            train_targets (torch.tensor): The labels to initialize this GP

            likelihood (gpytorch.likelihoods.Likelihood): The likelihood object
            to train

            mean_module (gpytorch.means.Mean): The mean object to use and train

            base_covar_module (gpytorch.kernels.Kernel): The kernel to use and train

            num_inducing_points (int): The number of inducing points to use

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GPU.
        """

        super().__init__(train_inputs=train_inputs,
                         train_targets=train_targets,
                         likelihood=likelihood)

        self.device = device
        if self.device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                           else torch.device('cpu'))

        self.mean_module = mean_module.to(self.device)
        self.base_covar_module = base_covar_module.to(self.device)

        sample_indices = np.random.choice(
            list(range(len(train_inputs))),
            size=num_inducing_points,
            replace=False,
        )
        self.inducing_points = train_inputs[sample_indices].to(self.device)

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel=self.base_covar_module,
            inducing_points=self.inducing_points,
            likelihood=likelihood,
        ).to(self.device)

    def forward(self,
                inputs: torch.tensor,
                ) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(inputs).to(self.device)
        covar = self.covar_module(inputs).to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def train_on_data(self,
                      train_inputs: torch.tensor,
                      train_targets: torch.tensor,
                      n_epochs: int = 20,
                      lr: float = 0.1,
                      device: torch.device = None,
                      ):
        """
        Train this GP on the input data given during `__init__`

        Args:
            train_inputs (torch.tensor): The features to train this GP

            train_targets (torch.tensor): The labels to train this GP

            n_epochs (int): The number of epochs that should be used to train
            these models

            lr (float): The learning rate to use during training

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses the device given during `__init__`

        Yields:
            float: The loss after each training epoch
        """

        device = device if device is not None else self.device

        # Initialize training
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self).to(device)

        epoch_iter = tqdm(range(n_epochs), total=n_epochs, desc='Epochs')
        for epoch in epoch_iter:

            # Train on this one batch
            optimizer.zero_grad()
            output = self(train_inputs)
            loss = -mll(output, train_targets)
            epoch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

            yield float(loss)

    def predict(self, inputs: torch.tensor) -> (np.ndarray, np.ndarray):
        """
        Make predictions using the model

        Args:
            inputs (torch.tensor): The features to use to make predictions

        Returns:
            (np.ndarray, np.ndarray): The mean and standard error predictions,
            respectively
        """

        self.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self(inputs)
            means = preds.mean.cpu().numpy().squeeze()
            stddevs = preds.stddev.cpu().numpy().squeeze()

        return np.array(means), np.array(stddevs)


class ParallelSparseGPs:
    """
    This class is meant to manage sparse GPs in parallel and independent of
    each other. There should be no information sharing between them. Thus, this
    class is created only for ease-of-use.
    """

    CHECKPOINT_NAME = 'pgp_chkpt.pth.tar'

    def __init__(self,
                 train_inputs: torch.tensor,
                 train_targets: torch.tensor,
                 num_inducing_points: int = 512,
                 device: torch.device = None,
                 ):
        """

        Args:
            train_inputs (torch.tensor): The features to train on

            train_targets (torch.tensor): The labels to train on

            num_inducing_points (int): The number of inducing points to use

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GPU.
        """

        self.train_inputs = train_inputs
        self.train_targets = train_targets

        self.device = device
        if self.device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                           else torch.device('cpu'))

        n_models = train_targets.shape[-1]
        self.gps = []

        for idx in range(n_models):
            inputs = train_inputs[:, idx, :]
            outputs = train_targets[:, idx]

            # Outputs are transmittance values. We can add a mean prior based
            # on training data, and we can also add a constraint between [0, 1]
            mean_prior = gpytorch.priors.NormalPrior(
                loc=outputs.mean(), scale=outputs.std())
            mean_constraint = gpytorch.constraints.constraints.Interval(
                lower_bound=0, upper_bound=1)
            mean_module = gpytorch.means.ConstantMean(
                constant_prior=mean_prior,
                constant_constraint=mean_constraint,
            )

            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=0.5)
            )

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            # Set the noise to be constant (i.e., don't train on the noise).
            # Some small value, but don't make it too small or numerical
            # performance will suffer.
            # likelihood.noise = 1e-2
            # likelihood.raw_noise.requires_grad_(False)

            gp = hclmp.gp.SparseGP(
                train_inputs=inputs,
                train_targets=outputs,
                likelihood=likelihood,
                mean_module=mean_module,
                base_covar_module=covar_module,
                num_inducing_points=num_inducing_points,
                device=self.device
            )
            self.gps.append(gp)

    def train_gps(self,
                  train_inputs: torch.tensor,
                  train_targets: torch.tensor,
                  n_epochs: int = 20,
                  lr: float = 0.1,
                  device: torch.device = None,
                  ):
        """
        Train all of the GPs

        Args:
            train_inputs (torch.tensor): The features to train on

            train_targets (torch.tensor): The labels to train on

            n_epochs (int): The number of epochs that should be used to train
            these models

            lr (float): The learning rate to use during training

            device (torch.device): The device (e.g., CPU or GPU) that should be
            used during training. If `None` will default to cuda (if
            available).
        """

        # If we're in a notebook, we'll use a dynamic plot of the loss.  But if
        # we're in a normal python thread, then we'll just use tqdm and print
        # statements.
        notebook = hclmp.core.is_notebook()
        if notebook:
            fig = plt.figure()
            ax = fig.subplots(1, 1)
            from IPython.display import display, clear_output

        for gp_idx, gp in enumerate(self.gps):
            train_inputs_slice = train_inputs[:, gp_idx, :]
            train_targets_slice = train_targets[:, gp_idx]

            epochs = []
            losses = []
            with gpytorch.settings.cholesky_jitter(1e-3):
                for loss in gp.train_on_data(
                        train_inputs=train_inputs_slice,
                        train_targets=train_targets_slice,
                        n_epochs=n_epochs,
                        lr=lr,
                        device=device,
                 ):
                    losses.append(loss)
                    epoch = epochs[-1] + 1 if epochs else 0
                    epochs.append(epoch)

                    if notebook:
                        import seaborn as sns
                        ax.clear()
                        ax.set_xlim([0, n_epochs])
                        min_loss = min(0, min(losses))
                        max_loss = max(losses)
                        ax.set_ylim([min_loss, max_loss])
                        ax.set_xlabel('Epoch #')
                        ax.set_ylabel('Loss')
                        title = f'Training GP {gp_idx}+1 of {len(self.gps)}'
                        ax.set_title(title)
                        sns.lineplot(x=epochs, y=losses, ax=ax)
                        display(fig)
                        clear_output(wait=True)

    def save_state_dict(self):
        """
        Standardized way of saving the state dictionaries of the various
        parallel GPs
        """

        checkpoint = {}
        for idx, gp in enumerate(self.gps):
            checkpoint[idx] = gp.state_dict()
        torch.save(checkpoint, self.CHECKPOINT_NAME)

    def load_state_dict(self):
        """
        Standardized way of loading the state dictionaries we saved via the
        `save_state_dicts` method
        """

        checkpoint = torch.load(self.CHECKPOINT_NAME)
        for idx, gp in enumerate(self.gps):
            gp.load_state_dict(checkpoint[idx])

    def predict(self, inputs: torch.tensor) -> (np.ndarray, np.ndarray):
        """
        Make predictions on the models

        Args:
            inputs (torch.tensor): The features to use to make predictions

        Returns:
            (np.ndarray, np.ndarray): The mean and standard error predictions,
            respectively
        """

        means, stddevs = [], []
        with torch.no_grad():
            for gp_idx, gp in enumerate(self.gps):
                inputs_slice = inputs[:, gp_idx, :]

                means_slice, stddevs_slice = gp.predict(inputs_slice)
                means.append(means_slice)
                stddevs.append(stddevs_slice)

        means = np.vstack(means).transpose()
        stddevs = np.vstack(stddevs).transpose()
        return means, stddevs


class SVGP(gpytorch.models.ApproximateGP):
    """
    Sparse, variational Gaussian process.
    """

    def __init__(self,
                 channel_slice: int,
                 inducing_points: torch.tensor,
                 device: torch.device = None,
                 ):
        """
        Initialize this multi-objective SVGP

        Args:
            channel_slice (int): This GP is meant to handle slices of 3D inputs.
            This argument governs which slice to take. The first dimension of
            the input corresponds to observations; the second dimension
            corresponds to channels; and the third dimension corresponds to
            features. Yes, we know this is a bad practice. We've made a
            conscious decision to take on this technical debt.

            inducing_points (torch.tensor): The inducing points to use

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GPU.
        """

        self.channel_slice = channel_slice
        self.inducing_points = inducing_points[:, self.channel_slice, :]
        self.num_inducing_points = self.inducing_points.size(0)
        self.device = device
        var_strat = self._create_var_strat()

        super().__init__(var_strat)

        self._create_mean()
        self._create_covar()
        self._create_likelihood()

        if self.device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                           else torch.device('cpu'))

    def _create_var_strat(self):
        """
        Creates a variational strategy suitable for a multi-task SVGP

        Returns:
            gpytorch.variational.VariationalStrategy: The variational strategy
            to use for this model
        """

        var_dist = gpytorch.variational.CholeskyVariationalDistribution
        variational_distribution = var_dist(
            num_inducing_points=self.num_inducing_points,
        ).to(self.device)

        var_strat = gpytorch.variational.VariationalStrategy
        variational_strategy = var_strat(
            model=self,
            inducing_points=self.inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True,
        ).to(self.device)
        return variational_strategy

    def _create_mean(self):
        """
        Creates the mean module to use for this model
        """

        mean_prior = gpytorch.priors.NormalPrior(
            loc=0.77, scale=0.16)

        mean_constraint = gpytorch.constraints.constraints.Interval(
            lower_bound=0, upper_bound=1)

        self.mean_module = gpytorch.means.ConstantMean(
            constant_prior=mean_prior,
            constant_constraint=mean_constraint,
        )

    def _create_covar(self):
        """
        Creates the covariance module to use for this model
        """

        # prior_class = gpytorch.priors.torch_priors.MultivariateNormalPrior
        # lengthscale_prior = prior_class(loc=0)

        # lengthscale_constraint = gpytorch.constraints.constraints.Interval(
        #     lower_bound=0, upper_bound=10)

        kernel = gpytorch.kernels.MaternKernel(
            nu=0.5,
            # lengthscale_prior=lengthscale_prior,
            # lengthscale_constraint=lengthscale_constraint,
        ).to(self.device)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel,
        ).to(self.device)

    def _create_likelihood(self):
        """
        Creates the likelihood object to use for this model
        """

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        # Lock the noise to some small value
        # likelihood.noise = 1e-4
        # likelihood.raw_noise.requires_grad_(False)

        self.likelihood = likelihood

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_on_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs: int = 20,
        lr: float = 0.1,
        device: torch.device = None
    ):
        """
        Train this SVGP

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader
            containing the training dataset to use

            n_epochs (int): The number of epochs that should be used to train
            this model

            device (torch.device): The device (e.g., CPU or GPU) that should be
            used during training. If `None` will default to cuda (if
            available).
        """

        # Auto-detection of the device to use (if necessary)
        if device is None:
            device = self.device

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
        mll = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood,
                                            model=self,
                                            num_data=len(data_loader.dataset))
        losses = []
        minibatches = []

        epoch_iter = tqdm(range(n_epochs), total=n_epochs, desc='Epochs')
        for epoch in epoch_iter:

            # Divide training of each epoch into batches
            minibatch_iter = tqdm(data_loader, desc='Minibatch')
            for x_batch, y_batch in minibatch_iter:
                x_batch = x_batch.to(device)[:, self.channel_slice, :]
                y_batch = y_batch.to(device)[:, self.channel_slice]

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
                    import seaborn as sns
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

    def predict(self,
                data_loader: torch.utils.data.DataLoader,
                ) -> (np.ndarray, np.ndarray):
        """
        Make predictions on the trained model

        Args:
            data_loader (torch.utils.data.DataLoader): The data to make
            predictions on

        Returns:
            (np.ndarray, np.ndarray): The mean and standard error predictions,
            respectively
        """

        self.eval()
        self.likelihood.eval()

        means, stddevs = [], []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)[:, self.channel_slice, :]
                y_batch = y_batch.to(self.device)
                preds = self(x_batch)
                means.extend(preds.mean.cpu().numpy().squeeze())
                stddevs.extend(preds.stddev.cpu().numpy().squeeze())

        return np.array(means), np.array(stddevs)

    def save_state(self, file_name: str):
        """
        Saves the state of this model

        Args:
            file_name (str): The name of the file to save the state to
        """

        state_dict = self.state_dict()
        torch.save(obj=state_dict, f=file_name)

    def load_state(self, file_name: str):
        """
        Sets the state of this model using a saved state dictionary

        Args:
            file_name (str): The name of the file to load the state from
        """

        state_dict = torch.load(f=file_name)
        self.load_state_dict(state_dict)


class ParallelSVGP:
    """
    This class is meant to manage sparse variational GPs in parallel and
    independent of each other. There should be no information sharing between
    them. Thus, this class is created only for ease-of-use.
    """

    CHECKPOINT_FILE_EXT = '.pth.tar'
    CHECKPOINT_NAME = 'gp_%s_cp'

    def __init__(self,
                 num_models: int,
                 inducing_points: torch.tensor,
                 device: torch.device = None,
                 ):
        """

        Args:
            num_models (int): The number of models you want to be using in
            parallel

            inducing_points (torch.tensor): The number of inducing points to use

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GPU.
        """

        self.device = device
        if self.device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                           else torch.device('cpu'))

        self.gps = []
        for idx in range(num_models):
            gp = hclmp.gp.SVGP(
                channel_slice=idx,
                inducing_points=inducing_points,
                device=self.device,
            )
            self.gps.append(gp)

    def train_gps(self,
                  data_loader: torch.utils.data.DataLoader,
                  n_epochs: int = 20,
                  lr: float = 0.1,
                  device: torch.device = None,
                  ):
        """
        Train all of the GPs

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader
            containing the training dataset to use

            n_epochs (int): The number of epochs that should be used to train
            these models

            lr (float): The learning rate to use during training

            device (torch.device): The device (e.g., CPU or GPU) that should be
            used during training. If `None` will default to cuda (if
            available).
        """

        for gp_idx, gp in enumerate(self.gps):
            gp.train_on_loader(
                data_loader=data_loader,
                n_epochs=n_epochs,
                lr=lr,
                device=device,
             )

    def predict(self,
                data_loader: torch.utils.data.DataLoader,
                ) -> (np.ndarray, np.ndarray):
        """
        Make predictions on the models

        Args:
            data_loader (torch.utils.data.DataLoader): The data to make
            predictions on

        Returns:
            (np.ndarray, np.ndarray): The mean and standard error predictions,
            respectively
        """

        means, stddevs = [], []
        for gp_idx, gp in enumerate(self.gps):
            means_slice, stddevs_slice = gp.predict(data_loader=data_loader)
            means.append(means_slice)
            stddevs.append(stddevs_slice)

        means = np.vstack(means).transpose()
        stddevs = np.vstack(stddevs).transpose()
        return means, stddevs

    def save_state_dicts(self, folder_name: str):
        """
        Standardized way of saving the state dictionaries of the various
        parallel GPs

        Args:
            folder_name (str): The folder to save the checkpoint files in
        """

        # Make the folder of checkpoints
        os.makedirs(folder_name, exist_ok=True)

        # Delete old checkpoints to prevent checkpoint file mangling
        for file_name in os.listdir(folder_name):
            if file_name.endswith(self.CHECKPOINT_FILE_EXT):
                file_path = os.path.join(folder_name, file_name)
                os.remove(file_path)

        # Save the new checkpoints
        for idx, gp in enumerate(self.gps):
            checkpoint_path = self._get_checkpoint_path(folder_name, idx)
            gp.save_state(checkpoint_path)

    def load_state_dicts(self, folder_name: str):
        """
        Standardized way of loading the state dictionaries we saved via the
        `save_state_dicts` method

        Args:
            folder_name (str): The folder to load the checkpoint files from
        """

        for idx, gp in enumerate(self.gps):
            checkpoint_path = self._get_checkpoint_path(folder_name, idx)
            gp.load_state(checkpoint_path)

    def _get_checkpoint_path(self, folder_name: str, idx: int):
        """
        Gets the full path to a checkpoint file

        Args:
            folder_name (str): The folder to load the checkpoint files from

            idx (int): The slice number/index of the SVGP you want to get the
            checkpoint for

        Returns:
            str: The path to the checkpoint file
        """

        cp_name = (self.CHECKPOINT_NAME % idx) + self.CHECKPOINT_FILE_EXT
        cp_path = os.path.join(folder_name, cp_name)
        return cp_path


class MultiTaskSVGP(gpytorch.models.ApproximateGP):
    """
    Multi-task/objective, sparse, variational Gaussian process. This code is
    modified from the `MultitaskGPModel` object in GpyTorch's example notebook,
    `SVGP_Multitask_GP_Regression.ipynb`.
    """

    def __init__(self,
                 inducing_points: torch.tensor,
                 num_tasks: int,
                 num_features: int,
                 num_latents: int = 1,
                 device: torch.device = None,
                 ):
        """
        Initialize this multi-objective SVGP

        Args:
            inducing_points (torch.tensor): The inducing points to use. The
            shape should be (num_latents, num_inducing_points, num_features)

            num_tasks (int): The number of tasks/objectives for this model to
            be learning/predicting

            num_features (int): The number of inputs/features for this GP

            num_latents (int): This model assumes that each of the
            tasks/objectives/outputs are linear combinations of some latent GP
            functions. This argument allows you to choose the number of these
            latent functions.

            device (torch.device): The device that you want this model to run
            on. If `None`, then uses any available GPU.
        """

        self.inducing_points = inducing_points
        self.num_inducing_points = self.inducing_points.size(1)
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.num_latents = num_latents
        self.device = device

        self._parse_init_args()
        var_strat = self._create_var_strat()

        super().__init__(var_strat)

        self._create_covar()
        self._create_likelihood()

    def _parse_init_args(self):
        """
        Argument handling and parsing of the arguments for `__init__`. You
        really shouldn't be calling this method outside `__init__`.
        """

        # Argument parsing
        # if self.inducing_points.size(0) != self.num_tasks:  # for ind multitask
        if self.inducing_points.size(0) != self.num_latents:  # for LMC
            msg = dedent(
                f"""
                The first dimension of the inducing points
                ({self.inducing_points.size(0)}) does not match the number of
                latent functions ({self.num_latents})
                """)
            raise ValueError(msg)

        if self.inducing_points.size(2) != self.num_features:
            msg = dedent(
                f"""
                The third dimension of the inducing points
                ({self.inducing_points.size(2)}) does not match the number of
                features ({self.num_features})
                """)
            raise ValueError(msg)

        if self.device is None:
            self.device = (torch.device('cuda') if torch.cuda.is_available()
                           else torch.device('cpu'))

    def _create_var_strat(self):
        """
        Creates a variational strategy suitable for a multi-task SVGP

        Returns:
            gpytorch.variational.VariationalStrategy: The variational strategy
            to use for this model
        """

        # We have to mark the CholeskyVariationalDistribution as batch so that
        # we learn a variational distribution for each task
        var_dist = gpytorch.variational.CholeskyVariationalDistribution
        variational_distribution = var_dist(
            num_inducing_points=self.num_inducing_points,
            batch_shape=torch.Size([self.num_latents]),
        ).to(self.device)

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than
        # a batch output
        base_var_strat = gpytorch.variational.VariationalStrategy
        base_variational_strategy = base_var_strat(
            model=self,
            inducing_points=self.inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True,
        ).to(self.device)

        # var_strat = gpytorch.variational.IndependentMultitaskVariationalStrategy(
        var_strat = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=self.num_tasks,
            num_latents=self.num_latents,
            latent_dim=-1,
            # task_dim=-1,  # for independent multitask var strat
        ).to(self.device)
        return var_strat

    def _create_covar(self):
        """
        Creates a the covariance module to use for this model
        """

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        mean_prior = gpytorch.priors.NormalPrior(
            loc=0.77, scale=0.16)
        mean_constraint = gpytorch.constraints.constraints.Interval(
            lower_bound=0, upper_bound=1)
        self.mean_module = gpytorch.means.ConstantMean(
            constant_prior=mean_prior,
            constant_constraint=mean_constraint,
            batch_shape=torch.Size([self.num_latents]),
        )

        # prior_class = gpytorch.priors.torch_priors.MultivariateNormalPrior
        # lengthscale_prior = prior_class(loc=0)

        # lengthscale_constraint = gpytorch.constraints.constraints.Interval(
        #     lower_bound=0, upper_bound=10)

        kernel = gpytorch.kernels.MaternKernel(
            batch_shape=torch.Size([self.num_latents]),
            nu=0.5,
            # lengthscale_prior=lengthscale_prior,
            # lengthscale_constraint=lengthscale_constraint,
        ).to(self.device)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel,
            batch_shape=torch.Size([self.num_latents]),
        ).to(self.device)

    def _create_likelihood(self):
        """
        Create a likelihood object for this model
        """

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.num_tasks,
        ).to(self.device)

        # Set the noise to be constant (i.e., don't train on the noise).
        # It should be some small value, but don't make it too small or
        # numerical performance will suffer.
        # likelihood.noise = 1e-4
        # likelihood.raw_noise.requires_grad_(False)

        self.likelihood = likelihood

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x).to(self.device)
        covar_x = self.covar_module(x).to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_on_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_epochs: int = 20,
        lr: float = 0.1,
        device: torch.device = None
    ):
        """
        Train this multitask SVGP

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader
            containing the training dataset to use

            n_epochs (int): The number of epochs that should be used to train
            this model

            lr (float): The learning rate to use during training

            device (torch.device): The device (e.g., CPU or GPU) that should be
            used during training. If `None` will default to cuda (if
            available).
        """

        # Auto-detection of the device to use (if necessary)
        if device is None:
            device = self.device

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
        mll = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood,
                                            model=self,
                                            num_data=len(data_loader.dataset))
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
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
                    import seaborn as sns
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

    def predict(self,
                data_loader: torch.utils.data.DataLoader,
                ) -> (np.ndarray, np.ndarray):
        """
        Make predictions on the trained model

        Args:
            data_loader (torch.utils.data.DataLoader): The data to make
            predictions on

        Returns:
            (np.ndarray, np.ndarray): The mean and standard error predictions,
            respectively
        """

        self.eval()
        self.likelihood.eval()

        means, stddevs = [], []
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                preds = self(x_batch)
                means.extend(preds.mean.cpu().numpy().squeeze())
                stddevs.extend(preds.stddev.cpu().numpy().squeeze())

        return np.array(means), np.array(stddevs)
