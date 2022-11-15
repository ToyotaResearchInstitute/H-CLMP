"""
This module is meant to be used to preprocess raw composition and spectral data
into format(s) ingestible by H-CLMP. The original code was taken from
the Gomes lab https://www.cs.cornell.edu/gomes/udiscoverit/?tag=materials and
then converted into an API by Toyota Research Institute.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import HCLMP.inkjet

DEFAULT_ELEMENTS = {'Ag', 'Bi', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 'Fe',
                    'Gd', 'In', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Nd', 'Ni', 'P',
                    'Pr', 'Rb', 'Sc', 'Sm', 'Ti', 'V', 'W', 'Yb', 'Zn', 'Zr'}
GENERATOR_STATE_DICT = os.path.join(Path(__file__).parent.absolute(), 'embeddings',
                                    'generator_MP2020.pt')


def preprocess(data_file: str,
               elements: set = DEFAULT_ELEMENTS,
               n_bins: int = 20,
               latent_dim: int = 50,
               transfer: bool = False) -> dict:
    """
    Get all of the elements that are present in the given data file

    Args:
        data_file (str): The file containing all of the data for the plates

        elements (set): The elements whose compositions you want to be
        considered as features

        n_bins (int): The number of discretized points in the spectra that you
        want to predict

        latent_dim (int): The number of dimensions for us to have it latent
        space

        transfer (bool): Whether to attempt to transfer generated features from
        WGAN.

    Returns:
        dict: The data in a dictionary format accepted by the H-CLMP model
    """

    # Open the raw data
    with open(data_file, 'rb') as pkl:
        plates_data = pickle.load(pkl)

    df = HCLMP.inkjet.parse_inkjet_data(plates_data=plates_data,
                                        n_bins=n_bins)

    # Read the features and labels
    ele_names = np.array(sorted(elements))
    element_comps = df[ele_names].to_numpy().astype(np.float32)
    # Note:  Here we assume that the spectra are contained as tuples inside DF
    # columns (as opposed to separate columns altogether, which is what
    # conventional H-CLMP does)
    spectra = [list(spectrum) for spectrum in df[HCLMP.inkjet.TRANS_COL]]
    foms = np.array(spectra)

    data_dict = {}
    data_dict['all_element_name'] = ele_names

    # Create (and validate) the generated features
    for idx, (ele_comp, fom) in enumerate(zip(element_comps, foms)):

        assert np.abs(1 - np.sum(ele_comp)) < 1e-5
        nonzero_idx = np.nonzero(ele_comp)
        assert np.abs(1 - np.sum(ele_comp[nonzero_idx])) < 1e-5

        # Store all the relevant data
        data_dict[idx] = {}
        data_dict[idx]['fom'] = fom
        data_dict[idx]['composition_nonzero'] = \
            ele_comp[nonzero_idx] / np.sum(ele_comp[nonzero_idx])
        data_dict[idx]['composition_nonzero_idx'] = nonzero_idx
        data_dict[idx]['nonzero_element_name'] = ele_names[nonzero_idx]
        data_dict[idx]['composition'] = ele_comp / np.sum(ele_comp)

    # Load the cached generative model for transfer learning
    if transfer:
        generator = Generator(label_dim=n_bins,
                              feature_dim=len(elements),
                              latent_dim=latent_dim)
        generator.load_state_dict(torch.load(GENERATOR_STATE_DICT,
                                             map_location=torch.device('cpu')))
        generator.eval()

        # Sample from the generative model to create transferred features
        iterator = tqdm(enumerate(element_comps),
                        desc='Transferring generated features',
                        total=len(df))
        for idx, ele_comp in iterator:
            gen_feat = sample_generator(
                generator,
                1,
                torch.from_numpy(ele_comp).unsqueeze(0)).detach().squeeze().numpy()
            data_dict[idx]['gen_dos_fea'] = gen_feat

    return data_dict


def sample_generator(generator, num_samples, feature):
    generated_data_all = 0
    num_sam = 100
    for i in range(num_sam):
        latent_samples = Variable(generator.sample_latent(num_samples))
        generated_data = generator(torch.cat((feature, latent_samples), dim=1))
        generated_data_all += generated_data
    generated_data = generated_data_all/num_sam
    return generated_data


class Generator(nn.Module):
    def __init__(self, label_dim, feature_dim, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.laten_to_label = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, 256),
            nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(128),
            nn.Linear(128, label_dim),
            # nn.Sigmoid()
        )

    def forward(self, input_data):
        return self.laten_to_label(input_data)

    def sample_latent(self, num_sample):
        return torch.randn((num_sample, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, label_dim, feature_dim):
        super(Discriminator, self).__init__()

        self.label_to_feature = nn.Sequential(
            nn.Linear(label_dim + feature_dim, 256),
            nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward(self, input_data):
        return self.label_to_feature(input_data)
