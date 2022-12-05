"""
A bunch of scripts to parse and process UV-VIS data of inkjet-printed plates
created by John Gregoire's group.

Copyright Toyota Research Institute 2022
"""

__author__ = 'Kevin Tran'
__email__ = 'kevin.tran@tri.global'


import collections
import pickle
from typing import Generator, Optional

import numpy as np
import pandas as pd
from scipy import constants
from tqdm.auto import tqdm


# Default file names
DATA_CACHE = 'inkjet_data_processed_%i.csv'
ELEMENTS_CACHE = 'elements.pkl'

# Default header labels
COMP_NAME_COL = 'comp_cols'
COMP_ARR_COL = 'comp_arr'
PLATE_ID_COL = 'plate_id'
SAMPLE_NUM_COL = 'sample_no'
ENERGIES_COL = 'energies'
TRANS_COL = 'cumulative_transmittances'


def parse_vector(vec: str) -> np.ndarray:
    """
    We end up saving vectors of data into strings so that we can save them into
    pandas columns. This function will turn that string back into a numpy array.

    Args:
        vec (str): the string-formatted vector of floats

    Returns:
        np.ndarray: the vector in numpy format
    """

    return np.array(str(vec)[1:-1].split(', '))


def get_plate_data(data_file: str,
                   n_bins: int = 20,
                   force: bool = False) -> pd.DataFrame:
    """
    Get all of the elements that are present in the given data file

    Args:
        data_file (str): The file containing all of the data for the plates

        n_bins (int): We will discretize the UV-VIS spectra into bins. Use this
        argument to define the number of bins.

        force (bool): Whether to ignore the cache

    Returns:
        pd.DataFrame: the processed plate data
    """

    cache_name = DATA_CACHE % n_bins

    # EAFP to read the cache, if it's there
    if not force:
        try:
            return pd.read_csv(cache_name)
        except FileNotFoundError:
            pass

    df = parse_inkjet_data(data_file=data_file, n_bins=n_bins)
    df.to_csv(cache_name, index=False)
    return df


def parse_inkjet_data(plates_data: dict,
                      n_bins: int = 20) -> pd.DataFrame:
    """
    Read and parse a file containing UV-VIS data from the given data file

    Args:
        plates_data (dict): The raw dictionary containing plate data

        n_bins (int): We will discretize the UV-VIS spectra into bins. Use this
        argument to define the number of bins.

    Returns:
        pd.DataFrame: The post-processed data
    """

    elements = get_elements(plates_data=plates_data)

    # It's fast to instantiate a data frame once with a dictionary, so we start
    # there
    data = collections.defaultdict(list)

    for plate_id, plate_data in tqdm(plates_data.items(), desc='Plates'):
        sample_iterator = tqdm(enumerate(plate_data[SAMPLE_NUM_COL]),
                               desc=f'Discretizing plate {plate_id}',
                               total=len(plate_data[SAMPLE_NUM_COL]))
        for sample_idx, sample_num in sample_iterator:

            # Grab the composition data for each sample
            compositions = collections.defaultdict(float)
            for ele_idx, element in enumerate(plate_data[COMP_NAME_COL]):
                comp = plate_data[COMP_ARR_COL][sample_idx, ele_idx]
                compositions[element] = float(comp)

            # Get the transmittance
            transmittance_generator = get_aggregated_spectrum(
                plate_data=plate_data, sample_num=sample_num, n_bins=n_bins)
            try:
                energies, transmittances = zip(*list(transmittance_generator))

            # If we get invalid transmittances, then we omit the sample
            except InvalidDataError:
                continue

            # Store the data
            data[PLATE_ID_COL].append(plate_id)
            data[SAMPLE_NUM_COL].append(sample_num)
            for element in elements:
                data[element].append(compositions[element])
            data[ENERGIES_COL].append(energies)
            data[TRANS_COL].append(transmittances)

    df = pd.DataFrame(data)
    return df


def get_elements(elements_file: str = ELEMENTS_CACHE,
                 plates_data: Optional[dict] = None) -> set:
    """
    Get all of the elements that are present in the given data file

    Args:
        elements_file (str): The cache file we try to read the elements from
        (to avoid reading the possibly large data file).

        plates_data (dict): The raw dictionary containing plate data

    Returns:
        set: the elements present in the given data file
    """

    # EAFP to read the cache, if it's there
    try:
        with open(elements_file, 'rb') as pkl:
            return pickle.load(pkl)
    except FileNotFoundError:
        pass

    # If the cache isn't there, then the plate data had better be provided
    if not plates_data:
        msg = (f'No {elements_file} cache file was found. Please provide plate '
               'instead.')
        raise TypeError(msg)

    elements = {element for plate_id, plate in plates_data.items()
                for element in plate[COMP_NAME_COL]}

    with open(elements_file, 'wb') as pkl:
        pickle.dump(elements, pkl)

    return elements


def get_aggregated_spectrum(plate_data: dict,
                            sample_num: str,
                            n_bins: int = 20,
                            ) -> Generator[tuple, None, None]:
    """
    Discretize the data from a plate into a number of bins

    Args:
        plate_data (dict): a dictionary containing the UV-VIS data of an entire plate

        sample_num (str): the sample number in the plate to analyze the spectrum of

        n_bins (int): the number of bins to discretize into

    Yields:
        tuple(float, float): a tuple whose first element is the average energy
        of a bin and whose second element is the transmittance in that bin
    """

    for wls, trans in discretize_spectrum(plate_data=plate_data,
                                          sample_num=sample_num,
                                          n_bins=n_bins):

        # Convert wavelength unis to energies. Since they're inverted,
        # we reverse the order of both the energies and the transmittances
        energies = list(reversed([wavelength_to_energy(wl) for wl in wls]))
        transmittances = list(reversed(trans))
        yield np.average(energies), np.average(transmittances)


def discretize_spectrum(plate_data: dict,
                        sample_num: str,
                        n_bins: int = 20) -> Generator[tuple, None, None]:
    """
    Discretize the data from a plate into a number of bins

    Args:
        plate_data (dict): a dictionary containing the UV-VIS data of an entire plate

        sample_num (str): the sample number in the plate to analyze the spectrum of

        n_bins (int): the number of bins to discretize into

    Yields:
        tuple((np.array, np,array): A tuple of arrays. The first array is a
        chunk of wavelengths, and the second array is the corresponding chunk
        of transmittance values
    """

    sample_idx = plate_data['sample_no'].tolist().index(str(sample_num))

    wavelengths = plate_data['wl']
    transmittances = plate_data['T'][sample_idx, :]

    # If more than 5% of the data are out of valid range, we raise an error
    n_points = len(wavelengths)

    if sum(transmittances < 0) > n_points / 20:
        msg = ('Found too many transmittance values below 0 in sample number '
               f'{sample_num}')
        raise InvalidDataError(msg)

    if sum(transmittances > 1) > n_points / 20:
        msg = ('Found too many transmittance values above 1 in sample number '
               f'{sample_num}')
        raise InvalidDataError(msg)

    # If less than 5% of the data are out of valid range, then clip it to
    # remain between [0, 1] and move on
    trans_clipped = np.clip(transmittances, a_min=0, a_max=1)
    spectrum = list(zip(wavelengths, trans_clipped))
    for chunk in np.array_split(spectrum, indices_or_sections=n_bins):
        wls = chunk[:, 0]
        trans = chunk[:, 1]
        yield wls, trans


def wavelength_to_energy(wavelength: float) -> float:
    """
    Convert a wavelength into an energy, in eV

    Args:
        wavelength (float): the wavelength to use in nm

    Returns:
        float: the energy corresponding to the given wavelength
    """

    wavelength_nm = wavelength
    wavelength_m = wavelength_nm / 1e9

    frequency = constants.lambda2nu(wavelength_m)
    energy_j = constants.Planck * frequency
    energy_ev = energy_j / constants.eV

    return energy_ev


def energy_to_wavelength(energy: float) -> float:
    """
    Convert an energy into its corresponding light wavelength

    Args:
        ev (float): the energy you want to get the wavelength of, in eV

    Returns:
        float: the wavelength, in nm
    """

    energy_ev = energy
    energy_j = energy_ev * constants.eV
    frequency = energy_j / constants.Planck

    wavelength_m = constants.nu2lambda(frequency)
    wavelength_nm = wavelength_m * 1e9
    return wavelength_nm


class InvalidDataError(ValueError):
    """
    Error to be raised when we get some non-sensical data, such as a negative
    reflectance
    """
