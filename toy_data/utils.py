""" From Mohammad Bashiri"""
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler



def gen_poisson_spike(spike_prob, iter_n=1):
    """Generates event (presence or absence) for a poisson process.
    Args:
        spike_prob (Numpy array): Probability of spike (in an iteration). Dimension is Nx1.
        iter_n (int): Number of iteratios, default is 1.
    Returns:
        array of boolean: Specifies whether at a specific time step there was an occurance of an event (i.e., spike) or not.
    """

    # TODO: change to exponential distribution
    spike_probs = spike_prob.repeat(iter_n, axis=1)
    random_vals = np.random.rand(*spike_probs.shape)
    return spike_probs > random_vals


def gabor_fn(theta, sigma=2, Lambda=10, psi=np.pi/2, gamma=.8, center=(0, 0), size=(28, 28), normalize=True):
    """Returns a gabor filter.
    Args:
        theta (float): Orientation of the sinusoid (in ratian).
        sigma (float): std deviation of the Gaussian.
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        size (tuple of integers): Image height and width.
        normalize (bool): Whether to normalize the entries. This is computed by
            dividing each entry by the root sum squared of the whole image.
    Returns:
        2D Numpy array: A gabor filter.
    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # shift the positon
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)

    if normalize:
        # root sum squared
        gb /= np.sqrt(np.sum(gb ** 2))
        # make sure the sum is equal to zero
        # gb[gb > 0] = gb[gb > 0] * (np.abs(gb[gb < 0].sum()) / gb[gb > 0].sum())
        gb -= gb.mean()

    return gb


def grating_image(theta, Lambda=10, psi=np.pi/2, size=(28, 28), normalize=True):
    """Returns a grating image.
    Args:
        theta (float): Orientation of the sinusoid (in ratian).
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        size (tuple of integers): Image height and width.
        normalize (bool): Whether to normalize the entries. This is computed by
            dividing each entry by the root sum squared of the whole image.
    Returns:
        2D Numpy array: A grating image.
    """

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.cos(2 * np.pi / Lambda * x_theta + psi)

    if normalize: # root sum squared
        gb /= np.sqrt(np.sum(gb ** 2))

    return gb


def gaussian2d(theta=0, sigma=2, gamma=1, size=(28, 28), center=(0, 0), normalize=True):
    """Return a 2D Gaussian filter.
    Args:
        theta (float): Orientation of the sinusoid (in ratian).
        sigma (float): std deviation of the Gaussian.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        size (tuple of integers): Image height and width.
        normalize (bool): Whether to normalize the entries. This is computed by
            dividing each entry by the root sum squared of the whole image.
    Returns:
        2D Numpy array: A 2D Gaussian filter.
    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # shift the position
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))

    if normalize: # root sum squared
        gaussian /= np.sqrt(np.sum(gaussian ** 2))


    return gaussian


def deformed_image(theta, sigma=5, gamma=1, center=(0, 0), size=(28, 28), normalize=True):
    """Return a 2D deformed gabor.
    Args:
        theta (float): Orientation of the sinusoid (in ratian).
        sigma (float): std deviation of the Gaussian.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        size (tuple of integers): Image height and width.
        normalize (bool): Whether to normalize the entries. This is computed by
            dividing each entry by the root sum squared of the whole image.
    Returns:
        2D Numpy array: A 2D deformed grating.
    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    bounds = (-7, -7, 7, 7)
    xvals = np.linspace(bounds[0], bounds[2], size[0])
    yvals = np.linspace(bounds[3], bounds[1], size[1])

    y, x = np.meshgrid(xvals, yvals)

    y -= center[0]
    x += center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(x_theta/2 * y_theta/2)

    if normalize: # root sum squared
        gb /= np.sqrt(np.sum(gb ** 2))

    return gb


def dft(img):
    """Return Fourier transform of an image.
    Args:
        img (2D Numpy array): Image to compute the Fourier transform for.
    Returns:
        2D Numpy array: Fourier transform of the input image.
    """

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) # compute the amplitude
    phase = np.arctan(fshift.imag / fshift.real) # compute the phase

    return magnitude, phase

def idft(z):
    """Retun image for a given image in Fourier space.
    Args:
        z (2D Numpy array): Fourier representation of an image.
    Returns:
        2D Numpy array: Image for the input Fourier representation.
    """

    return np.fft.ifft2(np.fft.ifftshift(z)).real


def compute_activity_simple(imgs, RFs):
    """Returns firing rate of simple neurons given their receptive fiels and set of images.
    Args:
        imgs (Numpy array): Input images.
        RFs (Numpy array): Receptive fiels.
    Returns:
        Numpy array: Firing rates.
    """

    rates_all = []
    for ind, img in enumerate(imgs):
        img_filtered = np.expand_dims(img, 0) * RFs
        rates = np.exp(img_filtered.sum(axis=(1, 2)).reshape(-1, 1))
        rates_all.append(rates)

    return np.concatenate(rates_all, axis=1).T


def compute_activity_complex(imgs, even_RFs, odd_RFs):
    """Returns firing rate of complex neurons given their even and odd receptive fiels and set of images.
    Args:
        imgs (Numpy array): Input images.
        even_RFs (Numpy array): even receptive fiels.
        odd_RFs (Numpy array): odd receptive fiels.
    Returns:
        Numpy array: Firing rates.
    """

    rates_all = []
    for ind, img in enumerate(imgs):

        img_filtered_even = np.expand_dims(img, 0) * even_RFs
        img_filtered_odd = np.expand_dims(img, 0) * odd_RFs
        energy = np.sqrt(img_filtered_odd.sum(axis=(1, 2)) ** 2 + img_filtered_even.sum(axis=(1, 2)) ** 2)

        rates = np.exp(energy.reshape(-1, 1))
        rates_all.append(rates)

    return np.concatenate(rates_all, axis=1).T


def get_dataloader(imgs, targets, batch_size=64, device='cuda'):
    """Returns data generators the given input and output
    Args:
        imgs (Numpy array): Inputs.
        targets (Numpy array): Outputs.
        batch_size (int): Batch size for each quiery of data from DataLoader object.
    Returns:
        DataLoader object: iterable dataloader object.
    """

    data_set = data.TensorDataset(torch.from_numpy(imgs.astype(np.float32)).to(device), torch.from_numpy(targets.astype(np.float32)).to(device))
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

    return data_loader

# TODO: doc
class RepeatsBatchSampler(Sampler):
    def __init__(self, keys, subset_index=None):
        if subset_index is None:
            subset_index = np.arange(len(keys))
        _, inv = np.unique(keys[subset_index], return_inverse=True)
        self.repeat_index = np.unique(inv)
        self.repeat_sets = inv
        self.subset_index = subset_index

    def __iter__(self):
        for u in self.repeat_index:
            yield list(self.subset_index[self.repeat_sets == u])

    def __len__(self):
        return len(self.repeat_index)