import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision.transforms.functional import to_grayscale

from toy_data.utils import gabor_fn, deformed_image


def gen_gabor_stim(img_size, n=1, seed=None, normalize=True):
    """Returns gabor-like images.
    Args:
        img_size (tuple of integers or int): Height and width of the images.
        n (int): Number of images.
        seed (int): Random seed to generates different gabor characteristics.
    Returns:
        Numpy array: Images.
    """

    np.random.seed(seed)
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    angles = np.random.choice(np.linspace(0, np.pi, 179), size=(n,))
    sigmas = np.random.choice(np.arange(max(img_size) // 11, max(img_size) // 3), size=(n,))
    lambds = np.random.choice(np.arange(max(img_size) // 10, min(img_size)), size=(n,))
    psis = np.random.choice(np.linspace(0, np.pi, 179), size=(n,))
    locs_y = np.random.randint(-img_size[0]//2, img_size[0]//2, size=(n, 2))
    locs_x = np.random.randint(-img_size[1]//2, img_size[1]//2, size=(n, 2))
    locs = np.hstack((locs_x, locs_y))

    imgs = np.concatenate([np.expand_dims(gabor_fn(angles[idx],
                                                   sigma=sigmas[idx],
                                                   Lambda=lambds[idx],
                                                   psi=psis[idx],
                                                   center=locs[idx],
                                                   size=img_size,
                                                   normalize=normalize), 0)
                                         for idx in range(n)], axis=0)
    return imgs


def gen_deformed_stim(img_size, n=1, seed=None):
    """Returns deformed gabor-like images.
    Args:
        img_size (tuple of integers or int): Height and width of the images.
        n (int): Number of images.
        seed (int): Random seed to generates different gabor characteristics.
    Returns:
        Numpy array: Images.
    """

    np.random.seed(seed)
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    angles = np.random.choice(np.linspace(0, np.pi, 179), size=(n,))
    locs_y = np.random.randint(-img_size[0]//3, img_size[0]//3, size=(n, 1))
    locs_x = np.random.randint(-img_size[1]//3, img_size[1]//3, size=(n, 1))
    locs = np.hstack((locs_x, locs_y))
    sigmas = np.random.choice(np.arange(2, 11), size=(n,))

    imgs = np.concatenate([np.expand_dims(deformed_image(angles[idx],
                                                         sigma=sigmas[idx],
                                                         center=locs[idx],
                                                         size=img_size), 0)
                                         for idx in range(n)], axis=0)
    return imgs


def CIFAR10(img_size, n=1, seed=None, normalize=True):
    """Returns grayscale CIFAR10 images.
    Args:
        img_size (tuple of integers or int): Height and width of the images.
        n (int): Number of images.
        seed (int): Random seed to generates different gabor characteristics.
        normalize (bool): Whether to normalize the images.
    Returns:
        Numpy array: Images.
    """

    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    train_dataset = datasets.CIFAR10('../data', train=True, download=True)

    imgs = []
    for img_np in train_dataset.data:
        img = to_grayscale(Image.fromarray(img_np))
        img = img.resize(img_size[::-1], Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        if normalize:
            img = img - img.min()
            img = img / img.max()
        imgs.append(img[None, :, :])

    np.random.seed(seed)
    imgs = np.concatenate(imgs, 0)
    idx = np.random.choice(np.arange(imgs.shape[0]), size=(n), replace=False)
    imgs = imgs[idx]

    return imgs


def gen_gabor_RF(img_size, rf_size, n=1, gabor_type='even', seed=None, normalize=True):
    """Returns gabor receptive fields.
    Args:
        img_size (tuple): Height and width of the image.
        rf_size (tuple): Height and width of the receptive field.
        n (int): number of neurons.
        gabor_type (string): Choose the type of gabor (even or odd).
        seed (type): random seed to choose the position of the receptive fields.
    Returns:
        Numpy array: Receptive fields.
    """

    np.random.seed(seed)
    locs_y = np.random.randint(-(img_size[0]-1.5*rf_size[0])//2, (img_size[0]-1.5*rf_size[0])//2, size=(n, 1))
    locs_x = np.random.randint(-(img_size[1]-1.5*rf_size[1])//2, (img_size[1]-1.5*rf_size[1])//2, size=(n, 1))
    locs = np.hstack((locs_x, locs_y))
    angles = np.linspace(0, np.pi, n)

    if gabor_type == 'even':
        RFs = np.concatenate([np.expand_dims(gabor_fn(angles[idx], center=loc, size=img_size, normalize=normalize), 0) for idx, loc in enumerate(locs)], axis=0)
    elif gabor_type == 'odd':
        RFs = np.concatenate([np.expand_dims(gabor_fn(angles[idx], psi=0, sigma=3.5, gamma=1.2, center=loc, size=img_size, normalize=normalize), 0) for idx, loc in enumerate(locs)], axis=0)

    return RFs