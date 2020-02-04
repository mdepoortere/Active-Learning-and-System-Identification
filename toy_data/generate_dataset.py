import h5py
from mlutils.data.datasets import StaticImageSet
from synthetic_data import gen_gabor_RF
from utils import compute_activity_simple


rf_shape = (13, 13)
n_neurons = 300
random_seed = 5


def main():
    file = h5py.File('/notebooks/data/static20892-3-14-preproc0.h5', "r")
    dat = StaticImageSet('/notebooks/data/static20892-3-14-preproc0.h5', 'images', 'responses')
    img_shape = dat.img_shape[2:]
    gabor_rf = gen_gabor_RF(img_shape, rf_shape, n=n_neurons, seed=random_seed)
    images= dat[()].images
    images = images.reshape(6000, 36, 64)
    responses = compute_activity_simple(images/255, gabor_rf)
    data_file = h5py.File("toy_dataset.hdf5", "w")

    im_set = data_file.create_dataset('images', data=dat[()].images)
    response_set = data_file.create_dataset('responses', data=responses)
    tier_set = data_file.create_dataset('tiers', data=file['tiers'])

    data_file.close()


if __name__ == "__main__":
    main()