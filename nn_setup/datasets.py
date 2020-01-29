import torch
from mlutils.data.datasets import H5ArraySet, AttributeHandler, AttributeTransformer
from mlutils.data.transforms import StaticTransform


class LabeledImageSet(H5ArraySet):
    def __init__(self, filename, *data_keys, transforms=None, cache_raw=False, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.cache_raw = cache_raw
        self.last_raw = None
        self.stats_source = stats_source if stats_source is not None else "all"

    def __getitem__(self, item):
        x = self.data_point(*(self._fid[g][item] for g in self.data_keys))
        for tr in self.transforms:
            assert isinstance(tr, StaticTransform)
            x = tr(x)
        im = x[0]
        resp = torch.cat([x[1], torch.tensor([item]).float().cuda()])
        return im, resp

    @property
    def n_neurons(self):
        return len(self[0].responses)

    @property
    def neurons(self):
        return AttributeTransformer("neurons", self._fid, self.transforms)

    @property
    def info(self):
        return AttributeHandler("item_info", self._fid)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source
        tmp = [np.atleast_1d(self.statistics["{}/{}/mean".format(dk, stats_source)].value) for dk in self.data_keys]
        return self.transform(self.data_point(*tmp))

    def __repr__(self):
        return super().__repr__() + (
            "\n\t[Stats source: {}]".format(self.stats_source) if self.stats_source is not None else ""
        )
