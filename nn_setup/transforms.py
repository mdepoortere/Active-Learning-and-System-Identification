import torch
from mlutils.data.transforms import MovieTransform, StaticTransform, Invertible


class Normalized(MovieTransform, StaticTransform, Invertible):
    def __init__(self, train_idx, responses, cuda=False):
        self.train_idx = train_idx
        self.cuda = cuda
        self.std = torch.from_numpy(responses[self.train_idx, :].std(axis=0)).float()
        if self.cuda:
            self.std = self.std.cuda()
        self.transforms, self.itransforms = {}, {}
        self.transforms["responses"] = lambda x: x / self.std
        self.itransforms["responses"] = lambda x: x * self.std
        self.transforms['images'] = lambda x: x
        self.itransforms['images'] = lambda x: x

    def inv(self, x):
        return x.__class__(
            **{k: (self.itransforms[k](v))for k, v in zip(x._fields, x)}
        )

    def __call__(self, x):
        return x.__class__(
            **{k: (self.transforms[k](v)) for k, v in zip(x._fields, x)}
        )

