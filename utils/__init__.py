from fuel.transformers import SourcewiseTransformer
import numpy as np


class AnnealingNoise(SourcewiseTransformer):

    """Docstring for AnnealingNoise. """

    def __init__(self, p, seed=None, **kwargs):
        """TODO: to be defined1. """
        super(AnnealingNoise, self).__init__(**kwargs)
        if seed is None:
            seed = 123

        self.rng = np.random.RandomState(seed)

        if "p" in kwargs and "std" in kwargs:
            raise ValueError("Can only infer the parameter of p or std."
                             "got std and p in the kwargs")

        self.p = p
        self.std = kwargs.get("std", None)

    def _source_inference(self):
        pass

    def _update_p(self):
        pass

    def _noise_funtion(self, source_batch):
        self.rng.binomial(p=self.p, n=1, size=())

    def tranform_batch(self, source_batch, _):
        return self._noise_funtion(source_batch)


class Analysis(object):
    pass


class EigneStructure(Analysis):
    pass


class ExperienceReplay():
    pass
