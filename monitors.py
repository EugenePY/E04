from blocks.extensions.monitoring import MonitoringExtension, SimpleExtension
from blocks.graph import ComputationGraph
import matplotlib.gridspec as gridspec
import logging
import theano
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _plot_images(images, img_size, iter, path=None):
    """
    Image batch is (Batch, flatten(width, length, channel_size))
    """
    img_batch = img_size[0]
    row = int(np.ceil(img_batch / 5.))
    fig = plt.figure(figsize=(row, 5))
    gs = gridspec.GridSpec(row, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
<<<<<<< HEAD
        plt.imshow(sample.reshape(img_size[1], img_size[2]),
                   cmap='Greys_r')
=======

        sample = ((sample + 1) * 254 / 2.).clip(0, 255).astype('uint8')

        if len(img_size) == 3:
            sample = np.repeat(
                sample.reshape(1, img_size[1], img_size[2]).swapaxes(0, 1),
                3, axis=0)
        elif len(img_size) == 4:
            sample = sample.reshape(img_size[3], img_size[1],
                                    img_size[2]).swapaxes(0, -1).swapaxes(0, 1)
        else:
            raise ValueError("Do not support {} dim of images".format(
                len(img_size)))
        plt.imshow(sample)

>>>>>>> 4865cf5... updates
    plt.savefig(path+'_samples_n_iter_{}.png'.format(iter),
                bbox_inches='tight')
    plt.close()


class GenerateNegtiveSample(SimpleExtension, MonitoringExtension):
    def __init__(self, img_variable, img_size, **kwargs):
        kwargs.setdefault("every_n_epochs", 10)
        super(GenerateNegtiveSample, self).__init__(**kwargs)
        # fix the samples
        self._cg = ComputationGraph(img_variable)
        # self._input = self._cg.inputs()
        logger.info("Initial the monitor")
        self.img_variable = img_variable
        self._compile()
        self._img_size = img_size

    def _compile(self):
        logger.debug("Compiling the Sampling function from {}".format(
            self.__class__.__name__))
        self.sampling = theano.function([], outputs=self.img_variable)

        # updates=self._cg.updates) this is using for variant seed

    def do(self, callback_name, *args):
        logger.info("Generate Image from the GAN")
        images = self.sampling()[0]
        _plot_images(images, self._img_size,
                     self.main_loop.status['iterations_done'],
                     path=self.main_loop.find_extension('Checkpoint').path)
        logger.debug("Image Generated")


class GenerateText():
    pass


class MarginMonitor():
    pass


class EarlyStopingGAN():
    pass
