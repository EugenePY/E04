from blocks.extensions.monitoring import MonitoringExtension, SimpleExtension

from blocks.graph import ComputationGraph
import skimage
import logging

logger = logging.getLogger(__name__)


def _plot_images(image_batch):
    pass


class GenerateNegtiveSample(SimpleExtension, MonitoringExtension):
    def __init__(self, img_variable, **kwargs):
        super(GenerateNegtiveSample, self).__init__(**kwargs)
        # fix the samples

    def do(self):
        logger.info("Generate Image from the GAN")

        images = self.img_variable
        _plot_images(images, file_name="./neg_samples_{}".format(
            self.main_loop.status['epoch_done']))


class MarginMonitor():
    pass
