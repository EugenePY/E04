import theano.tensor as T
import blocks.bricks as bricks
from blocks.bricks import Random
# from loss import AdverserialLoss


class Z_prior(Random):
    def __init__(self, dim, **kwargs):
        super(Z_prior, self).__init__(**kwargs)
        self.dim = dim

    def get_dim(self, name):
        if name == 'output':
            return self.dim

    @bricks.application(input=['size'])
    def apply(self, size):
        return self.theano_rng.normal(
            avg=0., std=1., size=(size, self.dim), dtype='float32')


class DropOut(Random):
    def __init__(self, p, activation, **kwargs):
        super(DropOut, self).__init__(theano_seed=kwargs.get('seed', None))
        self._p = p
        self.activation = activation
        self.children = [self.activation]

    @bricks.application(inputs=['inputs'], outputs=['drop_fprop'])
    def apply(self, inputs):
        noise = self.theano_rng.binomial(p=1-self._p, size=inputs.shape,
                                         dtype='float32')
        return self.activation.apply(inputs * noise)

    @bricks.application(inputs=['inputs'], outputs=['fprop'])
    def inference(self, inputs):
        return self.activation.apply(inputs)


class LeakReLU(bricks.Activation):
    def __init__(self, alpha=0.001, **kwargs):
        super(LeakReLU, self).__init__(**kwargs)
        self._alpha = alpha

    @bricks.application(inputs=['inputs'])
    def apply(self, inputs):
        return T.nnet.relu(inputs, alpha=self._alpha)


class Generator(bricks.MLP):
    """
    Weaker Generator
    DropOut
    """
    def __init__(self, input_dim, dims, alpha, **kwargs):
        n_layers = len(dims)  # add the final layer for output layer
        super(Generator, self).__init__(
            activations=[LeakReLU(alpha)]*(
                n_layers), dims=[input_dim] + dims, **kwargs)


class Discriminator(bricks.MLP):
    """
    Discriminator:
        simple config
    """
    def __init__(self, dims, alpha, **kwargs):
        super(Discriminator, self).__init__(
            activations=[LeakReLU(alpha)]*(len(dims)-1) + [None],
            dims=dims+[1], **kwargs)


class GAN(bricks.base.Brick):
    def __init__(self, dis, gen, prior, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self._prior = prior
        self._dis = dis
        self._gen = gen
        self.children = [self._prior, self._dis, self._gen]

    @bricks.application(inputs=['inputs'], outputs=['y_hat1', 'y_hat0', 'z'])
    def apply(self, inputs):
        size = inputs.shape[0]
        gz, z = self.sampling(size)
        y_hat1 = self._dis.apply(inputs)
        y_hat0 = self._dis.apply(gz)
        return y_hat1, y_hat0, z

    @bricks.application(inputs=['size'], outputs=['gz', 'z'])
    def sampling(self, size):
        z = self._prior.apply(size)
        return self._gen.apply(z), z


class SeqGAN(GAN):
    """Docstring for SeqGAN. """

    def __init__(self, max_n_step):
        """TODO: to be defined1. """
        pass


class EBGAN(bricks.base.Brick):
    pass

if __name__ == '__main__':
    import os
    import time
    import logging
    from fuel.datasets.mnist import MNIST
    from fuel.streams import DataStream
    from fuel.transformers import Flatten
    from fuel.schemes import SequentialScheme
    from blocks.model import Model
    from blocks.filter import VariableFilter
    from blocks.extensions import Printing
    from blocks.extensions import ProgressBar
    from blocks.algorithms import RMSProp
    from algorithms import AdverserialTraning
    from blocks.roles import PARAMETER
    from blocks.main_loop import MainLoop
    from blocks.graph import ComputationGraph
    from blocks.initialization import IsotropicGaussian, Constant
    from blocks.extensions.monitoring import TrainingDataMonitoring
    from blocks.extensions.saveload import Checkpoint
    import numpy as np
    from loss import WGANLoss
    from monitors import GenerateNegtiveSample

    # sys.setrecursionlimit(100000)

    logger = logging.Logger(__name__)
    FORMAT = '[%(asctime)s] %(name)s %(message)s'
    DATEFMT = "%M:%D:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.DEBUG)

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.)
    }

    batch_size = 100
    data_train = MNIST(which_sets=['train'], sources=['features'])

    train_stream = Flatten(DataStream.default_stream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size)))

    features_size = 28 * 28 * 1

    inputs = T.matrix('features')

    test_data = {inputs: 255 *
                 np.random.normal(size=(batch_size, 28*28)).astype('float32')}

    prior = Z_prior(dim=128)

    gen = Generator(input_dim=128, dims=[128,  64, 64, features_size],
                    alpha=0.001, **inits)

    dis = Discriminator(dims=[features_size, 128, 64], alpha=0.001, **inits)

    gan = GAN(dis=dis, gen=gen, prior=prior)
    gan.initialize()

    y_hat1, y_hat0, z = gan.apply(inputs)
    model = Model([y_hat0, y_hat1])
    loss = WGANLoss()
    dis_obj, gen_obj = loss.apply(y_hat0, y_hat1)

    dis_obj.name = 'Discriminator loss'
    gen_obj.name = 'Generator loss'
    cg = ComputationGraph([gen_obj, dis_obj])

    gen_filter = VariableFilter(roles=[PARAMETER],
                                bricks=gen.linear_transformations)

    dis_filter = VariableFilter(roles=[PARAMETER],
                                bricks=dis.linear_transformations)

    gen_params = gen_filter(cg.variables)
    dis_params = dis_filter(cg.variables)

    gan.dis_params = dis_params
    gan.gen_params = gen_params

    # print y_hat1.eval(test_data)
    # print y_hat0.eval(test_data)
    # raise

    algo = AdverserialTraning(gen_obj=gen_obj, dis_obj=dis_obj,
                              model=gan,
                              dis_iter=1,
                              step_rule=RMSProp(learning_rate=1e-4),
                              gen_consider_constant=z)

    neg_sample = gan.sampling(size=25)

    from blocks.monitoring.aggregation import minimum
    monitor = TrainingDataMonitoring(variables=[minimum(gen_obj),
                                                minimum(dis_obj)],
                                     prefix="train", after_batch=True)

    subdir = './exp/' + 'mnist' + "-" + time.strftime("%Y%m%d-%H%M%S")

    check_point = Checkpoint("{}/{}".format(subdir, 'mnist'),
                             every_n_epochs=50,
                             save_separately=['log', 'model'])

    neg_sampling = GenerateNegtiveSample(neg_sample, img_size=(25, 28, 28))

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(algorithm=algo, model=model,
                         data_stream=train_stream,
                         extensions=[Printing(), ProgressBar(), monitor,
                                     check_point, neg_sampling])

    main_loop.run()
