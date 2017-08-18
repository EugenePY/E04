import os
import time
import logging
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.extensions import Printing
from blocks.extensions import ProgressBar
from blocks.algorithms import Adam, RMSProp
from algorithms import AdverserialTraning
from blocks.roles import PARAMETER
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph, apply_dropout

from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from loss import WGANLoss
from monitors import GenerateNegtiveSample
from blocks.model import Model

from E04 import *
from data import PokemonGenYellowNormal
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from blocks.roles import INPUT

# sys.setrecursionlimit(100000)

logger = logging.Logger(__name__)
FORMAT = '[%(asctime)s] %(name)s %(message)s'
DATEFMT = "%M:%D:%S"

logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.DEBUG)


def _pokemon_dcgan():
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.)
    }

    batch_size = 20
    data_train = PokemonGenYellowNormal(which_sets=['train'],
                                        sources=['features'])

    train_stream = Flatten(DataStream.default_stream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size)))

    features_size = 56 * 56 * 1

    inputs = T.matrix('features')

    inputs = (inputs)/255. * 2. - 1.

# rng = MRG_RandomStreams(123)
# inputs = inputs * rng.binomial(size=inputs.shape, p=0.1)

    prior = Z_prior(dim=256)
    gen = Generator(input_dim=256, dims=[128, 64, 64, features_size],
                    alpha=0.1, **inits)

    dis = Discriminator(dims=[features_size, 128, 64, 64], alpha=0.1, **inits)

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

# Prepare the dropout
    _inputs = []
    for brick_ in [gen]:
        _inputs.extend(VariableFilter(roles=[INPUT],
                                      bricks=brick_.linear_transformations)(
                                          cg.variables))

    cg_dropout = apply_dropout(cg, _inputs, 0.02)

    gen_obj = cg_dropout.outputs[0]
    dis_obj = cg_dropout.outputs[1]

    gan.dis_params = dis_params
    gan.gen_params = gen_params

    algo = AdverserialTraning(gen_obj=gen_obj, dis_obj=dis_obj,
                            model=gan, dis_iter=5,
                            step_rule=RMSProp(learning_rate=1e-4),
                            gen_consider_constant=z)

    neg_sample = gan.sampling(size=25)

    monitor = TrainingDataMonitoring(variables=[gen_obj, dis_obj],
                                        prefix="train", after_batch=True)

    subdir = './exp/' + 'pokemon' + "-" + time.strftime("%Y%m%d-%H%M%S")

    check_point = Checkpoint("{}/{}".format(subdir, 'pokemon'),
                                every_n_epochs=100,
                                save_separately=['log', 'model'])

    neg_sampling = GenerateNegtiveSample(neg_sample, img_size=(25, 56, 56),
                                         every_n_epochs=100)

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(algorithm=algo, model=model,
                        data_stream=train_stream,
                        extensions=[Printing(), ProgressBar(), monitor,
                                    check_point, neg_sampling])

    main_loop.run()


def _pokemon():
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.)
    }

    batch_size = 20
    data_train = PokemonGenYellowNormal(which_sets=['train'],
                                        sources=['features'])

    train_stream = Flatten(DataStream.default_stream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size)))

    features_size = 56 * 56 * 1

    inputs = T.matrix('features')

    inputs = (inputs)/255. * 2. - 1.

# rng = MRG_RandomStreams(123)
# inputs = inputs * rng.binomial(size=inputs.shape, p=0.1)

    prior = Z_prior(dim=256)
    gen = Generator(input_dim=256, dims=[128, 64, 64, features_size],
                    alpha=0.1, **inits)

    dis = Discriminator(dims=[features_size, 128, 64, 64], alpha=0.1, **inits)

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

# Prepare the dropout
    _inputs = []
    for brick_ in [gen]:
        _inputs.extend(VariableFilter(roles=[INPUT],
                                      bricks=brick_.linear_transformations)(
                                          cg.variables))

    cg_dropout = apply_dropout(cg, _inputs, 0.02)

    gen_obj = cg_dropout.outputs[0]
    dis_obj = cg_dropout.outputs[1]

    gan.dis_params = dis_params
    gan.gen_params = gen_params

    algo = AdverserialTraning(gen_obj=gen_obj, dis_obj=dis_obj,
                            model=gan, dis_iter=5,
                            step_rule=RMSProp(learning_rate=1e-4),
                            gen_consider_constant=z)

    neg_sample = gan.sampling(size=25)

    monitor = TrainingDataMonitoring(variables=[gen_obj, dis_obj],
                                        prefix="train", after_batch=True)

    subdir = './exp/' + 'pokemon' + "-" + time.strftime("%Y%m%d-%H%M%S")

    check_point = Checkpoint("{}/{}".format(subdir, 'pokemon'),
                                every_n_epochs=100,
                                save_separately=['log', 'model'])

    neg_sampling = GenerateNegtiveSample(neg_sample, img_size=(25, 56, 56),
                                         every_n_epochs=100)

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(algorithm=algo, model=model,
                        data_stream=train_stream,
                        extensions=[Printing(), ProgressBar(), monitor,
                                    check_point, neg_sampling])

    main_loop.run()


def _cifir():
    from fuel.datasets import CIFAR10
    import os
    os.environ["FUEL_DATA_PATH"] = os.getcwd() + "/data/"

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.)
    }

    batch_size = 64
    data_train = CIFAR10(which_sets=['train'], sources=['features'])
    streams = DataStream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size))

    train_stream = Flatten(streams)

    # print train_stream.get_epoch_iterator(as_dict=True).next()
    # raise

    features_size = 32 * 32 * 3
    inputs = T.matrix('features')
    inputs = ((inputs / 255.) * 2. - 1.)

    rng = MRG_RandomStreams(123)

    prior = Z_prior(dim=512)
    gen = Generator(input_dim=512, dims=[512, 512, 512, 512,
                                         features_size],
                    alpha=0.1, **inits)

    dis = Discriminator(dims=[features_size, 512, 512 , 512, 512],
                        alpha=0.1, **inits)

    gan = GAN(dis=dis, gen=gen, prior=prior)
    gan.initialize()

    # gradient penalty
    fake_samples, _ = gan.sampling(inputs.shape[0])
    e = rng.uniform(size=(inputs.shape[0], 1))

    mixed_input = (e * fake_samples) + (1 - e) * inputs

    output_d_mixed = gan._dis.apply(mixed_input)

    grad_mixed = T.grad(T.sum(output_d_mixed), mixed_input)

    norm_grad_mixed = T.sqrt(T.sum(T.square(grad_mixed), axis=1))
    grad_penalty = T.mean(T.square(norm_grad_mixed -1))

    y_hat1, y_hat0, z = gan.apply(inputs)

    d_loss_real = y_hat1.mean()
    d_loss_fake = y_hat0.mean()
    d_loss = - d_loss_real + d_loss_fake + 10 * grad_penalty
    g_loss = - d_loss_fake


    dis_obj = d_loss
    gen_obj = g_loss

    model = Model([y_hat0, y_hat1])

    em_loss = -d_loss_real + d_loss_fake

    em_loss.name = "Earth Move loss"
    dis_obj.name = 'Discriminator loss'
    gen_obj.name = 'Generator loss'

    cg = ComputationGraph([gen_obj, dis_obj])

    gen_filter = VariableFilter(roles=[PARAMETER],
                                bricks=gen.linear_transformations)

    dis_filter = VariableFilter(roles=[PARAMETER],
                                bricks=dis.linear_transformations)

    gen_params = gen_filter(cg.variables)
    dis_params = dis_filter(cg.variables)

# Prepare the dropout
    _inputs = []
    for brick_ in [gen]:
        _inputs.extend(VariableFilter(roles=[INPUT],
                    bricks=brick_.linear_transformations)(cg.variables))

    cg_dropout = apply_dropout(cg, _inputs, 0.02)

    gen_obj = cg_dropout.outputs[0]
    dis_obj = cg_dropout.outputs[1]

    gan.dis_params = dis_params
    gan.gen_params = gen_params

    # gradient penalty

    algo = AdverserialTraning(gen_obj=gen_obj, dis_obj=dis_obj,
                              model=gan, dis_iter=5, gradient_clip=None,
                              step_rule=RMSProp(learning_rate=1e-4),
                              gen_consider_constant=z)

    neg_sample = gan.sampling(size=25)

    from blocks.monitoring.aggregation import mean

    monitor = TrainingDataMonitoring(variables=[mean(gen_obj), mean(dis_obj),
                                                mean(em_loss)],
                                     prefix="train", after_batch=True)

    subdir = './exp/' + 'CIFAR10' + "-" + time.strftime("%Y%m%d-%H%M%S")

    check_point = Checkpoint("{}/{}".format(subdir, 'CIFAR10'),
                                every_n_epochs=100,
                                save_separately=['log', 'model'])

    neg_sampling = GenerateNegtiveSample(inputs, neg_sample,
                                         img_size=(25, 32, 32, 3),
                                         every_n_epochs=10)

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(algorithm=algo, model=model,
                         data_stream=train_stream,
                         extensions=[Printing(), ProgressBar(), monitor,
                                     check_point, neg_sampling])

    main_loop.run()


def _pokemon_wgan_gp():
    import os
    os.environ["FUEL_DATA_PATH"] = os.getcwd() + "/data/"
    batch_size = 20
    data_train = PokemonGenYellowNormal(which_sets=['train'],
                                        sources=['features'])

    train_stream = Flatten(DataStream.default_stream(
        data_train, iteration_scheme=SequentialScheme(
            data_train.num_examples, batch_size)))

    features_size = 56 * 56 * 1

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.)
    }

    # print train_stream.get_epoch_iterator(as_dict=True).next()
    # raise

    inputs = T.matrix('features')
    inputs = ((inputs / 255.) * 2. - 1.)

    rng = MRG_RandomStreams(123)

    prior = Z_prior(dim=512)
    gen = Generator(input_dim=512, dims=[512, 512, 512, 512,
                                         features_size],
                    alpha=0.1, **inits)

    dis = Discriminator(dims=[features_size, 512, 512 , 512, 512],
                        alpha=0.1, **inits)

    gan = GAN(dis=dis, gen=gen, prior=prior)
    gan.initialize()

    # gradient penalty
    fake_samples, _ = gan.sampling(inputs.shape[0])
    e = rng.uniform(size=(inputs.shape[0], 1))

    mixed_input = (e * fake_samples) + (1 - e) * inputs

    output_d_mixed = gan._dis.apply(mixed_input)

    grad_mixed = T.grad(T.sum(output_d_mixed), mixed_input)

    norm_grad_mixed = T.sqrt(T.sum(T.square(grad_mixed), axis=1))
    grad_penalty = T.mean(T.square(norm_grad_mixed -1))

    y_hat1, y_hat0, z = gan.apply(inputs)

    d_loss_real = y_hat1.mean()
    d_loss_fake = y_hat0.mean()
    d_loss = - d_loss_real + d_loss_fake + 10 * grad_penalty
    g_loss = - d_loss_fake


    dis_obj = d_loss
    gen_obj = g_loss

    model = Model([y_hat0, y_hat1])

    em_loss = -d_loss_real + d_loss_fake

    em_loss.name = "Earth Move loss"
    dis_obj.name = 'Discriminator loss'
    gen_obj.name = 'Generator loss'

    cg = ComputationGraph([gen_obj, dis_obj])

    gen_filter = VariableFilter(roles=[PARAMETER],
                                bricks=gen.linear_transformations)

    dis_filter = VariableFilter(roles=[PARAMETER],
                                bricks=dis.linear_transformations)

    gen_params = gen_filter(cg.variables)
    dis_params = dis_filter(cg.variables)

# Prepare the dropout
    _inputs = []
    for brick_ in [gen]:
        _inputs.extend(VariableFilter(roles=[INPUT],
                    bricks=brick_.linear_transformations)(cg.variables))

    cg_dropout = apply_dropout(cg, _inputs, 0.02)

    gen_obj = cg_dropout.outputs[0]
    dis_obj = cg_dropout.outputs[1]

    gan.dis_params = dis_params
    gan.gen_params = gen_params

    # gradient penalty

    algo = AdverserialTraning(gen_obj=gen_obj, dis_obj=dis_obj,
                              model=gan, dis_iter=5, gradient_clip=None,
                              step_rule=RMSProp(learning_rate=1e-4),
                              gen_consider_constant=z)

    neg_sample = gan.sampling(size=25)

    from blocks.monitoring.aggregation import mean

    monitor = TrainingDataMonitoring(variables=[mean(gen_obj), mean(dis_obj),
                                                mean(em_loss)],
                                     prefix="train", after_batch=True)

    subdir = './exp/' + 'pokemon-wgan-gp' + "-" + time.strftime("%Y%m%d-%H%M%S")

    check_point = Checkpoint("{}/{}".format(subdir, 'CIFAR10'),
                                every_n_epochs=100,
                                save_separately=['log', 'model'])

    neg_sampling = GenerateNegtiveSample(neg_sample,
                                         img_size=(25, 56, 56),
                                         every_n_epochs=10)

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(algorithm=algo, model=model,
                         data_stream=train_stream,
                         extensions=[Printing(), ProgressBar(), monitor,
                                     check_point, neg_sampling])

    main_loop.run()

if __name__ == '__main__':
    _pokemon_wgan_gp()


