from blocks.algorithms import UpdatesAlgorithm
from theano.updates import OrderedDict
from blocks.graph import ComputationGraph
from blocks.theano_expressions import l2_norm
import theano
import theano.tensor as tensor
import logging

logger = logging.getLogger(__name__)


class AdverserialTraning(UpdatesAlgorithm):
    def __init__(self, gen_obj, dis_obj, step_rule, model,
                 gen_consider_constant,
                 dis_iter=1,
                 **kwargs):
        super(AdverserialTraning, self).__init__(**kwargs)

        self.model = model

        self.gen_cost = gen_obj
        self.dis_cost = dis_obj

        self.dis_iter = dis_iter
        self._n_call = 0

        self.gen_gradients = self._compute_gradients(
            gen_obj, self.model.gen_params,
            consider_constant=gen_consider_constant,
            name='Generator'
        )

        self.dis_gradients = self._compute_gradients(dis_obj,
                                                     self.model.dis_params,
                                                     name='Discriminator')
        gradient_values = self.dis_gradients + self.gen_gradients

        self.total_gradient_norm = (l2_norm(gradient_values)
                                    .copy(name="total_gradient_norm"))

        self.step_rule = step_rule
        logger.debug("Computing parameter steps...")

        self._dis_updates = self._prepare_updates(self.model.dis_params,
                                                  self.dis_gradients,
                                                  self.step_rule,
                                                  name='Discriminator')

        self._gen_updates = self._prepare_updates(self.model.gen_params,
                                                  self.gen_gradients,
                                                  self.step_rule,
                                                  name='Generator')

    def _prepare_updates(self, params, gradients, step_rule, name):
        logger.debug("Prepare the {}'s update".format(name))

        grads = {}

        for param, grad in zip(params, gradients):
            grads[param] = grad

        steps, step_rule_updates = step_rule.compute_steps(grads)
        updates = OrderedDict()
        for param in params:
            updates[param] = param - steps[param]
        updates.update(step_rule_updates)
        updates.update(self.updates)
        return updates

    def add_updates(self, updates):
        """Add updates to the training process.
        The updates will be done _before_ the parameters are changed.
        Parameters
        ----------
        updates : list of tuples or :class:`~collections.OrderedDict`
            The updates to add.
        """
        if isinstance(updates, OrderedDict):
            updates = list(updates.items())
        if not isinstance(updates, list):
            raise ValueError
        self._gen_updates.update(updates)
        self._dis_updates.update(updates)

    def initialize(self):
        logger.info("Initializing the Adverserial training algorithm")

        update_values = self._gen_updates.values() + self._dis_updates.values()

        logger.debug("Inferring graph inputs...")
        self.inputs = ComputationGraph(update_values).inputs
        logger.debug("Compiling training function...")

        self._gen_function = theano.function(
            self.inputs, [], updates=self._gen_updates)

        self._dis_function = theano.function(
            self.inputs, [], updates=self._dis_updates)

        logger.info("The training algorithm is initialized")

    def _compute_gradients(self, cost, params, name, consider_constant=[]):
        gradients = tensor.grad(cost, params)
        logger.info("The {} gradient computation graph is built".format(name))

        return gradients

    def process_batch(self, batch):
        ordered_batch = [batch[v.name] for v in self.inputs]

        if self._n_call % self.dis_iter == 0:
            self._dis_function(*ordered_batch)
        self._gen_function(*ordered_batch)
        self._n_call += 1
