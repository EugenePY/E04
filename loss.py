from blocks.bricks.cost import CostMatrix
import theano.tensor as T
import blocks.bricks as bricks
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from blocks.bricks.cost import Cost
from blocks.bricks import application
import theano.tensor as T


@add_metaclass(ABCMeta)
class AdverserialLossCostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.
    Assumes that the data has format (batch, features).
    """
    @application(outputs=["d_obj", "g_obj"])
    def apply(self, *args, **kwargs):
        d_obj, g_obj = self.cost_matrix(*args, **kwargs)
        return T.sum(d_obj, axis=1).mean(), T.sum(g_obj, axis=1).mean()

    @abstractmethod
    @application
    def cost_matrix(self, *args, **kwargs):
        pass


class AdverserialLoss(AdverserialLossCostMatrix):
    """
    The loss of GAN is in this form:
        G is the generator, D is the discriminator

            min_{G}max_{D} V(G, D) = E(D(X) + (1-D(G(X))))

            d_obj =  0.5 * (d.layers[-1].cost(y1, y_hat1) +
                    d.layers[-1].cost(y0, y_hat0))))

            g_obj = d.layers[-1].cost(y1, y_hat0_no_drop))

    """
    def _dis_cost(self, y, y_hat):
        return T.nnet.binary_crossentropy(y_hat, y)

    @bricks.application(inputs=['y_hat0', 'y_hat1'], outputs=["d_obj", "g_obj"])
    def cost_matrix(self, y_hat0, y_hat1):
        y0 = T.zeros_like(y_hat0)
        y1 = T.ones_like(y_hat1)
        d_obj = 0.5 * (self._dis_cost(y0, y_hat0) +
                       self._dis_cost(y1, y_hat1))
        g_obj = self._dis_cost(y1, y_hat0)

        return d_obj, g_obj


class LossSensitiveGan():
    pass
