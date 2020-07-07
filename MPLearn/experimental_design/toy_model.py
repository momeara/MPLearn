# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 noet:


import math
from contextlib import ExitStack

# pytorch libraries
import torch
from torch.distributions import constraints
from torch import nn


# pyro libraries
import pyro
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape
from pyro.contrib.util import lexpand, rmv

from . import dose_response_model
from . import methods



class PosteriorGuide(nn.Module):
    def __init__(
            self,
            observation_dim,
            batching):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = methods.TensorLinear(*batching, observation_dim, n_hidden)
        self.linear2 = methods.TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = methods.TensorLinear(*batching, n_hidden, 2 + 2)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(
            self,
            observation_dict,
            design_prototype,
            observation_labels,
            target_labels):

        y = observation_dict[observation_labels[0]] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        knot0_mu = final[..., 0]
        knot0_sd = self.softplus(final[..., 1])

        knot1_mu = final[..., 2]
        knot1_sd = self.softplus(final[..., 3])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("knot0", dist.Normal(knot0_mu, knot0_sd))
            pyro.sample("knot1", dist.Normal(knot1_mu, knot1_sd))

class ToyModel(dose_response_model.DoseResponseExperimentalDesignModel):
    def __init__(self, hparams):
        super(ToyModel, self).__init__(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = dose_response_model.DoseResponseExperimentalDesignModel.add_model_specific_args(
            parent_parser, root_dir)

        parser.add_argument('--design_size', default=10, type=int)
        parser.add_argument('--design_range', default=[0, 1], type=float, nargs=2)
        parser.add_argument('--init_range', default=[0, 1], type=float, nargs=2)
        parser.add_argument('--knot0_prior_mu', default=0., type=float)
        parser.add_argument('--knot0_prior_sd', default=5., type=float)
        parser.add_argument('--knot1_prior_mu', default=1., type=float)
        parser.add_argument('--knot1_prior_sd', default=1., type=float)
        parser.add_argument('--observation_label', default="observation", type=str)
        parser.add_argument('--target_labels', default=["knot0", "knot1"], nargs=2)
        return parser


    def model(self, design_prototype):
        """
            :param torch.tensor design
                 shape: [<batch_dims>, <design_size>]
        """
        batch_dims = design_prototype.shape[:-1]
        #design_size = design_prototype.shape[-1]

        # initialize design and store it in the param store
#        design_init = torch.linspace(
#            *self.hparams.init_range,
#            steps=self.hparams.design_size,
#            device=self.hparams.device)
        design_init = torch.tensor(
            [1.0]*self.hparams.design_size,
            device=self.hparams.device)
        design_constraint = constraints.interval(*self.hparams.design_range)
        design = pyro.param("design", design_init, constraint=design_constraint)
        design = design.expand(design_prototype.shape)

        # Indicate that the batch dimensions are independent
        # with a plate_stack over the <batch_dims>
        with pyro.plate_stack('plate_stack', batch_dims):

            # Define the prior distribution for the knots, p(theta).
            knot0_distribution = dist.Normal(
                torch.tensor(self.hparams.knot0_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.knot0_prior_sd, device=self.hparams.device))
            knot1_distribution = dist.Normal(
                torch.tensor(self.hparams.knot1_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.knot1_prior_sd, device=self.hparams.device))

            # Sample a common value of the knots for each batch.
            # The unsqueeze(-1) add dimension on the right that gets
            # broadcast across the <design_size> dimension
            #    knot0.shape = knot1.shape = [<batch_dims>, 1]
            knot0 = pyro.sample("knot0", knot0_distribution).unsqueeze(-1)
            knot1 = pyro.sample("knot1", knot1_distribution).unsqueeze(-1)

            # Define the observation distribution for an experiment
            # The .to_event(1) indicates design points are dependent
            observation_distribution = dist.Normal(
                knot0*(1-design) + knot1*design, .1).to_event(1)

            #Sample observations at each design point in each batch
            #   observation.shape = [<batch_dims>, <design_size>]
            observation = pyro.sample(
                self.hparams.observation_label,
                observation_distribution)
            return observation

    def build_guide(self):
        guide = PosteriorGuide(
            self.hparams.design_size,
            (self.hparams.num_parallel,))
        guide.to(self.hparams.device)
        return guide
