# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 noet:


import math
from contextlib import ExitStack

# pytorch libraries
import torch
from torch.distributions import constraints
from torch import nn


ggg# pyro libraries
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
        self.output_layer = methods.TensorLinear(*batching, n_hidden, 2 + 2 + 2 + 2 + 1)
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

        top_mu = final[..., 0]
        top_sigma = self.softplus(final[..., 1])
        
        bottom_mu = final[..., 2]
        bottom_sigma = self.softplus(final[..., 3])

        mid_mu = final[..., 4]
        mid_sigma = self.softplus(final[..., 5])

        slope_mu = final[..., 6]
        slope_sigma = self.softplus(final[..., 7])

        response_sigma = self.softplus(final[..., 8])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("top", dist.Normal(top_mu, top_sigma))
            pyro.sample("bottom", dist.Normal(bottom_mu, bottom_sigma))
            pyro.sample("mid", dist.Normal(mid_mu, mid_sigma))
            pyro.sample("slope", dist.Normal(slope_mu, slope_sigma))
            pyro.sample("response", dist.Normal(0, response_sigma))

class HillModel(dose_response_model.DoseResponseExperimentalDesignModel):
    def __init__(self, hparams):
        super(HillModel, self).__init__(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = dose_response_model.DoseResponseExperimentalDesignModel.add_model_specific_args(
            parent_parser, root_dir)

        parser.add_argument('--design_size', default=10, type=int)
        parser.add_argument('--design_range', default=[-9, -4], type=float, nargs=2)
        parser.add_argument('--init_range', default=[-9, -4], type=float, nargs=2)
        parser.add_argument('--top_prior_mu', default=100., type=float)
        parser.add_argument('--top_prior_sd', default=100., type=float)
        parser.add_argument('--bottom_prior_mu', default=100., type=float)
        parser.add_argument('--bottom_prior_sd', default=100., type=float)
        parser.add_argument('--mid_prior_mu', default=50., type=float)
        parser.add_argument('--mid_prior_sd', default=15., type=float)
        parser.add_argument('--slope_prior_mu', default=-.15, type=float)
        parser.add_argument('--slope_prior_sd', default=0.1, type=float)
        parser.add_argument('--response_prior_sd', default=5., type=float)
        parser.add_argument('--observation_label', default="observation", type=str)
        parser.add_argument('--target_labels',
            default=["top", "bottom", "mid", "slope", "response"],
            nargs=5)
        return parser

    def sigmoid(self, x, top, bottom, mid, slope):
        return (top - bottom) * torch.sigmoid((x - mid) * slope) + bottom

    def model(self, design_prototype):
        design_init = lexpand(
            torch.linspace(
                *self.hparams.init_range,
                self.hparams.design_size,
                device=self.hparams.device),
            self.hparams.num_parallel)
        design_constraint = constraints.interval(*self.hparams.design_range)
        design = pyro.param("design", design_init, constraint=design_constraint)
        design = design.expand(design_prototype.shape)

        with pyro.plate_stack("plate_stack", design_prototype.shape[:-1])
            # define the prior distribution for the parameters for the model
            top_distribution = dist.Normal(
                torch.tensor(self.hparams.top_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.top_prior_sd, device=self.hparams.device))
            bottom_distribution = dist.Normal(
                torch.tensor(self.hparams.bottom_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.bottom_prior_sd, device=self.hparams.device))
            mid_distribution = dist.Normal(
                torch.tensor(self.hparams.mid_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.mid_prior_sd, device=self.hparams.device))
            slope_distribution = dist.Normal(
                torch.tensor(self.hparams.slope_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.slope_prior_sd, device=self.hparams.device))

            # sample  
            top = pyro.sample("top", top_distribution).unsqueeze(-1)
            bottom = pyro.sample("bottom", bottom_distribution).unsqueeze(-1)
            mid = pyro.sample("mid", mid_distribution).unsqueeze(-1)
            slope = pyro.sample("slope", slope_distribution).unsqueeze(-1)
            response = pyro.sample("response", response_distribution).unsqueeze(-1)
            
            # define the response distribution for each sample point
            response_distribution = dist.Normal(
                torch.zeros(design_size, device=self.hparams.device),
                torch.tensor(
                    self.hparams.response_prior_sd,
                    device=self.hparams.device).expand(design_size))

            # combine the model and the response into the observation distribution
            # the .to_event(1) indicates the design points are depenent
            observation_distribution = dist.Delta(
                self.sigmoid(design, top, bottom, mid, slope) + response).to_event(1)

            # sample observations for each design point
            #    observation.shape = [<batch_dims>, <design_size>]
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
