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

class TensorLinear(nn.Module):

    __constants__ = ['bias']

    def __init__(self, *shape, bias=True):
        super(TensorLinear, self).__init__()
        self.in_features = shape[-2]
        self.out_features = shape[-1]
        self.batch_dims = shape[:-2]
        self.weight = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return rmv(self.weight, input) + self.bias


class PosteriorGuide(nn.Module):
    def __init__(self, y_dim, batching):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = TensorLinear(*batching, y_dim, n_hidden)
        self.linear2 = TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batching, n_hidden, 2 + 2 + 2 + 2)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def set_prior(self, rho_concentration, alpha_concentration, slope_mu, slope_sigma):
        self.prior_rho_concentration = rho_concentration
        self.prior_alpha_concentration = alpha_concentration
        self.prior_slope_mu = slope_mu
        self.prior_slope_sigma = slope_sigma

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        top_c = self.softplus(final[..., 0:2])
        bottom_c = self.softplus(final[..., 2:4])
        mid_mu = final[..., 4]
        mid_sigma = self.softplus(final[..., 5])
        slope_mu = final[..., 6]
        slope_sigma = self.softplus(final[..., 7])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("top", dist.Dirichlet(top_c))
            pyro.sample("bottom", dist.Dirichlet(bottom_c))
            pyro.sample("mid", dist.Normal(mid_mu, mid_sigma))
            pyro.sample("slope", dist.Normal(slope_mu, slope_sigma))

class HitRateModel(dose_response_model.DoseResponseExperimentalDesignModel):
    def __init__(self, hparams):
        super(HitRateModel, self).__init__(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = dose_response_model.DoseResponseExperimentalDesignModel.add_model_specific_args(
            parent_parser, root_dir)

        parser.add_argument('--design_size', default=10, type=int)
        parser.add_argument('--design_range', default=[-100, -1e-6], type=float, nargs=2)
        parser.add_argument('--init_range', default=[-75, -30], type=float, nargs=2)
        parser.add_argument('--top_prior_concentration', default=[25., 75.], type=float, nargs=2)
        parser.add_argument('--bottom_prior_concentration', default=[4., 96.], type=float, nargs=2)
        parser.add_argument('--mid_prior_mu', default=50., type=float)
        parser.add_argument('--mid_prior_sd', default=15., type=float)
        parser.add_argument('--slope_prior_mu', default=-.15, type=float)
        parser.add_argument('--slope_prior_sd', default=0.1, type=float)
        parser.add_argument('--observation_label', default="y", type=str)
        parser.add_argument('--target_labels', default=["top", "bottom", "mid", "slope"], nargs=4)
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

        with ExitStack() as stack:
            for plate in iter_plates_to_shape((self.hparams.num_parallel,)):
                stack.enter_context(plate)

            top_distribution = dist.Dirichlet(
                torch.tensor(self.hparams.top_prior_concentration, device=self.hparams.device))
            top = pyro.sample("top", top_distribution).select(-1, 0).unsqueeze(-1)

            bottom_distribution = dist.Dirichlet(
                torch.tensor(self.hparams.bottom_prior_concentration, device=self.hparams.device))
            bottom = pyro.sample("bottom", bottom_distribution).select(-1, 0).unsqueeze(-1)

            mid_distribution = dist.Normal(
                torch.tensor(self.hparams.mid_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.mid_prior_sd, device=self.hparams.device))
            mid = pyro.sample("mid", mid_distribution).unsqueeze(-1)

            slope_distribution = dist.Normal(
                torch.tensor(self.hparams.slope_prior_mu, device=self.hparams.device),
                torch.tensor(self.hparams.slope_prior_sd, device=self.hparams.device))
            slope = pyro.sample("slope", slope_distribution).unsqueeze(-1)

            hit_rate = self.sigmoid(design, top, bottom, mid, slope)
            y = pyro.sample(self.hparams.observation_label, dist.Bernoulli(hit_rate).to_event(1))
            return y

    def build_guide(self):
        guide = PosteriorGuide(self.hparams.design_size, (self.hparams.num_parallel,))
        guide.to(self.hparams.device)
        return guide
