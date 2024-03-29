# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 noet:


from collections import OrderedDict
from argparse import ArgumentParser
import math
import time

# pytorch libraries
import numpy as np
import torch
from torch.distributions import constraints
from torch import nn


#
import pytorch_lightning

# pyro libraries
import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape
from pyro.contrib.oed.differentiable_eig import (
    _differentiable_posterior_loss,
    differentiable_nce_eig,
    _differentiable_ace_eig_loss)
from pyro import poutine
from pyro.contrib.util import lexpand, rmv
from pyro.contrib.oed.eig import _eig_from_ape, nce_eig, _ace_eig_loss, nmc_eig, vnmc_eig, _safe_mean_terms
from pyro.util import is_bad
from pyro.contrib.autoguide import mean_field_entropy


class DoseResponseExperimentalDesignModel(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super(DoseResponseExperimentalDesignModel, self).__init__()
        self.hparams = self.initialize_hparams(hparams)

        self.design_prototype = torch.zeros(
            self.hparams.num_parallel,
            self.hparams.design_size,
            device=self.hparams.device)

        # initialize optimizatier state
        # is this the best place to initialize these?
        self.params = None
        self.baseline = 0.
        self.t = time.time()
        self.wall_times = []

        self.guide = self.build_guide()
        self.loss = self.__build_loss()
        self.eig_upper = self.__build_eig_upper()
        self.eig_lower = self.__build_eig_lower()


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--num_parallel', default=10, type=int)

        # estimator
        parser.add_argument('--estimator', default='posterior', type=str)

        # training params (opt)
        parser.add_argument('--num_samples', default=10, type=int)
        parser.add_argument('--num_steps', default=100, type=int)
        parser.add_argument('--check_bounds_frequency', default=1000, type=int)
        parser.add_argument('--m_final', default=20, type=int)

        parser.add_argument('--optimizer_name', default='exponential', type=str)

        # optimizer_name == 'exponential' -> pyro.optim.ExponentialLR
        parser.add_argument('--exponential_lr_start', default=0.01, type=float)
        parser.add_argument('--exponential_lr_end', default=0.01, type=float)

        # optimizer_name == 'cosine' -> pyro.optim.CosineAnnealingLR
        parser.add_argument('--cosine_lr_T_max', default = 10, type=float)
        parser.add_argument('--cosine_lr_eta_min', default=0, type=float)
        parser.add_argument('--cosine_lr_last_epoch', default=-1, type=int)

        parser.add_argument('--device', default='cuda:0', type=str)
        return parser

    def initialize_hparams(self, hparams):
        assert hparams.design_size > 0

        assert len(hparams.design_range) == 2
        assert hparams.design_range[0] < hparams.design_range[1]

        return hparams

    def check_design(self, design):
        assert all(self.hparams.design_range[0] <= design)
        assert all(design <= self.hparams.design_range[1])
        assert all(design.shape == self.design_prototype.shape)

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def model(self, design_prototype):
        raise NotImplementedError

    def build_guide(self):
        raise NotImplementedError

    def prior_entropy(self):
        return mean_field_entropy(
            self.model,
            [torch.zeros(
                self.hparams.num_parallel,
                self.hparams.design_size,
                device=self.hparams.device)],
            whitelist=self.hparams.target_labels)

    def __build_loss(self):
        if self.hparams.estimator == 'posterior':
            loss = _differentiable_posterior_loss(
                self.model,
                self.guide,
                [self.hparams.observation_label],
                self.hparams.target_labels)
        elif self.hparams.estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: differentiable_nce_eig(
                model=self.model,
                design=d,
                observation_labels=[self.hparams.observation_label],
                target_labels=self.hparams.target_labels,
                N=N,
                M=self.hparams.num_samples,
                **kwargs)
            loss = lambda *args, **kwargs: (-a for a in eig_loss(*args, **kwargs))
        elif self.hparams.estimator == 'ace':
            eig_loss = _differentiable_ace_eig_loss(
                self.model,
                self.guide,
                self.hparams.num_samples,
                [self.hparams.observation_label],
                self.hparams.target_labels)
            loss = lambda *args, **kwargs: (-a for a in eig_loss(*args, **kwargs))
        else:
            raise ValueError("Unexpected estimator")
        return loss

    def __build_eig_lower(self):
        if self.hparams.estimator == 'posterior':
            high_acc = self.loss
        elif self.hparams.estimator == 'nce':
            high_acc = lambda d, N, **kwargs: nce_eig(
                model=self.model,
                design=d,
                observation_labels=[self.hparams.observation_label],
                target_labels=self.hparams.target_labels,
                N=N,
                M=int(math.sqrt(N)),
                **kwargs)
        elif self.hparams.estimator == 'ace':
            high_acc = _ace_eig_loss(
                self.model,
                self.guide,
                self.hparams.m_final,
                [self.hparams.observation_label],
                self.hparams.target_labels)
        else:
            raise ValueError("Unexpected estimator")
        return high_acc

    def __build_eig_upper(self):
        if self.hparams.estimator == 'posterior':
            upper_loss = lambda d, N, **kwargs: vnmc_eig(
                model=self.model,
                design=d,
                observation_labels=[self.hparams.observation_label],
                target_labels=self.hparams.target_labels,
                num_samples=(N, int(math.sqrt(N))),
                num_steps=0,
                guide=self.guide,
                optim=None)
        elif self.hparams.estimator == 'nce':
            upper_loss = lambda d, N, **kwargs: nmc_eig(
                model=self.model,
                design=d,
                observation_labels=[self.hparams.observation_label],
                target_labels=self.hparams.target_labels,
                N=N,
                M=int(math.sqrt(N)),
                **kwargs)
        elif self.hparams.estimator == 'ace':
            upper_loss = lambda d, N, **kwargs: vnmc_eig(
                model=self.model,
                design=d,
                oservation_labels=[self.hparams.observation_label],
                target_labels=self.hparams.target_labels,
                num_samples=(N, int(math.sqrt(N))),
                num_steps=0,
                guide=self.guide,
                optim=None)
        else:
            raise ValueError("Unexpected estimator")
        return upper_loss

    def training_step(self, batch, batch_nb):
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = self.loss(
                self.design_prototype,
                self.hparams.num_samples,
                evaluation=True,
                control_variate=self.baseline)
        self.baseline = -loss.detach()
        self.params = set(
            site["value"].unconstrained()
            for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            import pdb; pdb.set_trace()
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")

        # monkey patch agg_loss to retain graph on the backwards pass
        agg_loss.backwards = lambda self: self.backward(retain_graph=True)

        tqdm_dict = {'train_loss': loss.mean().detach()}
        output = OrderedDict({
            'loss': agg_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def backward(
            self,
            trainer,
            loss: torch.Tensor,
            optimizer: torch.optim.Optimizer,
            optimizer_idx: int) -> None:
        """
        Override backward with your own implementation.
        The loss passed in has already been scaled for accumulated gradients if requested.
        """
        if trainer.precision == 16:
            # .backward is not special on 16-bit with TPUs
            if trainer.on_tpu:
                return

            if self.trainer.use_native_amp:
                self.trainer.scaler.scale(loss).backward(retain_graph=True)

            # TODO: remove in v0.8.0
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)

    def forward(self):
        import pdb
        pdb.set_trace()

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            second_order_closure=None,
            on_tpu=False,
            using_native_amp=False,
            using_lbgfs=False):
        optimizer(self.params)
        optimizer.step()
        pyro.infer.util.zero_grads(self.params)
        if batch_idx == 0:
            for i, scheduler in enumerate(optimizer.optim_objs.values()):
                learning_rate = scheduler.get_last_lr()
                if learning_rate is None:
                    print(f"  not logging because last learning rate is None")
                    continue
                self.logger.experiment.add_scalar(
                    tag=f"learning_rate_{i}",
                    scalar_value=learning_rate[0],
                    global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        lower = self.eig_lower(self.design_prototype, self.hparams.m_final**2, evaluation=True)
        upper = self.eig_upper(self.design_prototype, self.hparams.m_final**2, evaluation=True)
        if isinstance(lower, tuple): lower = lower[1]
        if isinstance(upper, tuple): upper = upper[1]
        lower_mean = lower.mean().cpu()
        upper_mean = upper.mean().cpu()

        a_design = pyro.param('design').squeeze().detach().clone().cpu().data.numpy()
        a_design.sort()
        print(a_design)
        self.logger.experiment.add_histogram(
            tag='design',
            values=a_design,
            global_step=self.global_step)

        print(f"lower {self.global_step}: {lower_mean}")
        self.logger.experiment.add_scalar(
            tag='eig_lower',
            scalar_value=lower_mean,
            global_step=self.global_step)

        print(f"upper {self.global_step}: {upper_mean}")
        self.logger.experiment.add_scalar(
            tag='eig_upper',
            scalar_value=upper_mean,
            global_step=self.global_step)

        tqdm_dict = {
            'eig_lower': lower_mean,
            'eig_upper': upper_mean,
            'wall_time': time.time() - self.t}
        outputs = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict}
        print("done validation step")
        return outputs

    def validation_epoch_end(self, outputs):
        return outputs[0]


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # note both pyro and lightning can manage schedules
        # for now we're going to use Pyro's

        # shim for pyro.optim.PyroOptim and nn.Module interface save state interface
        # implement only state_dict() support for now
        def state_dict(self, destination=None, prefix='', keep_vars=False):
            assert destination is None
            assert prefix == ''
            assert not keep_vars
            return self.get_state()
        pyro.optim.PyroLRScheduler.state_dict = state_dict

        if self.hparams.optimizer_name == 'one_cycle':
            optimizer = pyro.optim.OneCycleLR({
                'optimizer': torch.optim.Adam,
                'optim_args' : {'lr': self.hparams.exponential_lr_start},
                'max_lr' : .1,
                'total_steps' : 20})

        if self.hparams.optimizer_name == "cosine":
            optimizer = pyro.optim.CosineAnnealingLR({
                'optimizer': torch.optim.Adam,
                'optim_args' : {'lr': self.hparams.exponential_lr_start},
                'T_max' : self.hparams.cosine_lr_T_max,
                'eta_min' : self.hparams.cosine_lr_eta_min,
                'last_epoch' : self.hparams.cosine_lr_last_epoch})

        if self.hparams.optimizer_name == "exponential":
            optimizer = pyro.optim.ExponentialLR({
                'optimizer': torch.optim.Adam,
                'optim_args': {'lr': self.hparams.exponential_lr_start},
                'gamma': (
                    self.hparams.exponential_lr_end / \
                    self.hparams.exponential_lr_start) ** (1 / self.hparams.num_steps)
            })

        return [optimizer], []

    def optimizer_zero_grad(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx):
        if optimizer is not None and optimizer.optim_objs is not None:
            for optim_obj in optimizer.optim_objs.items():
                optim_obj[1].optimizer.zero_grad()

    @pytorch_lightning.data_loader
    def train_dataloader(self):
        print('training data loader called')
        return [0]*self.hparams.num_steps

    @pytorch_lightning.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return ([0]*self.hparams.check_bounds_frequency,)
