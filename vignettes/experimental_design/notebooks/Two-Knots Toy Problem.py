#!/usr/bin/env python
# coding: utf-8

# ## Model
# ```python
# Model(design):
#     knot0 ~ Normal( μ0, σ0 )
#     knot1 ~ Normal( μ1, σ1 )
#     observation ~ Normal(knot0・(1-design) + knot1・design, 1)
# ```
# 
# ![image1.png](attachment:image1.png)
# 
# ## Problem statment:
# Choose design points D = [d1, d2, … dn], to optimally infer the Knot0 and Knot1. If a design point is on the left, more is learned about knot0 and less about knot1. Similarly, if a design point is on the right, then more is learned about knot1 and less about knot0.
# 
# 

# In[8]:


import math
import pandas as pd
import matplotlib
import seaborn
import torch
import pyro
from pyro import poutine
from pyro.contrib.util import lexpand, rmv
from pyro.contrib.oed.eig import nmc_eig, vnmc_eig
from pyro.contrib.oed.differentiable_eig import (
        _differentiable_ace_eig_loss)
pyro.enable_validation(True)


# In[19]:


def model(design):
    """
        :param torch.tensor design
             shape: [<batch_dims>, <design_size>]
    """
    
    batch_dims = design.shape[:-1]
    design_size = design.shape[-1]
    
    # Indicate that the batch dimensions are independent with a plate_stack over the <batch_dims>
    with pyro.plate_stack('plate_stack', batch_dims):

        # Define the prior distribution for the knots, p(theta).
        knot0_distribution = pyro.distributions.Normal(torch.tensor(0.0), torch.tensor(5.0))
        knot1_distribution = pyro.distributions.Normal(torch.tensor(0.0), torch.tensor(5.0))

        # Sample a common value of the knots for each batch.
        # The unsqueeze(-1) add dimension on the right that gets
        # broadcast across the <design_size> dimension
        #    knot0.shape = knot1.shape = [<batch_dims>, 1]
        knot0 = pyro.sample("knot0", knot0_distribution).unsqueeze(-1)
        knot1 = pyro.sample("knot1", knot1_distribution).unsqueeze(-1)
        
        # Define the observation distribution for an experiment
        # The .to_event(1) indicates design points are dependent
        observation_distribution = pyro.distributions.Normal(
            knot0*(1-design) + knot1*design, .1).to_event(1)
        
        # Sample observations at each design point in each batch
        #   observation.shape = [<batch_dims>, <design_size>]
        return pyro.sample("observation", observation_distribution) 


# Plot `5000` experiments with `1000` design points each. 

# In[20]:


pyro.clear_param_store()
design = torch.linspace(start=0, end=1, steps=1000)
design = lexpand(design, 5000)
observations = model(design)

print(f"design.shape: {design.shape}")
print(f"observations.shape: {observations.shape}")

matplotlib.pyplot.figure(figsize=(16, 6))
seaborn.scatterplot(
    x=design.flatten(),
    y=observations.flatten(),
    alpha=.05,
    s=.1,
    color=matplotlib.colors.hex2color('#000000'),
    edgecolor=None)


# ## Solution: Nested Monte Carlo
# 
# Use Nested Monte Carlo to estimate an upperboud for the expected inforamtiong gain for designs that put different fractions of the desing points at either end of the range. For example, design_id is the sum across design points, so `design_id` == 17 corresponds to [0, 0, 0, 1, 1, ..., 1] 
# 

# In[28]:


pyro.clear_param_store()
design_size = 20
N = 10000
M = 100
M_prime = None
n_replicas = 20

designs = []
for i in range(design_size+1):
    design = (i)*[0.0] + (design_size-i)*[1.0]
    designs.append(design)

data=[]
for design in designs:
    for replica in range(n_replicas):
        eig = nmc_eig(
            model=model,
            design=torch.tensor(design),
            observation_labels=['observation'],
            target_labels=['knot0', 'knot1'],
            N = N,
            M = M,
            M_prime=M_prime)
        data.append({
            'design' : design,
            'design_id' : sum(design),
            'N' : N,
            'M' : M,
            'M_prime' : M_prime,
            'replica' : replica,
            'eig' : eig,
            'eig_mean' : eig.mean()
        })
        print(f"design: {design}")
        print(f"  eig: {eig}")
        print(f"  mean: {eig.mean()}")


# In[27]:


data = pd.DataFrame(data)
matplotlib.pyplot.figure(figsize=(16, 6))
seaborn.scatterplot(
    x='design_id',
    y='eig_mean',
    data=data)


# ## Solution ACE
# 
# Remake model where the design is trainable and the input `design_prototype` is just used to specify the shape.
# 
# To fit the model requires using a `guide`, a trainable variational network that maps the `observations` to the model parameters. The TensorLinear module is similar to `torch.nn.Linear` except the batching dimensions are handled differently.
# 
# 

# In[5]:


def model_trainable(design_prototype):
    """
        :param torch.tensor design
             shape: [<batch_dims>, <design_size>]
    """
    
    batch_dims = design_prototype.shape[:-1]
    design_size = design_prototype.shape[-1]
    
    design = pyro.param(
        "design",
        torch.linspace(start=0, end=1, steps=design_size),
        constraint=torch.distributions.constraints.interval(0, 1)
    ).expand(design_prototype.shape)
    
    # Indicate that the batch dimensions are independent with a plate_stack over the <batch_dims>
    with pyro.plate_stack('plate_stack', batch_dims):

        # Define the prior distribution for the knots, p(theta).
        knot0_distribution = pyro.distributions.Normal(torch.tensor(0.0), torch.tensor(5.0))
        knot1_distribution = pyro.distributions.Normal(torch.tensor(0.0), torch.tensor(5.0))

        # Sample a common value of the knots for each batch.
        # The unsqueeze(-1) add dimension on the right that gets
        # broadcast across the <design_size> dimension
        #    knot0.shape = knot1.shape = [<batch_dims>, 1]
        knot0 = pyro.sample("knot0", knot0_distribution).unsqueeze(-1)
        knot1 = pyro.sample("knot1", knot1_distribution).unsqueeze(-1)
        
        # Define the observation distribution for an experiment
        # The .to_event(1) indicates design points are dependent
        observation_distribution = pyro.distributions.Normal(
            knot0*(1-design) + knot1*design, .1).to_event(1)
        
        # Sample observations at each design point in each batch
        #   observation.shape = [<batch_dims>, <design_size>]
        return pyro.sample("observation", observation_distribution) 


# In[6]:


class TensorLinear(torch.nn.Module):

    __constants__ = ['bias']

    def __init__(self, *shape, bias=True):
        super(TensorLinear, self).__init__()
        self.in_features = shape[-2]
        self.out_features = shape[-1]
        self.batch_dims = shape[:-2]
        self.weight = torch.nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return pyro.contrib.util.rmv(self.weight, input) + self.bias

class PosteriorGuide(torch.nn.Module):
    def __init__(self, observation_dim, batching):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = TensorLinear(*batching, observation_dim, n_hidden)
        self.linear2 = TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batching, n_hidden, 2 + 2)
        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()

    def forward(self, observation_dict, design_prototype, observation_labels, target_labels):
        y = observation_dict["observation"] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        knot0_mu = final[..., 0]
        knot0_sigma = self.softplus(final[..., 1])
        knot1_mu = final[..., 2]
        knot1_sigma = self.softplus(final[..., 3])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in pyro.contrib.util.iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("knot0", pyro.distributions.Normal(knot0_mu, knot0_sigma))
            pyro.sample("knot1", pyro.distributions.Normal(knot1_mu, knot1_sigma))


# In[9]:


from contextlib import ExitStack
import tqdm

pyro.clear_param_store()

design_size = 20    # number of design points
num_parallel = 10   # how many are in a batch
num_samples = 100   # number of contrastice samples for estimating bounds
num_steps = 10000   # optimization iterations
h_freq = 100

design_prototype = torch.zeros([num_parallel, design_size])
guide = PosteriorGuide(design_size, (num_parallel,))

eig_loss = _differentiable_ace_eig_loss(
    model_trainable,
    guide,
    num_samples,
    ['observation'],
    ['knot0', 'knot1'])
loss_fn = lambda *args, **kwargs: (-a for a in eig_loss(*args, **kwargs))

optim = pyro.optim.ExponentialLR({
    'optimizer': torch.optim.Adam,
    'optim_args': {'lr': .01},
    'gamma': 1.0})

params=None
baseline = 0.
for step in range(num_steps):
    if params is not None: pyro.infer.util.zero_grads(params)
    with pyro.poutine.trace(param_only=True) as param_capture:
        agg_loss, loss = loss_fn(
            design_prototype,
            num_samples,
            evaluation=True,
            control_variate=baseline)
    baseline = -loss.detach()
    params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
    agg_loss.backward(retain_graph=True)
    optim(params)
    optim.step()

    if step % h_freq == 0:
        a_design = pyro.param('design').squeeze().detach().clone().cpu().data.numpy()
        a_design.sort()
        print(a_design)
        print('eig', baseline.squeeze().mean())


# In[1]:


import os
from argparse import ArgumentParser
import torch
import pyro
import pytorch_lightning
import MPLearn.embedding_notebook
from MPLearn.experimental_design import toy_model
get_ipython().run_line_magic('load_ext', 'tensorboard')
root_dir = os.path.dirname(os.path.realpath("~/opt/MPLearn/vignettes/dose_response/intermediate_data"))


# In[2]:


from tensorboard import notebook
notebook.list()
get_ipython().run_line_magic('tensorboard', '--logdir {root_dir}')


# In[ ]:


def run_toy_model():
    SEED = 2334
    torch.manual_seed(SEED)
    #np.random.seed(SEED)
    pyro.clear_param_store()


    logger = pytorch_lightning.loggers.TestTubeLogger(
        save_dir=root_dir,
        name="test_top_model")
    #logger.experiment.tag({'design_size': 51, 'optimizer': 'ace'}) 

    parent_parser = ArgumentParser(add_help=False)

    parser = toy_model.ToyModel.add_model_specific_args(
        parent_parser, root_dir)
    hparams = parser.parse_args(args=[
        '--device', 'cpu:0',
        '--optimizer_name', 'cosine',
        '--exponential_lr_start', '.01',
        '--exponential_lr_end', '.005',
        '--num_samples', '20',
        '--design_size', '51'])
    model = toy_model.ToyModel(hparams)

    trainer = pytorch_lightning.Trainer(
        nb_sanity_val_steps=0,
        max_nb_epochs=2000,
        logger=logger)
    trainer.fit(model)
run_toy_model()


# In[ ]:




