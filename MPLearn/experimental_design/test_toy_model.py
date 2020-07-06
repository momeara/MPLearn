



### Questions about Pyro
###
###    * Should the dose_response model be a PyroModule or registered with pyro.module()?
###    * 
###


import torch
import pyro
from pyro.contrib.oed.eig import nmc_eig, vnmc_eig

designs = {
    'design1' : torch.tensor([0.0, 0.0, 0.0, 0.0]),
    'design2' : torch.tensor([0.0, 0.0, 0.0, 1.0]),
    'design3' : torch.tensor([0.0, 0.0, 1.0, 1.0]),
    'design4' : torch.tensor([0.0, 1.0, 1.0, 1.0]),
    'design5' : torch.tensor([1.0, 1.0, 1.0, 1.0])}

def model(design):
    mode0_distribution = pyro.distributions.Normal(torch.tensor(0.0), torch.tensor(5.0))
    mode0 = pyro.sample("mode0", mode0_distribution)

    mode1_distribution = pyro.distributions.Normal(torch.tensor(1.0), torch.tensor(1.0))
    mode1 = pyro.sample("mode1", mode1_distribution)

    observation_distribution = pyro.distributions.Delta(
        mode0*(1-design) + mode1*design).to_event(1)
    return pyro.sample('observation', observation_distribution)

class PosteriorGuide(torch.nn.Module):
    def __init__(self):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = torch.nn.Linear(1, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, 2 + 2)
        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()

    def forward(self, observation_dict):
        y = observation_dict['observation'] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        mode0_mu = final[..., 0]
        mode0_sd = self.softplus(final[..., 1])

        mode1_mu = final[..., 2]
        mode1_sd = self.softplus(final[..., 3])

        pyro.module("posterior_guide", self)

        pyro.sample("mode0", pyro.distributions.Normal(mode0_mu, mode0_sd))
        pyro.sample("mode1", pyro.distributions.Normal(mode1_mu, mode1_sd))



loss = vnmc_eig(
    model=model,
    design=designs['design1'],
    observation_labels=['observation'],
    target_labels=['mode1'],
    num_samples=(1000, 100),
    num_steps=0,
    optim=None,
    return_histor=False,
    final_design=None,
    final_num_samples=None)


loss = nmc_eig(
    model=model,
    design=designs['design1'],
    observation_labels=['observation'],
    target_labels=['mode1'],
    N = 1000,
    M = 100)

