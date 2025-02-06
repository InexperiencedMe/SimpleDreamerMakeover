import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, Bernoulli

from utils import create_normal_dist, sequentialModel1D, horizontal_forward



# Needs a remake
class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config

        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        x = torch.cat((latentState, action), -1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, recurrentState)
        return x


# Needs a remake
class PriorNet(nn.Module):
    def __init__(self, inputSize, outputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, outputSize*2, self.config.activation)

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior


# Needs a remake
class PosteriorNet(nn.Module,):
    def __init__(self, inputSize, outputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, outputSize*2, self.config.activation)

    def forward(self, encodedObservation, recurrentState):
        x = self.network(torch.cat((encodedObservation, recurrentState), 1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


# Remade
class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean, torch.exp(logStd))


# Remade
class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x))

# Needs a remake
class Encoder(nn.Module):
    def __init__(self, observationShape, config):
        super().__init__()
        self.config = config

        activation = getattr(nn, self.config.activation)()
        self.observationShape = observationShape

        self.network = nn.Sequential(
            nn.Conv2d(self.observationShape[0], self.config.depth*1, self.config.kernel_size, self.config.stride),    activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernel_size, self.config.stride),       activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernel_size, self.config.stride),       activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernel_size, self.config.stride),       activation)

    def forward(self, x):
        return horizontal_forward(self.network, x, input_shape=self.observationShape)

# Needs a remake
class Decoder(nn.Module):
    def __init__(self, inputSize, observationShape, config):
        super().__init__()
        self.inputSize = inputSize
        self.observationShape = observationShape
        self.config = config
        activation = getattr(nn, self.config.activation)()

        self.network = nn.Sequential(
            nn.Linear(inputSize, self.config.depth*32),
            nn.Unflatten(1, (self.config.depth*32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth*32, self.config.depth*4, self.config.kernel_size, self.config.stride),         activation,
            nn.ConvTranspose2d(self.config.depth*4, self.config.depth*2, self.config.kernel_size, self.config.stride),          activation,
            nn.ConvTranspose2d(self.config.depth*2, self.config.depth*1, self.config.kernel_size + 1, self.config.stride),      activation,
            nn.ConvTranspose2d(self.config.depth*1, self.observationShape[0], self.config.kernel_size + 1, self.config.stride))

    def forward(self, x):
        x = horizontal_forward(self.network, x, input_shape=(self.inputSize,), output_shape=self.observationShape)
        dist = create_normal_dist(x, std=1, event_shape=len(self.observationShape))
        return dist


# Needs a remake
class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, discreteActionBool, config):
        super().__init__()
        self.config = config
        self.discreteActionBool = discreteActionBool
        actionSize = actionSize if discreteActionBool else 2 * actionSize
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)

    def forward(self, posterior, recurrentState):
        x = torch.cat((posterior, recurrentState), -1)
        x = self.network(x)
        if self.discreteActionBool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist = create_normal_dist(x, mean_scale=self.config.mean_scale, init_std=self.config.init_std, min_std=self.config.min_std, activation=torch.tanh)
            dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
            action = torch.distributions.Independent(dist, 1).rsample()
        return action


# Remade
class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean, torch.exp(logStd))
