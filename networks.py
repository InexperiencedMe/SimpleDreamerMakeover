import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, Bernoulli

from utils import create_normal_dist, sequentialModel1D, horizontal_forward


# Needs to go. Take out recurrent, prior, posterior
class RSSM(nn.Module):
    def __init__(self, actionSize, config, device):
        super().__init__()
        self.config = config.rssm
        self.device = device

        self.recurrentModel = RecurrentModel(actionSize, config)
        self.priorNet = PriorNet(config)
        self.posteriorNet = PosteriorNet(config)

    def recurrentModelInitialInput(self, batchSize):
        return self.priorNet.initialInput(batchSize).to(self.device), self.recurrentModel.initialInput(batchSize).to(self.device)

# Needs a remake
class RecurrentModel(nn.Module):
    def __init__(self, actionSize, config):
        super().__init__()
        self.config = config.rssm.recurrentModel
        self.latentSize = config.latentSize
        self.recurrentSize = config.recurrentSize

        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(self.latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, self.recurrentSize)

    def forward(self, embedded_state, action, deterministic):
        x = torch.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def initialInput(self, batchSize):
        return torch.zeros(batchSize, self.recurrentSize)

# Needs a remake
class PriorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.rssm.priorNet
        self.latentSize = config.latentSize
        self.recurrentSize = config.recurrentSize

        self.network = sequentialModel1D(self.recurrentSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize*2, self.config.activation)

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def initialInput(self, batchSize):
        return torch.zeros(batchSize, self.latentSize)

# Needs a remake
class PosteriorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.rssm.posteriorNet
        self.encodedObservationSize = config.encodedObservationSize
        self.latentSize = config.latentSize
        self.recurrentSize = config.recurrentSize

        self.network = sequentialModel1D(self.encodedObservationSize + self.recurrentSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize*2, self.config.activation)

    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


# Remade
class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config.reward
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean, torch.exp(logStd))


# Remade
class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config.continue_
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x))

# Needs a remake
class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.encoder

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Conv2d(self.observation_shape[0], self.config.depth * 1, self.config.kernel_size, self.config.stride),   activation,
            nn.Conv2d(self.config.depth * 1, self.config.depth * 2, self.config.kernel_size, self.config.stride),       activation,
            nn.Conv2d(self.config.depth * 2, self.config.depth * 4, self.config.kernel_size, self.config.stride),       activation,
            nn.Conv2d(self.config.depth * 4, self.config.depth * 8, self.config.kernel_size, self.config.stride),       activation)

    def forward(self, x):
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)

# Needs a remake
class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.decoder
        self.latentSize = config.latentSize
        self.recurrentSize = config.recurrentSize

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(self.recurrentSize + self.latentSize, self.config.depth * 32),
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(self.config.depth * 32, self.config.depth * 4, self.config.kernel_size, self.config.stride),
            activation,
            nn.ConvTranspose2d(self.config.depth * 4, self.config.depth * 2, self.config.kernel_size, self.config.stride),
            activation,
            nn.ConvTranspose2d(self.config.depth * 2, self.config.depth * 1, self.config.kernel_size + 1, self.config.stride),
            activation,
            nn.ConvTranspose2d(self.config.depth * 1, self.observation_shape[0], self.config.kernel_size + 1, self.config.stride))

    def forward(self, posterior, deterministic):
        x = horizontal_forward(self.network, posterior, deterministic, output_shape=self.observation_shape)
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist


# Needs a remake
class Actor(nn.Module):
    def __init__(self, discrete_action_bool, actionSize, config):
        super().__init__()
        self.config = config.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.latentSize = config.latentSize
        self.recurrentSize = config.recurrentSize

        actionSize = actionSize if discrete_action_bool else 2 * actionSize

        self.network = sequentialModel1D(self.latentSize + self.recurrentSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)

    def forward(self, posterior, deterministic):
        x = torch.cat((posterior, deterministic), -1)
        x = self.network(x)
        if self.discrete_action_bool:
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
        self.config = config.agent.critic
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean, torch.exp(logStd))
