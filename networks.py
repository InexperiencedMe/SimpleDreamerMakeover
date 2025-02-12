import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

from utils import create_normal_dist, sequentialModel1D, horizontal_forward



class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.linear = nn.Linear(latentSize + actionSize, self.config.hiddenSize)
        self.recurrent = nn.GRUCell(self.config.hiddenSize, recurrentSize)

    def forward(self, recurrentState, latentState, action):
        return self.recurrent(self.activation(self.linear(torch.cat((latentState, action), -1))), recurrentState)


class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.mlp = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.mlp(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.mlp = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.mlp(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))

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


LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, device, config, actionLow=[-1], actionHigh=[1]):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias", ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))


    def forward(self, x, training=False):
        x = self.network(x)
        mean, logStd = x.chunk(2, dim=-1)
        logStd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logStd + 1)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh * self.actionScale + self.actionBias
        if training:
            logprobs = distribution.log_prob(sample)
            logprobs -= torch.log(self.actionScale * (1 - sampleTanh.pow(2)) + 1e-6)
            entropy = distribution.entropy()
            return action, logprobs.sum(-1), entropy.sum(-1)
        else:
            return action


class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
