import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def getEnvProperties(env):
    observationShape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discreteActionBool = True
        actionSize = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discreteActionBool = False
        actionSize = env.action_space.shape[0]
    else:
        raise Exception
    return observationShape, discreteActionBool, actionSize


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue
        
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))
        
        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))
    
    fig.add_trace(pgo.Scatter(
    x=data["envSteps"], 
    y=[None] * len(data["envSteps"]),  # Dummy points to show second axis
    mode='lines',
    name="Environment Steps (Secondary)",
    xaxis='x2',
    showlegend=False))

    fig.update_layout(
    title=f"{title}",
    title_x=0.5,
    xaxis=dict(
        title="Gradient Steps",
        showgrid=True,
        zeroline=False
    ),
    xaxis2=dict(
        title="Environment Steps",
        overlaying='x',
        side='top',
        showgrid=False,
        zeroline=False
    ),
    yaxis_title="Value",
    template="plotly_dark",
    height=1080,
    width=1920,
    legend=dict(
        x=0.04,
        y=0.04,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="White",
        borderwidth=2))

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize
    
    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


def computeLambdaValues(rewards, values, continues, horizon_length, device, lambda_):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


class Moments(nn.Module):
    def __init__( self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh = 0.95):
        super().__init__()
        self._decay = decay
        self._min = torch.tensor(min_)
        self._percentileLow = percentileLow
        self._percentileHigh = percentileHigh
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()
        low = torch.quantile(x, self._percentileLow)
        high = torch.quantile(x, self._percentileHigh)
        self.low = self._decay*self.low + (1 - self._decay)*low
        self.high = self._decay*self.high + (1 - self._decay)*high
        inverseScale = torch.max(self._min, self.high - self.low)
        return self.low.detach(), inverseScale.detach()