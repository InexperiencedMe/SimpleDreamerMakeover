import os
import argparse
from datetime import datetime
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics
from buffer import ReplayBuffer
import torch
import numpy as np
import random
import gymnasium as gym
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils import saveLossesToCSV
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = f"{config.folderNames.checkpointsFolder}/{runName}_{config.checkpointToLoad}"
    metricsFilename         = f"{config.folderNames.metricsFolder}/{runName}"     # folders arent created automatically. They have to exist for now
    plotFilename            = f"{config.folderNames.plotsFolder}/{runName}"
    checkpointFilenameBase  = f"{config.folderNames.checkpointsFolder}/{runName}"
    videoFilenameBase       = f"{config.folderNames.videosFolder}/{runName}"
    env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
    envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))
    observationShape, discreteActionBool, actionSize = getEnvProperties(env)
    print(f"envProperties: {getEnvProperties(env)}")

    dreamer = Dreamer(observationShape, discreteActionBool, actionSize, config.dreamer, device)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    for _ in range(1, config.iterationsNum + 1):
        for _ in range(1, config.replayRatio + 1):
            data = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            posteriors, recurrentStates, worldModelMetrics = dreamer.worldModelTraining(data)
            actorCriticMetrics = dreamer.behaviorTraining(posteriors, recurrentStates)
            dreamer.totalGradientSteps += 1

            if dreamer.totalGradientSteps % config.checkpointInterval == 0:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
        saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | actorCriticMetrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
