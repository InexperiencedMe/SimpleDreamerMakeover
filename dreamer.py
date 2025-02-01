import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os

from networks import RSSM, RewardModel, ContinueModel, Encoder, Decoder, Actor, Critic

from utils import computeLambdaValues, create_normal_dist, DynamicInfos, saveLossesToCSV
from buffer import ReplayBuffer
import imageio


class Dreamer:
    def __init__(self, observationShape, discreteActionBool, actionSize, config, device):
        self.device = device
        self.actionSize = actionSize
        self.discreteActionBool = discreteActionBool

        self.encoder = Encoder(observationShape, config).to(self.device)
        self.decoder = Decoder(observationShape, config).to(self.device)
        self.worldModel = RSSM(actionSize, config, device).to(self.device)
        self.rewardPredictor = RewardModel(config).to(self.device)
        if config.useContinuationPrediction:
            self.continuePredictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discreteActionBool, actionSize, config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.buffer = ReplayBuffer(observationShape, actionSize, config, self.device)

        self.config = config

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.worldModel.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters, lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(), lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(), lr=self.config.criticLR)

        self.worldModelTrainingInfos = DynamicInfos(self.device)
        self.behaviorTrainingInfos   = DynamicInfos(self.device)

        self.totalEpisodes = 0
        self.totalEnvSteps = 0
        self.totalGradientSteps = 0

    def train(self, env, metricsFilename, videoFilenameBase):
        if len(self.buffer) < 1:
            self.environmentInteraction(env, self.config.startupEpisodes)

        for i in range(self.config.trainingIterations):
            for _ in range(self.config.replayRatio):
                data = self.buffer.sample(self.config.batchSize, self.config.batchLength)
                posteriors, recurrentStates, worldModelMetrics = self.worldModelTraining(data)
                actorCriticMetrics = self.behaviorTraining(posteriors, recurrentStates)
                self.totalGradientSteps += 1

            mostRecentScore = self.environmentInteraction(env, self.config.numInteractionEpisodes)

            metricsBase = {"envSteps": self.totalEnvSteps, "gradientSteps": self.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | actorCriticMetrics)

            if i % 10 == 0:
                self.evaluate(env, f"{videoFilenameBase}_{self.totalGradientSteps}_{self.totalEnvSteps}")

    def evaluate(self, env, filename):
        self.environmentInteraction(env, 1, saveVideo=True, filename=filename)

    def worldModelTraining(self, data):
        prior, recurrentState = self.worldModel.recurrentModelInitialInput(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batchLength):
            recurrentState = self.worldModel.recurrentModel(prior, data.action[:, t-1], recurrentState)
            priorDistribution, prior = self.worldModel.priorNet(recurrentState)
            posteriorDistribution, posterior = self.worldModel.posteriorNet(data.embedded_observation[:, t], recurrentState)

            self.worldModelTrainingInfos.append(
                priors                      = prior,
                priorDistributionsMeans     = priorDistribution.mean,
                priorDistributionsStds      = priorDistribution.scale,
                posteriors                  = posterior,
                posteriorDistributionsMeans = posteriorDistribution.mean,
                posteriorDistributionsStds  = posteriorDistribution.scale,
                recurrentStates             = recurrentState)
            prior = posterior
        infos = self.worldModelTrainingInfos.get_stacked()

        reconstructedObservationsDistributions = self.decoder(infos.posteriors, infos.recurrentStates)
        reconstructionLoss                     = -reconstructedObservationsDistributions.log_prob(data.observation[:, 1:]).mean()

        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(infos.posteriors, infos.recurrentStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.done[:, 1:])

        rewardDistribution = self.rewardPredictor(infos.posteriors, infos.recurrentStates)
        rewardLoss         = -rewardDistribution.log_prob(data.reward[:, 1:]).mean()

        priorDistribution   = create_normal_dist(infos.priorDistributionsMeans, infos.priorDistributionsStds, event_shape=1)
        posterior_dist      = create_normal_dist(infos.posteriorDistributionsMeans, infos.posteriorDistributionsStds, event_shape=1)
        klLoss = torch.mean(torch.distributions.kl_divergence(posterior_dist, priorDistribution)) # Change that to maxing individual dists
        klLoss = torch.max(torch.tensor(self.config.freeNats).to(self.device), klLoss)

        worldModelLoss = (klLoss + reconstructionLoss + rewardLoss)
        if self.config.useContinuationPrediction:
            worldModelLoss += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}

        return infos.posteriors.detach(), infos.recurrentStates.detach(), metrics

    def behaviorTraining(self, latentStates, recurrentStates):
        latentState = latentStates.reshape(-1, self.config.latentSize)
        recurrentState = recurrentStates.reshape(-1, self.config.recurrentSize)

        for _ in range(self.config.imaginationHorizon):
            action = self.actor(latentState, recurrentState)
            recurrentState = self.worldModel.recurrentModel(latentState, action, recurrentState)
            _, latentState = self.worldModel.priorNet(recurrentState)
            self.behaviorTrainingInfos.append(latentStates=latentState, recurrentStates=recurrentState)

        infos = self.behaviorTrainingInfos.get_stacked()
        
        predictedRewards = self.rewardPredictor(infos.latentStates, infos.recurrentStates).mean
        values = self.critic(infos.latentStates, infos.recurrentStates).mean

        if self.config.useContinuationPrediction:
            continues = self.continuePredictor(infos.latentStates, infos.recurrentStates).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambdaValues = computeLambdaValues(predictedRewards, values, continues, self.config.imaginationHorizon, self.device, self.config.lambda_)
        actorLoss = -torch.mean(lambdaValues)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions = self.critic(infos.latentStates.detach()[:, :-1], infos.recurrentStates.detach()[:, :-1])
        criticLoss = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"         : actorLoss.item(),
            "criticLoss"        : criticLoss.item(),
            "criticValues"      : values.mean().item()}
        return metrics

    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, seed=0, evaluation=False, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        scores = []
        for i in range(numEpisodes):
            posterior, recurrentState = self.worldModel.recurrentModelInitialInput(1)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed=seed + self.totalEpisodes)
            encodedObservation = self.encoder(torch.from_numpy(observation).float().to(self.device))

            currentScore, stepCount, done, frames = 0, 0, False, []
            while not done:
                recurrentState = self.worldModel.recurrentModel(posterior, action, recurrentState)
                _, posterior   = self.worldModel.posteriorNet(encodedObservation.reshape(1, -1), recurrentState)
                action         = self.actor(posterior, recurrentState).detach()

                if self.discreteActionBool:
                    actionBuffered = action.cpu().numpy()
                    actionForEnv = actionBuffered.argmax()
                else:
                    actionBuffered = action.cpu().numpy()[0]
                    actionForEnv = actionBuffered

                nextObservation, reward, done = env.step(actionForEnv)
                if not evaluation:
                    self.buffer.add(observation, actionBuffered, reward, nextObservation, done)

                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio error
                    targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                    frames.append(cv.resize(frame, (targetWidth, targetHeight), interpolation=cv.INTER_LINEAR))

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().to(self.device))
                observation = nextObservation
                
                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)

                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)

                    break
        return sum(scores)/numEpisodes
    
    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'worldModel'            : self.worldModel.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.worldModel.load_state_dict(checkpoint['worldModel'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])
