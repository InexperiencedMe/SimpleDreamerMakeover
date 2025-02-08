import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, Encoder, Decoder, Actor, Critic

from utils import computeLambdaValues, create_normal_dist
from buffer import ReplayBuffer
import imageio


class Dreamer:
    def __init__(self, observationShape, discreteActionBool, actionSize, config, device):
        self.device = device
        self.actionSize = actionSize
        self.discreteActionBool = discreteActionBool
        self.config = config

        self.recurrentSize = config.recurrentSize
        self.latentSize = config.latentLength*config.latentClasses
        self.fullStateSize = config.recurrentSize + self.latentSize

        # self.actor           = Actor2(fullStateSize, actionSize, device, config.actor, actionHigh=[1, 1, 1], actionLow=[-1, 0, 0]).to(self.device)
        self.actor           = Actor(self.fullStateSize, actionSize, discreteActionBool,                             config.actor          ).to(self.device)
        self.critic          = Critic(self.fullStateSize,                                                            config.critic         ).to(self.device)
        self.encoder         = Encoder(observationShape,                                                        config.encoder        ).to(self.device) # Should have output size
        self.decoder         = Decoder(self.fullStateSize, observationShape,                                         config.decoder        ).to(self.device)
        self.recurrentModel  = RecurrentModel(config.recurrentSize, self.latentSize, actionSize,              config.recurrentModel ).to(self.device)
        self.priorNet        = PriorNet(config.recurrentSize, self.latentSize,                                config.priorNet       ).to(self.device)
        self.posteriorNet    = PosteriorNet(config.recurrentSize + config.encodedObsSize, self.latentSize,    config.posteriorNet   ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize,                                                       config.reward         ).to(self.device)
        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                              config.continuation   ).to(self.device)

        self.buffer = ReplayBuffer(observationShape, actionSize, config, device)


        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters, lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(), lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(), lr=self.config.criticLR)

        self.totalEpisodes = 0
        self.totalEnvSteps = 0
        self.totalGradientSteps = 0

    def worldModelTraining(self, data):
        data.encodedObservation = self.encoder(data.observation)
        previousRecurrentState, previousLatentState = torch.zeros(len(data.action), self.recurrentSize, device=self.device), torch.zeros(len(data.action), self.latentSize, device=self.device)

        priors, priorDistributionsMeans, priorDistributionsStds, posteriors, posteriorDistributionsMeans, posteriorDistributionsStds, recurrentStates = [], [], [], [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState = self.recurrentModel(previousRecurrentState, previousLatentState, data.action[:, t-1])
            priorDistribution, prior = self.priorNet(recurrentState)
            posteriorDistribution, posterior = self.posteriorNet(data.encodedObservation[:, t], recurrentState)

            priors.append(prior)
            priorDistributionsMeans.append(priorDistribution.mean)
            priorDistributionsStds.append(priorDistribution.scale)
            posteriors.append(posterior)
            posteriorDistributionsMeans.append(posteriorDistribution.mean)
            posteriorDistributionsStds.append(posteriorDistribution.scale)
            recurrentStates.append(recurrentState)

            previousLatentState = posterior
            previousRecurrentState = recurrentState

        priors                      = torch.stack(priors,                       dim=1)
        priorDistributionsMeans     = torch.stack(priorDistributionsMeans,      dim=1)
        priorDistributionsStds      = torch.stack(priorDistributionsStds,       dim=1)
        posteriors                  = torch.stack(posteriors,                   dim=1)
        posteriorDistributionsMeans = torch.stack(posteriorDistributionsMeans,  dim=1)
        posteriorDistributionsStds  = torch.stack(posteriorDistributionsStds,   dim=1)
        recurrentStates             = torch.stack(recurrentStates,              dim=1)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1)

        reconstructedObservationsDistributions  =  self.decoder(fullStates)
        reconstructionLoss                      = -reconstructedObservationsDistributions.log_prob(data.observation[:, 1:]).mean()

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.reward[:, 1:]).mean()

        priorDistribution     = create_normal_dist(priorDistributionsMeans, priorDistributionsStds, event_shape=1)
        posteriorDistribution = create_normal_dist(posteriorDistributionsMeans, posteriorDistributionsStds, event_shape=1)
        klLoss = torch.mean(torch.distributions.kl_divergence(posteriorDistribution, priorDistribution)) # Change that to maxing individual dists
        klLoss = torch.max(torch.tensor(self.config.freeNats).to(self.device), klLoss)

        worldModelLoss = klLoss + reconstructionLoss + rewardLoss
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.done[:, 1:])
            worldModelLoss      += continueLoss.mean()

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

        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
        fullStates = []
        for _ in range(self.config.imaginationHorizon):
            action = self.actor(fullState)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            _, latentState = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
        fullStates = torch.stack(fullStates, dim=1)
        
        predictedRewards = self.rewardPredictor(fullStates).mean
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else self.config.discount*torch.ones_like(values)
        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.imaginationHorizon, self.device, self.config.lambda_)

        actorLoss        = -torch.mean(lambdaValues)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

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
            posterior, recurrentState = torch.zeros(1, self.latentSize, device=self.device), torch.zeros(1, self.recurrentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed=seed + self.totalEpisodes)
            encodedObservation = self.encoder(torch.from_numpy(observation).float().to(self.device))

            currentScore, stepCount, done, frames = 0, 0, False, []
            while not done:
                recurrentState = self.recurrentModel(recurrentState, posterior, action)
                _, posterior   = self.posteriorNet(encodedObservation.reshape(1, -1), recurrentState)
                action         = self.actor(torch.cat((recurrentState, posterior), -1)).detach()

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
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
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
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
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
