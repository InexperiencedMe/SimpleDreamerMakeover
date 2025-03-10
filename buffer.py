import attridict
import numpy as np
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observation        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.next_observation   = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.action             = np.empty((self.capacity, action_size), dtype=np.float32)
        self.reward             = np.empty((self.capacity, 1), dtype=np.float32)
        self.done               = np.empty((self.capacity, 1), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, next_observation, done):
        self.observation[self.bufferIndex] = observation
        self.action[self.bufferIndex] = action
        self.reward[self.bufferIndex] = reward
        self.next_observation[self.bufferIndex] = next_observation
        self.done[self.bufferIndex] = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observation         = torch.as_tensor(self.observation[sampleIndex], device=self.device).float()
        next_observation    = torch.as_tensor(self.next_observation[sampleIndex], device=self.device).float()

        action  = torch.as_tensor(self.action[sampleIndex], device=self.device)
        reward  = torch.as_tensor(self.reward[sampleIndex], device=self.device)
        done    = torch.as_tensor(self.done[sampleIndex], device=self.device)

        sample = attridict({
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done})
        return sample
