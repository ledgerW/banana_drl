import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, ConvQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, ConvDQN, m_frames, layer_units, hyperams, extensions):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            hyperams (tuple): tuple of agent hyperameters; (BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.BUFFER_SIZE, self.BATCH_SIZE, self.GAMMA, self.TAU, self.LR, self.UPDATE_EVERY = hyperams
        layers, fc1, fc2, fc3 = layer_units
        self.DDQN, self.PER, self.DUELING, self.DISTRIBUTIONAL = extensions
        self.ConvDQN = ConvDQN

        if ConvDQN:
            # Conv-Q-Network
            self.qnetwork_local = ConvQNetwork(state_size, m_frames, action_size, seed).to(device)
            self.qnetwork_target = ConvQNetwork(state_size, m_frames, action_size, seed).to(device)
            self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.LR, momentum=0.95)
        else:
            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed, layers, fc1, fc2, fc3).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, layers, fc1, fc2, fc3).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(ConvDQN, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA, self.DDQN)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.ConvDQN:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))     # return as int to avoid error on Windows7
        else:
            return int(random.choice(np.arange(self.action_size)))      # return as int to avoid error on Windows7

    def learn(self, experiences, gamma, DDQN):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if DDQN:
            local_max_nextA = self.qnetwork_local(next_states).detach().argmax(1) 
             
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, local_max_nextA.unsqueeze(1))
        else:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, ConvDQN, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.ConvDQN = ConvDQN
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.ConvDQN:
            e = self.experience(torch.from_numpy(state).unsqueeze(0), action, reward, torch.from_numpy(next_state).unsqueeze(0), done)
        else:
            e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)