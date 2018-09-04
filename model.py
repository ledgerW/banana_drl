import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers=3, fc1_units=256, fc2_units=128, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.layers == 2:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        if self.layers == 3:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)
        

class ConvQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, m_frames, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.Conv1 = nn.Conv2d(in_channels=m_frames, out_channels=32, kernel_size=8, stride=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3136, 512)
        self.output = nn.Linear(512, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.Conv1(state))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.output(x)
        
        
        
        
        
        
