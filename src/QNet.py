import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Constants import ZONE0, ZONE1

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SpatialAttention3D(nn.Module):
    """Spatial attention module for 3D data"""
    def __init__(self, channels):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Generate attention map
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        return x * attention

class CNNQNetwork(nn.Module):
    """
    CNN-based Q-Network for processing 3D state and action representations.
    """
    def __init__(self, base_channels=32):
        super(CNNQNetwork, self).__init__()
        
        # State encoders
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Action encoders
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened dimensions
        self.fc_input_dim = (
            base_channels * ZONE0[0] * ZONE0[1] * ZONE0[2] +  # state_zone0
            base_channels * ZONE1[0] * ZONE1[1] * ZONE1[2] +  # state_zone1
            base_channels * ZONE0[0] * ZONE0[1] * ZONE0[2] +  # action_zone0
            base_channels * ZONE1[0] * ZONE1[1] * ZONE1[2]    # action_zone1
        )
        
        print(f"Total FC input dimension: {self.fc_input_dim}")
        
        # MLP for Q-value prediction
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        # Fix tensor dimensions for conv3d
        # Conv3d expects 5D input: [batch_size, channels, depth, height, width]
        
        # Handle state tensors
        if state_zone0.dim() == 4:  # [batch, depth, height, width]
            state_zone0 = state_zone0.unsqueeze(1)  # Add channel dimension
        elif state_zone0.dim() == 6:  # [1, batch, channels, depth, height, width]
            state_zone0 = state_zone0.squeeze(0)  # Remove extra first dimension
            
        if state_zone1.dim() == 4:
            state_zone1 = state_zone1.unsqueeze(1)
        elif state_zone1.dim() == 6:
            state_zone1 = state_zone1.squeeze(0)
        
        # Handle action tensors
        if action_zone0.dim() == 4:
            action_zone0 = action_zone0.unsqueeze(1)
        elif action_zone0.dim() == 6:
            action_zone0 = action_zone0.squeeze(0)
            
        if action_zone1.dim() == 4:
            action_zone1 = action_zone1.unsqueeze(1)
        elif action_zone1.dim() == 6:
            action_zone1 = action_zone1.squeeze(0)
        
        # Process state branches
        s0 = self.state_zone0_conv(state_zone0)
        s1 = self.state_zone1_conv(state_zone1)
        s0 = s0.view(s0.size(0), -1)
        s1 = s1.view(s1.size(0), -1)
        state_features = torch.cat([s0, s1], dim=1)
        
        # Process action branches
        a0 = self.action_zone0_conv(action_zone0)
        a1 = self.action_zone1_conv(action_zone1)
        a0 = a0.view(a0.size(0), -1)
        a1 = a1.view(a1.size(0), -1)
        action_features = torch.cat([a0, a1], dim=1)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=1)
        
        # Compute Q-value
        q_value = self.fc(combined_features)
        return q_value

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture that separates state value and action advantage
    for more stable learning and better policy evaluation.
    """
    def __init__(self, base_channels=32):
        super(DuelingQNetwork, self).__init__()
        
        # State encoders
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Action encoders
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened dimensions
        self.fc_input_dim = (
            base_channels * ZONE0[0] * ZONE0[1] * ZONE0[2] +  # state_zone0
            base_channels * ZONE1[0] * ZONE1[1] * ZONE1[2] +  # state_zone1
            base_channels * ZONE0[0] * ZONE0[1] * ZONE0[2] +  # action_zone0
            base_channels * ZONE1[0] * ZONE1[1] * ZONE1[2]    # action_zone1
        )
        
        print(f"Total FC input dimension: {self.fc_input_dim}")
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        # Fix tensor dimensions for conv3d
        # Conv3d expects 5D input: [batch_size, channels, depth, height, width]
        
        # Handle state tensors
        if state_zone0.dim() == 4:  # [batch, depth, height, width]
            state_zone0 = state_zone0.unsqueeze(1)  # Add channel dimension
        elif state_zone0.dim() == 6:  # [1, batch, channels, depth, height, width]
            state_zone0 = state_zone0.squeeze(0)  # Remove extra first dimension
            
        if state_zone1.dim() == 4:
            state_zone1 = state_zone1.unsqueeze(1)
        elif state_zone1.dim() == 6:
            state_zone1 = state_zone1.squeeze(0)
        
        # Handle action tensors
        if action_zone0.dim() == 4:
            action_zone0 = action_zone0.unsqueeze(1)
        elif action_zone0.dim() == 6:
            action_zone0 = action_zone0.squeeze(0)
            
        if action_zone1.dim() == 4:
            action_zone1 = action_zone1.unsqueeze(1)
        elif action_zone1.dim() == 6:
            action_zone1 = action_zone1.squeeze(0)
        
        # Process state branches
        s0 = self.state_zone0_conv(state_zone0)
        s1 = self.state_zone1_conv(state_zone1)
        s0 = s0.view(s0.size(0), -1)
        s1 = s1.view(s1.size(0), -1)
        state_features = torch.cat([s0, s1], dim=1)
        
        # Process action branches
        a0 = self.action_zone0_conv(action_zone0)
        a1 = self.action_zone1_conv(action_zone1)
        a0 = a0.view(a0.size(0), -1)
        a1 = a1.view(a1.size(0), -1)
        action_features = torch.cat([a0, a1], dim=1)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=1)
        
        # Compute value and advantage streams
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)
        
        # Combine value and advantage to get Q-value
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value

if __name__ == "__main__":
    # Quick test using dummy inputs.
    net = CNNQNetwork(base_channels=32)
    dueling_net = DuelingQNetwork(base_channels=32)
    
    dummy_state_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_state_zone1 = torch.randn(2, 1, *ZONE1)
    dummy_action_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_action_zone1 = torch.randn(2, 1, *ZONE1)
    
    q_val = net(dummy_state_zone0, dummy_state_zone1, dummy_action_zone0, dummy_action_zone1)
    print("CNNQNetwork output shape:", q_val.shape)  # Expected: [2, 1]
    print("CNNQNetwork values:", q_val)
    
    dueling_q_val = dueling_net(dummy_state_zone0, dummy_state_zone1, dummy_action_zone0, dummy_action_zone1)
    print("DuelingQNetwork output shape:", dueling_q_val.shape)  # Expected: [2, 1]
    print("DuelingQNetwork values:", dueling_q_val)
