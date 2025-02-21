import torch
import torch.nn as nn
from Constants import ZONE0, ZONE1

class CNNQNetwork(nn.Module):
    """
    A CNN that takes a 3D state and action representation and processes
    each zone (ZONE0 and ZONE1) with separate convolutional branches.
    
    The inputs are 5D tensors of shape:
      [batch_size, channels, depth, height, width]
    where channels=2 (channel 0: ZONE0, channel 1: ZONE1). Each channel is processed
    independently, then the features are concatenated and passed through fully connected layers
    to produce the Q value.
    """
    def __init__(self, state_channels=2, action_channels=2, base_channels=16):
        super(CNNQNetwork, self).__init__()
        
        # --- State branches ---
        # Branch for ZONE0 from the state input.
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        # Branch for ZONE1 from the state input.
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # --- Action branches ---
        # Branch for ZONE0 from the action input.
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        # Branch for ZONE1 from the action input.
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # After adaptive pooling, each branch produces a tensor of shape [batch, base_channels*2, 1, 1, 1].
        # Flattening gives base_channels*2 features per branch.
        # For state, we have two branches -> total state features = base_channels*4.
        # Similarly for action.
        # Concatenated, the combined feature vector has (base_channels*4)*2 = base_channels*8 features.
        fc_input_dim = base_channels * 8
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        """
        Forward pass of the network.
        
        Args:
            state: Tensor of shape [batch_size, 2, D, H, W] where
                   state[:,0] is the ZONE0 channel and state[:,1] is the ZONE1 channel.
            action: Tensor of shape [batch_size, 2, D, H, W] with a similar channel structure.
        
        Returns:
            q_value: Tensor of shape [batch_size, 1]
        """
        # Split state channels.
        state_zone0 = state[:, 0:1, :, :, :]  # Shape: [batch, 1, D, H, W]
        state_zone1 = state[:, 1:2, :, :, :]  # Shape: [batch, 1, D, H, W]
        # Process state channels.
        state_zone0_features = self.state_zone0_conv(state_zone0)
        state_zone1_features = self.state_zone1_conv(state_zone1)
        state_zone0_features = state_zone0_features.view(state_zone0_features.size(0), -1)
        state_zone1_features = state_zone1_features.view(state_zone1_features.size(0), -1)
        state_features = torch.cat([state_zone0_features, state_zone1_features], dim=1)  # [batch, base_channels*4]
        
        # Split action channels.
        action_zone0 = action[:, 0:1, :, :, :]
        action_zone1 = action[:, 1:2, :, :, :]
        # Process action channels.
        action_zone0_features = self.action_zone0_conv(action_zone0)
        action_zone1_features = self.action_zone1_conv(action_zone1)
        action_zone0_features = action_zone0_features.view(action_zone0_features.size(0), -1)
        action_zone1_features = action_zone1_features.view(action_zone1_features.size(0), -1)
        action_features = torch.cat([action_zone0_features, action_zone1_features], dim=1)  # [batch, base_channels*4]
        
        # Combine state and action features.
        combined_features = torch.cat([state_features, action_features], dim=1)  # [batch, base_channels*8]
        
        q_value = self.fc(combined_features)
        return q_value

if __name__ == "__main__":
    # Quick test using dummy inputs.
    net = CNNQNetwork(state_channels=2, action_channels=2)
    dummy_state = torch.randn(2, 2, *ZONE0)   # For example, with ZONE0 dimensions
    dummy_action = torch.randn(2, 2, *ZONE1)    # For example, with ZONE1 dimensions
    q_val = net(dummy_state, dummy_action)
    print("Q value output shape:", q_val.shape)  # Expected: torch.Size([2, 1])
    print("Q values:", q_val)
