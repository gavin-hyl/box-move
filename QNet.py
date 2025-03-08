import torch
import torch.nn as nn
from Constants import ZONE0, ZONE1

class CNNQNetwork(nn.Module):
    """
    A simpler CNN for Q-value estimation with reduced network complexity.
    Processes 3D state and action representations in two zones (ZONE0 and ZONE1)
    with one convolutional layer per branch.
    """
    def __init__(self, base_channels=16):
        super(CNNQNetwork, self).__init__()
        
        # --- State branches: one conv layer each ---
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # --- Action branches: one conv layer each ---
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Compute the flattened dimension for each branch using dummy inputs.
        dummy_state_zone0 = torch.zeros(1, 1, *ZONE0)
        dummy_state_zone1 = torch.zeros(1, 1, *ZONE1)
        dummy_action_zone0 = torch.zeros(1, 1, *ZONE0)
        dummy_action_zone1 = torch.zeros(1, 1, *ZONE1)
        
        out_state0 = self.state_zone0_conv(dummy_state_zone0)
        dim_state0 = out_state0.view(1, -1).shape[1]
        
        out_state1 = self.state_zone1_conv(dummy_state_zone1)
        dim_state1 = out_state1.view(1, -1).shape[1]
        
        out_action0 = self.action_zone0_conv(dummy_action_zone0)
        dim_action0 = out_action0.view(1, -1).shape[1]
        
        out_action1 = self.action_zone1_conv(dummy_action_zone1)
        dim_action1 = out_action1.view(1, -1).shape[1]
        
        total_fc_input_dim = dim_state0 + dim_state1 + dim_action0 + dim_action1
        print("Total FC input dimension:", total_fc_input_dim)
        
        # Fully connected layers: a simpler, smaller head.
        self.fc = nn.Sequential(
            nn.Linear(total_fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        # Process state branches.
        s0 = self.state_zone0_conv(state_zone0)
        s1 = self.state_zone1_conv(state_zone1)
        s0 = s0.view(s0.size(0), -1)
        s1 = s1.view(s1.size(0), -1)
        state_features = torch.cat([s0, s1], dim=1)
        
        # Process action branches.
        a0 = self.action_zone0_conv(action_zone0)
        a1 = self.action_zone1_conv(action_zone1)
        a0 = a0.view(a0.size(0), -1)
        a1 = a1.view(a1.size(0), -1)
        action_features = torch.cat([a0, a1], dim=1)
        
        # Combine state and action features and compute the Q-value.
        combined_features = torch.cat([state_features, action_features], dim=1)
        q_value = self.fc(combined_features)
        return q_value

if __name__ == "__main__":
    # Quick test using dummy inputs.
    net = CNNQNetwork(base_channels=16)
    dummy_state_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_state_zone1 = torch.randn(2, 1, *ZONE1)
    dummy_action_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_action_zone1 = torch.randn(2, 1, *ZONE1)
    q_val = net(dummy_state_zone0, dummy_state_zone1, dummy_action_zone0, dummy_action_zone1)
    print("Q value output shape:", q_val.shape)  # Expected: [2, 1]
    print("Q values:", q_val)
