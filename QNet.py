import torch
import torch.nn as nn
from Constants import ZONE0, ZONE1

class CNNQNetwork(nn.Module):
    """
    A CNN that takes a 3D state and action representation and processes
    each zone (ZONE0 and ZONE1) with separate convolutional branches.
    """
    def __init__(self, base_channels=16):
        super(CNNQNetwork, self).__init__()
        
        # --- State branches ---
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # --- Action branches ---
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Use dummy inputs to determine the flattened dimension.
        dummy_state = torch.zeros(1, 1, *ZONE0)  # shape: [1,1,5,4,3]
        dummy_action = torch.zeros(1, 1, *ZONE0) # using same dims for simplicity
        # Forward through one branch:
        out_state0 = self.state_zone0_conv(dummy_state)
        flat_state0 = out_state0.view(1, -1)  # should be [1, base_channels*2]
        
        # Do it for one state branch and one action branch.
        fc_input_dim_branch = flat_state0.shape[1]  # number of features per branch
        
        # We have four branches (state_zone0, state_zone1, action_zone0, action_zone1).
        total_fc_input_dim = fc_input_dim_branch * 4
        
        print("Total FC input dimension:", total_fc_input_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(total_fc_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        """
        Forward pass that accepts state and action representations as separate tensors.
        Each input is assumed to be of shape [batch, 1, D, H, W], where D, H, W may differ
        between zone0 and zone1.
        """
        # Process state zones separately.
        s0 = self.state_zone0_conv(state_zone0)  # [batch, base_channels*2, 1,1,1]
        s1 = self.state_zone1_conv(state_zone1)
        s0 = s0.view(s0.size(0), -1)
        s1 = s1.view(s1.size(0), -1)
        state_features = torch.cat([s0, s1], dim=1)
        
        # Process action zones separately.
        a0 = self.action_zone0_conv(action_zone0)
        a1 = self.action_zone1_conv(action_zone1)
        a0 = a0.view(a0.size(0), -1)
        a1 = a1.view(a1.size(0), -1)
        action_features = torch.cat([a0, a1], dim=1)
        
        # Combine features and compute Q value.
        combined_features = torch.cat([state_features, action_features], dim=1)
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