import torch
import torch.nn as nn
from Constants import ZONE0, ZONE1

class CNNQNetwork(nn.Module):
    """
    A CNN that takes a 3D state and action representation and processes
    each zone (ZONE0 and ZONE1) with separate convolutional branches.
    """
    def __init__(self, base_channels=32):
        super(CNNQNetwork, self).__init__()
        
        # --- State branches ---
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # --- Action branches ---
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Compute the flattened dimension for each branch using dummy inputs.
        dummy_state_zone0 = torch.zeros(1, 1, *ZONE0)  # shape: [1,1,5,4,3]
        dummy_state_zone1 = torch.zeros(1, 1, *ZONE1)  # shape: [1,1,3,3,3]
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
        print("Total FC input dimension:", total_fc_input_dim)  # Expected: 640

        self.fc = nn.Sequential(
            nn.Linear(total_fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        """
        Forward pass that accepts state and action representations as separate tensors.
        Each input is assumed to be of shape [batch, 1, D, H, W].
        """
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
        
        # Combine features and compute Q value.
        combined_features = torch.cat([state_features, action_features], dim=1)
        q_value = self.fc(combined_features)
        return q_value
        

if __name__ == "__main__":
    # Quick test using dummy inputs.
    net = CNNQNetwork(base_channels=32)
    # Create dummy inputs with the expected dimensions:
    # For state: zone0 is ZONE0 = (5,4,3) and zone1 is ZONE1 = (3,3,3)
    dummy_state_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_state_zone1 = torch.randn(2, 1, *ZONE1)
    dummy_action_zone0 = torch.randn(2, 1, *ZONE0)
    dummy_action_zone1 = torch.randn(2, 1, *ZONE1)
    q_val = net(dummy_state_zone0, dummy_state_zone1, dummy_action_zone0, dummy_action_zone1)
    print("Q value output shape:", q_val.shape)  # Expected: torch.Size([2, 1])
    print("Q values:", q_val)
