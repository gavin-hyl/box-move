import torch
import torch.nn as nn
import torch.nn.functional as F
from Constants import ZONE0, ZONE1

class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow through deep networks
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out

class CNNQNetwork(nn.Module):
    """
    A simpler CNN for Q-value estimation with reduced network complexity.
    Processes 3D state and action representations in two zones (ZONE0 and ZONE1)
    with one convolutional layer per branch.
    """
    def __init__(self, base_channels=16):
        super(CNNQNetwork, self).__init__()
        
        # --- State branches: deeper architecture ---
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # --- Action branches: deeper architecture ---
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
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
        
        # Fully connected layers: deeper with Dropout for regularization
        self.fc = nn.Sequential(
            nn.Linear(total_fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply better weight initialization for improved convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
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

# More advanced DuelingQNetwork with value and advantage streams
class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture that separates state value and action advantage
    for more stable learning and better policy evaluation.
    """
    def __init__(self, base_channels=16):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extractor for states
        self.state_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        self.state_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Shared feature extractor for actions
        self.action_zone0_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        self.action_zone1_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Compute the flattened dimension for each branch using dummy inputs
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
        print("Total FC input dimension (Dueling):", total_fc_input_dim)
        
        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(total_fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value output
        )
        
        # Advantage stream - estimates advantages of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(total_fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single advantage output for the specific action
        )
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Apply better weight initialization for improved convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
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
        
        # Compute value and advantage
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)
        
        # Combine value and advantage to get Q-value
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_value = value + advantage
        
        return q_value

if __name__ == "__main__":
    # Quick test using dummy inputs.
    net = CNNQNetwork(base_channels=16)
    dueling_net = DuelingQNetwork(base_channels=16)
    
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
