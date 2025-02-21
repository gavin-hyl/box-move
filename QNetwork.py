import torch
import torch.nn as nn
import torch.nn.functional as F
from Constants import ZONE0, ZONE1  # Import zone sizes in case they change

class CNNQNetwork(nn.Module):
    """
    A CNN that takes a 3D state representation and a 3D action representation,
    processes them with separate convolutional branches, and returns the Q value.
    
    Both inputs are assumed to be 5D tensors of shape:
      [batch_size, channels, depth, height, width]
    For example, if each input is a single-channel volume, then channels=1.
    """
    def __init__(self, state_channels=1, action_channels=1, base_channels=16):
        super(CNNQNetwork, self).__init__()
        
        # Convolutional branch for the state input.
        self.state_conv = nn.Sequential(
            nn.Conv3d(in_channels=state_channels, out_channels=base_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Global pooling reduces the feature map to one value per channel.
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Convolutional branch for the action input.
        self.action_conv = nn.Sequential(
            nn.Conv3d(in_channels=action_channels, out_channels=base_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=base_channels, out_channels=base_channels * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Fully connected layers that combine the extracted features and output a Q value.
        # Each branch outputs base_channels*2 features.
        self.fc = nn.Sequential(
            nn.Linear((base_channels * 2) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        """
        Forward pass of the network.
        
        Args:
            state: Tensor of shape [batch_size, state_channels, D, H, W]
            action: Tensor of shape [batch_size, action_channels, D, H, W]
        
        Returns:
            q_value: Tensor of shape [batch_size, 1]
        """
        state_features = self.state_conv(state)   # [batch, base_channels*2, 1, 1, 1]
        action_features = self.action_conv(action)  # [batch, base_channels*2, 1, 1, 1]
        
        # Flatten the features for each branch.
        state_features = state_features.view(state_features.size(0), -1)
        action_features = action_features.view(action_features.size(0), -1)
        
        # Concatenate the features from state and action branches.
        combined_features = torch.cat([state_features, action_features], dim=1)
        
        # Pass through the fully connected layers to produce the Q value.
        q_value = self.fc(combined_features)
        return q_value

if __name__ == "__main__":
    # Create an instance of the network.
    net = CNNQNetwork(state_channels=1, action_channels=1)
    
    # -------------------------------
    # Quick test using dummy inputs.
    # -------------------------------
    # Here we use a fixed shape (5,3,3) for the dummy inputs,
    # but the sizes are taken from Constants so they adapt if ZONE0 or ZONE1 change.
    dummy_state = torch.randn(2, 1, *ZONE0)   # Batch size 2, shape: (2, 1, depth, height, width)
    dummy_action = torch.randn(2, 1, *ZONE1)  # Similarly for action.
    
    q_val = net(dummy_state, dummy_action)
    print("Q value output shape:", q_val.shape)  # Should print: torch.Size([2, 1])
    print("Q values:", q_val)
    
    # --------------------------------------------------
    # Generate sample action-rewards based on Constants.
    # --------------------------------------------------
    # For demonstration, we generate a batch of sample state and action representations
    # using the zone dimensions from the Constants file.
    sample_batch_size = 4  # You can change the batch size as needed.
    
    # Generate a sample state representation.
    sample_state = torch.randn(sample_batch_size, 1, *ZONE0)
    # Generate a sample action representation.
    sample_action = torch.randn(sample_batch_size, 1, *ZONE1)
    
    # Compute the Q values (action-rewards) for these samples.
    sample_q_values = net(sample_state, sample_action)
    print("\nSample action-rewards (Q values):")
    print(sample_q_values)
