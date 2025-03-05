import torch
import torch.nn as nn
from QNet import CNNQNetwork
from Constants import ZONE0, ZONE1

class QNetEnsemble(nn.Module):
    """
    A deep ensemble of CNN Q-networks with learnable ensemble weights.
    Instead of simply averaging the predictions, this class learns a set of weights
    (via a softmax over trainable logits) to combine the Q-value estimates from each network.
    """
    def __init__(self, ensemble_size=5, base_channels=16, device='cpu'):
        super(QNetEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.device = device
        
        # Create an ensemble (ModuleList) of CNNQNetwork instances.
        self.networks = nn.ModuleList([
            CNNQNetwork(base_channels=base_channels).to(device)
            for _ in range(ensemble_size)
        ])
        
        # Learnable logits for ensemble weights.
        # These logits will be converted to weights via softmax.
        # Initialized to zeros, so the initial weights are uniform.
        self.ensemble_logits = nn.Parameter(torch.zeros(ensemble_size, device=device))

    def forward(self, state_zone0, state_zone1, action_zone0, action_zone1):
        """
        Forward pass: Computes Q-value estimates from each network and aggregates them
        using learned weights.
        
        Args:
            state_zone0, state_zone1: Tensors of shape [batch, 1, D, H, W] for each zone.
            action_zone0, action_zone1: Tensors of shape [batch, 1, D, H, W] for the action.
            
        Returns:
            weighted_q: The weighted Q-value (tensor of shape [batch, 1]) after combining
                        the ensemble outputs.
            all_q: A tensor of all ensemble predictions (shape [ensemble_size, batch, 1]).
            weights: The softmax-normalized ensemble weights (tensor of shape [ensemble_size]).
        """
        predictions = []
        for net in self.networks:
            q_val = net(state_zone0, state_zone1, action_zone0, action_zone1)
            predictions.append(q_val)
        # Shape: [ensemble_size, batch, 1]
        all_q = torch.stack(predictions, dim=0)
        
        # Compute ensemble weights via softmax.
        weights = torch.softmax(self.ensemble_logits, dim=0)  # shape: [ensemble_size]
        # Reshape for broadcasting: [ensemble_size, 1, 1]
        weights = weights.view(self.ensemble_size, 1, 1)
        
        # Compute the weighted average Q-value.
        weighted_q = torch.sum(weights * all_q, dim=0)
        return weighted_q, all_q, weights


    def load_state_dicts(self, state_dict_list):
        """
        Loads a list of state dictionaries—one for each network in the ensemble.
        
        Args:
            state_dict_list: List of state dicts with length equal to ensemble_size.
        """
        if len(state_dict_list) != self.ensemble_size:
            raise ValueError("Number of state dicts must match ensemble_size.")
        for net, sd in zip(self.networks, state_dict_list):
            net.load_state_dict(sd)

    def save_state_dicts(self, base_path):
        """
        Saves each network's state dict using a base filename.
        For example, if base_path is 'models/ensemble_qnet', the networks will be saved as
        'models/ensemble_qnet_0.pth', 'models/ensemble_qnet_1.pth', etc.
        
        Args:
            base_path: Base filename (without index or extension).
        """
        for i, net in enumerate(self.networks):
            torch.save(net.state_dict(), f"{base_path}_{i}.pth")


if __name__ == "__main__":
    # Quick test using dummy inputs.
    device = 'cpu'
    ensemble = QNetEnsemble(ensemble_size=3, device=device)
    
    # Create dummy state tensors with shapes matching ZONE0 and ZONE1.
    state_zone0 = torch.randn(2, 1, *ZONE0, device=device)
    state_zone1 = torch.randn(2, 1, *ZONE1, device=device)
    # Dummy action tensors.
    action_zone0 = torch.randn(2, 1, *ZONE0, device=device)
    action_zone1 = torch.randn(2, 1, *ZONE1, device=device)
    
    weighted_q, all_q, weights = ensemble(state_zone0, state_zone1, action_zone0, action_zone1)
    print("Weighted Q values:", weighted_q)
    print("All ensemble predictions shape:", all_q.shape)
    print("Learned ensemble weights:", weights)
