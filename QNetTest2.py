from BoxMoveEnvGym import BoxMoveEnvGym
from Box import Box
from QNet import CNNQNetwork
import torch
from BoxAction import BoxAction

bmeg = BoxMoveEnvGym()
bmeg.set_boxes([
    Box([0, 0, 0], [2, 2, 2], 0, 1),
    Box([2, 0, 0], [2, 2, 2], 0, 1),
])

qnet = CNNQNetwork()
qnet.load_state_dict(torch.load("models/vanilla/cnn_qnet_epoch45.pth"))

action_good = BoxAction([0, 0, 0], [0, 0, 0])
action_bad = BoxAction([0, 0, 0], [1, 0, 0])

action_good_3d = bmeg.env.action_3d(action_good)
action_bad_3d = bmeg.env.action_3d(action_bad)
state = bmeg.env.state_3d()

state_zone0 = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
state_zone1 = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
action_good_zone0 = torch.tensor(action_good_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
action_good_zone1 = torch.tensor(action_good_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
action_bad_zone0 = torch.tensor(action_bad_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
action_bad_zone1 = torch.tensor(action_bad_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Evaluate Q-values for each action using the Q-network
with torch.no_grad():
    q_good = qnet(state_zone0, state_zone1, action_good_zone0, action_good_zone1)
    q_bad = qnet(state_zone0, state_zone1, action_bad_zone0, action_bad_zone1)

print("Q value for action_good (move box from [0,0,0] to [0,0,0]):", q_good.item())
print("Q value for action_bad (move box from [0,0,0] to [1,0,0]):", q_bad.item())
