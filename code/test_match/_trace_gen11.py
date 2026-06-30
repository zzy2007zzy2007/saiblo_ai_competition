"""Trace top1 individual from gen 11 vs ExampleAI."""
import sys, torch
sys.path = ['Ant-Game', 'code'] + sys.path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from AI.ai_example import AI as ExampleAI
from my_ai.network import create_model
from my_ai.agent import NeuralAgent

ckpt = torch.load(
    'training_history/20260630_152924/gen_0011.pt',
    map_location='cpu', weights_only=True)
model = create_model(single_head=True)
model.set_parameters_from_vector(ckpt['top2_params'][0].numpy())
agent = NeuralAgent(model=model)
opp = ExampleAI(seed=42)

state = GameState.initial(seed=42, cold_handle_rule_illegal=True)
p = 0
for rnd in range(MAX_ROUND):
    if state.terminal:
        break
    u = agent._choose_operations(state, p)
    v = opp.choose_operations(state, 1-p)
    state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)
    if p == 0:
        ops_str = ' ; '.join(str(op) for op in u)
        print(f'rnd {rnd:2d} | {ops_str}')

my_hp = state.bases[0].hp
opp_hp = state.bases[1].hp
print(f'\nFinal: me(hp={my_hp}) opp(hp={opp_hp})  rnd={state.round}')
print('WIN' if my_hp > opp_hp else ('DRAW' if my_hp == opp_hp else 'LOSS'))
