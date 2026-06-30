"""Trace gen_0002 model (with HOLD class 23) vs ExampleAI."""
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
    'training_history/20260630_182516/gen_0002.pt',
    map_location='cpu', weights_only=True)

# Try both mean and top1 params
for label, vec in [("mean", ckpt['mean']), ("top1", ckpt['top2_params'][0])]:
    model = create_model(single_head=True)
    model.set_parameters_from_vector(vec.numpy())
    agent = NeuralAgent(model=model)
    opp = ExampleAI(seed=42)

    state = GameState.initial(seed=42, cold_handle_rule_illegal=True)
    p = 0
    class_counts = {}
    for rnd in range(MAX_ROUND):
        if state.terminal:
            break
        u = agent._choose_operations(state, p)
        v = opp.choose_operations(state, 1-p)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)
        if p == 0:
            label_key = 'HOLD' if not u else str(u[0].op_type.name)
            class_counts[label_key] = class_counts.get(label_key, 0) + 1
            if rnd < 8:
                ops_str = ' ; '.join(str(op) for op in u)
                print(f'  rnd {rnd:2d} | {ops_str if u else "HOLD"}')

    me_hp = state.bases[0].hp
    opp_hp = state.bases[1].hp
    print(f'[{label}] Final: me(hp={me_hp}) opp(hp={opp_hp}) rnd={state.round_index}')
    top_actions = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
    print(f'[{label}] Top actions: {top_actions}')
    print()
