"""Diagnose mean (not top1) from checkpoint."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND, OperationType
from AI.ai_example import AI as ExampleAI
from my_ai.network import create_model
from my_ai.agent import NeuralAgent

OP_NAMES = {
    OperationType.BUILD_TOWER: "BUILD", OperationType.UPGRADE_TOWER: "UPGRADE",
    OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
    OperationType.USE_LIGHTNING_STORM: "LIGHTNING", OperationType.USE_EMP_BLASTER: "EMP",
    OperationType.USE_DEFLECTOR: "DEFLECTOR", OperationType.USE_EMERGENCY_EVASION: "EVASION",
    OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED", OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
}

def op_desc(op):
    name = OP_NAMES.get(op.op_type, f"OP_{op.op_type}")
    if op.op_type == OperationType.BUILD_TOWER:
        return f"{name}({op.arg0},{op.arg1})"
    elif op.op_type == OperationType.UPGRADE_TOWER:
        return f"{name}(id={op.arg0}->type={op.arg1})"
    elif op.op_type == OperationType.DOWNGRADE_TOWER:
        return f"{name}(id={op.arg0})"
    elif OperationType.USE_LIGHTNING_STORM <= op.op_type <= OperationType.USE_EMERGENCY_EVASION:
        return f"{name}({op.arg0},{op.arg1})"
    return name

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
param_vec = ckpt["mean"].numpy()
num_heads = ckpt.get("num_heads", 3)

model = create_model(num_heads=num_heads)
model.set_parameters_from_vector(param_vec)
agent = NeuralAgent(model=model)

state = GameState.initial(seed=0, cold_handle_rule_illegal=True)
opp = ExampleAI(seed=0)
side = "1st"

for turn in range(MAX_ROUND):
    if state.terminal:
        break
    ops = agent._choose_operations(state, 0)
    opp_ops = opp.choose_operations(state, 1)
    if ops:
        descs = [op_desc(op) for op in ops]
        print(f"  turn={turn:3d}  {' | '.join(descs)}")
    else:
        print(f"  turn={turn:3d}  HOLD")
    state.resolve_turn(ops, opp_ops)

hp_us, hp_opp = state.bases[0].hp, state.bases[1].hp
print(f"\nResult: {'WIN' if hp_us > hp_opp else 'LOSS'}  HP {int(hp_us)} vs {int(hp_opp)}  turns={turn}")
