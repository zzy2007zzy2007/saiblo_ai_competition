"""Play gen_6 top1 vs rule_v4 with per-turn action logs (unpackaged, direct import)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (_ROOT / "Ant-Game", _ROOT / "code", _ROOT / "其他版本ai" / "rule_v4"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent
from ai import AI as RuleV4


def op_str(op):
    if op is None:
        return "HOLD"
    if op.op_type.name == "BUILD_TOWER":
        return f"BUILD({op.arg0},{op.arg1})"
    if op.op_type.name == "DOWNGRADE_TOWER":
        return f"DOWNGRADE(id={op.arg0})"
    if op.op_type.name == "UPGRADE_TOWER":
        return f"UPGRADE(id={op.arg0}->type={op.arg1})"
    if op.op_type.name in ("USE_LIGHTNING_STORM", "USE_EMP_BOMB", "USE_GRAVITY_DEFLECTOR", "USE_EMERGENCY_EVASION"):
        return f"{op.op_type.name.split('_')[1].title()}({op.arg0},{op.arg1})"
    return op.op_type.name


def main():
    ckpt = torch.load(str(_ROOT / "training_history" / "20260702_005921" / "gen_0006.pt"),
                      map_location="cpu", weights_only=True)
    model = create_model(num_heads=3)
    model.set_parameters_from_vector(ckpt["top2_params"][0].numpy())
    neural = NeuralAgent(model=model)
    rule = RuleV4()

    state = GameState.initial(seed=0, cold_handle_rule_illegal=True)

    print(f"{'Turn':>5}  {'Neural':<55}  {'RuleV4':<55}")
    print("-" * 120)

    for turn in range(MAX_ROUND):
        if state.terminal:
            break

        u = neural._choose_operations(state, 0)
        v = rule.choose_operations(state, 1)

        u_str = ", ".join(op_str(op) for op in (u or [])) if u else "HOLD"
        v_str = ", ".join(op_str(op) for op in (v or [])) if v else "HOLD"

        if turn < 20 or (turn < 200 and turn % 20 == 0):
            print(f"{turn:>5}  {u_str:<55}  {v_str:<55}")

        state.resolve_turn(u or [], v or [])

    h0, h1 = state.bases[0].hp, state.bases[1].hp
    print("-" * 120)
    print(f"Result: Neural {h0} vs RuleV4 {h1}  ->  {'Neural WIN' if h0 > h1 else 'RuleV4 WIN'}")
    print(f"Turns: {turn+1}")


if __name__ == "__main__":
    main()
