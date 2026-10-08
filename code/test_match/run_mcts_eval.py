"""Run MCTS depth=3 evaluation vs rule_v4, logging to file."""
import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Ant-Game"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import get_logger, redirect_stderr_to_log

log = get_logger(str(Path(__file__).resolve().parents[2] / "mcts_vs_rulev4.log"))
redirect_stderr_to_log(log)

from test_match.run_eval_seq import run_mcts_eval

log.print("=" * 60, timestamp=False)
log.print("Starting MCTS depth=3 evaluation vs rule_v4 (50 games)", timestamp=False)
log.print("=" * 60, timestamp=False)

run_mcts_eval(
    model_a="training_history/ss_20260709_191050/gen_0030.pt",
    model_b="training_history/ss_20260710_155327/gen_0101.pt",
    value_net_path="training_history/value_20260712_054444/epoch_0010.pt",
    mcts_depth=3,
    games=50,
    opponent="rule_v4",
    device="cuda",
    small_value=False,
)
