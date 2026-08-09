"""Run a match between two external C++ AIs (champion magica_v3 vs runner-up
ai_cpp_lure_v4) using the C++ game binary as the judge.

The game binary (Ant-Game/game/output/main) is byte-identical logic to our
embedded cpp_engine (code/cpp_engine), so this tests whether the two AIs play
correctly under the C++ engine's protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GAME_BIN = REPO / "Ant-Game" / "game" / "output" / "main"
CHAMPION = REPO / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3.exe"
RUNNERUP = REPO / "其他版本ai" / "saiblo-30th-AI" / "Game1" / "antgame_ai_cpp" / "cpp_lure_v4" / "build" / "ai_cpp_lure_v4.exe"
GAME_DIR = REPO / "Ant-Game" / "game"
TIMEOUT = 30.0


def packet(payload: object) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def read_exact(stream, size: int, proc, label: str, timeout: float = TIMEOUT) -> bytes:
    # BufferedReader.read(n) blocks until n bytes or EOF — deterministic framing
    # on Windows pipes (select() only works on sockets there).  A hung AI is
    # caught by the overall wall-clock watchdog in main().
    data = stream.read(size)
    if len(data) < size:
        raise EOFError(f"{label} EOF (exit {proc.poll()})")
    return data


def read_game_packet(game):
    size = struct.unpack(">I", read_exact(game.stdout, 4, game, "game size"))[0]
    obj = struct.unpack(">i", read_exact(game.stdout, 4, game, "game obj"))[0]
    payload = read_exact(game.stdout, size, game, "game payload")
    return obj, payload


def read_ai_packet(ai, name: str) -> bytes:
    size = struct.unpack(">I", read_exact(ai.stdout, 4, ai, f"{name} size"))[0]
    payload = read_exact(ai.stdout, size, ai, f"{name} payload")
    return struct.pack(">I", size) + payload


def write_all(stream, payload: bytes) -> None:
    stream.write(payload)
    stream.flush()


def launch(proc: Path, stderr_path: Path):
    handle = stderr_path.open("wb")
    # add C:/mingw64/bin to PATH so MinGW runtime DLLs resolve
    env = dict(os.environ)
    env["PATH"] = r"C:/mingw64/bin" + os.pathsep + env.get("PATH", "")
    return subprocess.Popen(
        [str(proc)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=handle, env=env,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ai0", type=Path, default=CHAMPION, help="player 0 binary")
    ap.add_argument("--ai1", type=Path, default=RUNNERUP, help="player 1 binary")
    ap.add_argument("--workdir", type=Path, default=REPO / "training_history" / "cpp_ai_matches")
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = let the game end naturally")
    args = ap.parse_args()

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    replay_path = workdir / f"replay_seed{args.seed}.json"

    names = ["ai0(champion)", "ai1(runnerup)"]
    game = None
    ai = [None, None]
    result = {"seed": args.seed, "ai0": str(args.ai0.name), "ai1": str(args.ai1.name)}

    try:
        ai[0] = launch(args.ai0, workdir / "ai0.stderr.log")
        ai[1] = launch(args.ai1, workdir / "ai1.stderr.log")
        game = subprocess.Popen(
            [str(GAME_BIN)], cwd=str(GAME_DIR),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(workdir / "game.stderr.log").open("wb"),
        )

        init = {"player_list": [1, 1], "player_num": 2,
                "config": {"random_seed": args.seed}, "replay": replay_path.as_posix()}
        write_all(game.stdin, packet(init))

        rounds = 0
        start = time.monotonic()
        wall_limit = 1500.0
        while True:
            if time.monotonic() - start > wall_limit:
                raise TimeoutError(f"match exceeded {wall_limit:.0f}s wall clock")
            obj, payload = read_game_packet(game)
            if obj in (0, 1):
                write_all(ai[obj].stdin, payload)
                continue
            message = json.loads(payload.decode("utf-8"))
            if isinstance(message, dict) and "player" in message and "content" in message:
                for player, content in zip(message["player"], message["content"]):
                    write_all(ai[int(player)].stdin, content.encode("utf-8"))
            if isinstance(message, dict) and message.get("listen"):
                for player in message["listen"]:
                    ai_packet = read_ai_packet(ai[int(player)], names[int(player)])
                    reply = {"player": int(player), "content": ai_packet.decode("latin1"), "time": 0}
                    write_all(game.stdin, packet(reply))
                    rounds += 1
            if isinstance(message, dict) and "end_state" in message:
                result["end_state"] = message["end_state"]
                result["end_info"] = message.get("end_info")
                break
            if args.max_rounds and rounds >= args.max_rounds * 2:
                result["forced_stop"] = rounds
                break

        game.wait(timeout=5)
        for p in ai:
            if p is not None:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.terminate()
        result["game_rc"] = game.returncode
        result["ai0_rc"] = ai[0].returncode
        result["ai1_rc"] = ai[1].returncode
        result["rounds_recorded"] = _replay_len(replay_path)
        result["game_stderr"] = _read(workdir / "game.stderr.log")[-1500:]
        result["ai0_stderr"] = _read(workdir / "ai0.stderr.log")[-800:]
        result["ai1_stderr"] = _read(workdir / "ai1.stderr.log")[-800:]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        if game is not None:
            result["game_rc"] = game.poll()
        for i, p in enumerate(ai):
            if p is not None:
                result[f"ai{i}_rc"] = p.poll()
        result["game_stderr"] = _read(workdir / "game.stderr.log")[-1500:]
        result["ai0_stderr"] = _read(workdir / "ai0.stderr.log")[-800:]
        result["ai1_stderr"] = _read(workdir / "ai1.stderr.log")[-800:]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


def _replay_len(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return 0


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
