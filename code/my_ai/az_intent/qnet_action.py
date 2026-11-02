"""动作价值 Q 网络诊断（2026-11-02）：Q(s, a) = 值网 encoder(state) + 动作多重集 MLP。

## 动机（docs/qnet_action_gradient_20261102.md §1）
现役值网 V(s) 只吃局面（42 stats + board 通道），读不懂"降级/建塔/闪电"各自好坏。
本诊断训一个 Q(s, a)：把"动作"喂进输入，用**同款价值标签**回归训练，再用 ∂Q/∂a 读网络对
每类动作的"边际价值"判断。

判读逻辑：
  * 若 Q 网能学会"闪电正、花钱类动作(建塔/升级)在金币<90 时负" ⇒ 病在值网架构（只吃局面不吃动作）；
  * 若 Q 网也学不会（∂Q/∂a 是噪声 / 无金币分层） ⇒ 病在训练数据/标签（标签给不出动作级监督）。

## 标签（多臂，照 docs/value_net_retrain_v1_20261101.md 的标定）
  terminal : 终局血量差 clip((hp_self-hp_enemy)/20,±1)，一局一值（对决策级无监督）
  kgeo50   : abs + kgeo τ50，label_scale 1.5（逐帧 exp 加权未来血量差，削"抄当前血量"捷径）
  abs50    : abs + geo  τ50，label_scale 1.5（逐帧，但 geo 权重仍贴近当前血量）
  relk50   : rel + kgeo τ50，label_scale 3.4（逐帧，预测"未来优势变化"）

## 动作编码（24 维多重集）
类通道语义照 `unified_cand.py` 顶部注释（0-15 塔型 / 16 降级 / 17-20 超武(闪电=17) / 21-22 基地升级 / 23 HOLD）。
chosen bundle 是 op 三元组 `[op_type, arg0, arg1]`，这里做 op→类 的**确定式**映射：
  op 11 BUILD_TOWER            → 类 0（建 Basic，别名塌缩到"建塔"）
  op 12 UPGRADE_TOWER(id,step) → 类 = CHANNEL_TO_TOWER_TYPE 的反查(step)（step=PRODUCER(4) 归 13）
  op 13 DOWNGRADE_TOWER        → 类 16
  op 21 USE_LIGHTNING_STORM    → 类 17   op 22 USE_EMP_BLASTER → 类 18
  op 23 USE_DEFLECTOR          → 类 19   op 24 USE_EMERGENCY_EVASION → 类 20
  op 31 UPGRADE_GENERATION_SPEED → 类 21  op 32 UPGRADE_GENERATED_ANT → 类 22
  空 bundle（HOLD）            → 类 23 = 1
多位 = 该类在多重集里的数量（例 `[降级,降级,闪电]` → 第16维=2、第17维=1）。

## 用法
    PY=D:/anaconda3/envs/pytorch-gpu/python.exe; S=code/my_ai/az_intent/qnet_action.py
    $PY -u $S --build --data training_history/collect_v2_s1 --cache training_history/qnet_action/cache
    for M in terminal kgeo50 abs50 relk50; do
      $PY -u $S --train --cache training_history/qnet_action/cache --label-mode $M \
          --ckpt checkpoints/posnet_A_k5_m32.pt --epochs 4 \
          --out training_history/qnet_action/qnet_$M.pt
      $PY -u $S --probe --cache training_history/qnet_action/cache --label-mode $M \
          --ckpt checkpoints/posnet_A_k5_m32.pt --model training_history/qnet_action/qnet_$M.pt \
          --out training_history/qnet_action/probe_$M.npz
    done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_CLASS = 24
HOLD = 23
DOWNGRADE = 16
LIGHTNING = 17
CD_IDX = 26     # stats 里"己方闪电冷却"的索引（features.py encode_stats: extras[4]=cd/cooldown → stats[22+4]）
BUILD_OP, UPGRADE_OP, DOWNGRADE_OP = 11, 12, 13
SUPER_OP = {21: 17, 22: 18, 23: 19, 24: 20}   # op_type -> class（闪电/EMP/偏转/急避）
BASE_OP = {31: 21, 32: 22}                     # op_type -> class（基地升级）

# 升级类里**实际在 chosen 数据里出现过的**（有训练信号）：1=HEAVY / 5=QUICK / 9=MORTAR / 13=PRODUCER
UPGRADE_SEEN_CLASSES = (1, 5, 9, 13)
UPGRADE_NOISE_CLASSES = tuple(c for c in range(1, 16) if c not in UPGRADE_SEEN_CLASSES)

CLASS_NAME = {0: "建塔Build", 16: "降级Downgrade", 17: "闪电Lightning", 18: "EMP",
              19: "偏转Deflector", 20: "急避Evasion", 21: "基地产速", 22: "基地产蚁",
              23: "HOLD"}

# 标签臂（照 docs/value_net_retrain_v1_20261101.md §1）
LABEL_MODES = {
    "terminal": {"kind": "terminal"},
    "kgeo50": {"kind": "exp", "tau": 50.0, "scale": 1.5, "mode": "abs", "weight": "kgeo"},
    "abs50": {"kind": "exp", "tau": 50.0, "scale": 1.5, "mode": "abs", "weight": "geo"},
    "relk50": {"kind": "exp", "tau": 50.0, "scale": 3.4, "mode": "rel", "weight": "kgeo"},
}


def _tower_type_to_class() -> dict:
    from my_ai.decoder import CHANNEL_TO_TOWER_TYPE
    from SDK.utils.constants import TowerType
    m = {int(t): int(c) for c, t in CHANNEL_TO_TOWER_TYPE.items()}
    m.setdefault(int(TowerType.PRODUCER), 13)   # PRODUCER(4) 是 13/14/15 的第一跳 ⇒ 归 13
    return m


_TOWER_TYPE_TO_CLASS = _tower_type_to_class()


def encode_ops(ops) -> np.ndarray:
    """chosen bundle（op 三元组列表）→ 24 维多重集向量（float32）。空 bundle = HOLD(23)=1。"""
    v = np.zeros(N_CLASS, dtype=np.float32)
    if not ops:
        v[HOLD] = 1.0
        return v
    for op in ops:
        t = int(op[0])
        if t == BUILD_OP:
            v[0] += 1
        elif t == UPGRADE_OP:
            v[_TOWER_TYPE_TO_CLASS[int(op[2])]] += 1
        elif t == DOWNGRADE_OP:
            v[DOWNGRADE] += 1
        elif t in SUPER_OP:
            v[SUPER_OP[t]] += 1
        elif t in BASE_OP:
            v[BASE_OP[t]] += 1
        else:
            print(f"[qnet] ⚠️ 未知 op_type {t}，跳过（ops={ops}）", file=sys.stderr, flush=True)
    return v


def compute_labels(stats: np.ndarray, player: np.ndarray, mode: str) -> np.ndarray:
    """给一局（或全部帧）算标签。`stats` (N,42) `player` (N,)。terminal 用 npz 的 value_target。"""
    cfg = LABEL_MODES[mode]
    if cfg["kind"] == "terminal":
        raise ValueError("terminal 标签直接取 npz 的 value_target，不走这里")
    from my_ai.az_intent.value_warmup import _exp_weighted_labels
    return _exp_weighted_labels(stats, player, tau=cfg["tau"], label_scale=cfg["scale"],
                                label_mode=cfg["mode"], label_weight=cfg["weight"])


# ----------------------------------------------------------------------------- 模型

class QActionNet(nn.Module):
    """Q(s, a) = 冻结值网 encoder(state) → state_emb，动作 24 维 → 小 MLP → action_emb，
    concat 后过一个 Q head → 单值。值网 encoder **冻结且 eval**（BN 用 running 统计，= 部署行为）。"""

    def __init__(self, value_model, action_dim: int = N_CLASS,
                 action_emb: int = 64, head_dim: int = 128):
        super().__init__()
        self.encoder = value_model
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.state_dim = int(value_model.LATENT_DIM) * (int(value_model.value_mult) + 1)
        self.action_mlp = nn.Sequential(nn.Linear(action_dim, action_emb), nn.ReLU())
        self.q_head = nn.Sequential(
            nn.Linear(self.state_dim + action_emb, head_dim), nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def train(self, mode: bool = True):
        """冻结 encoder：即使 `model.train()` 也把 encoder 强制拉回 eval，防止 BN running
        统计量在训练时漂移（否则 reload 后 encoder 与训练时不一致，探针梯度失真）。"""
        super().train(mode)
        self.encoder.eval()
        return self

    def state_embed(self, board, stats) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(board, stats)["state_emb"]

    def forward(self, board, stats, action) -> torch.Tensor:
        state_emb = self.state_embed(board, stats)      # 常量（对 action 无梯度）
        a = self.action_mlp(action)
        x = torch.cat([state_emb, a], dim=1)
        return self.q_head(x).squeeze(-1)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ----------------------------------------------------------------- 部署：Q 网打分器
class QNetScorer:
    """部署用：给一个局面 + 一批动作多重集向量**直接前向**求 Q(s,a)（不取偏导）。

    state_emb 只算一次（encoder 与 action 无关），动作向量批量过 `action_mlp` + `q_head`。
    """

    def __init__(self, model: "QActionNet", feature_extractor, device):
        self.model = model                  # QActionNet（已 eval）
        self.feat = feature_extractor
        self.device = torch.device(device)
        self._zeros = np.zeros(96)

    @torch.no_grad()
    def score_actions(self, state, player: int, action_vecs) -> np.ndarray:
        """`action_vecs`: (N, 24) float32/float64 → (N,) float64 Q(s,a)。"""
        obs = self.feat.encode_observation(state, int(player), self._zeros)
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float().to(self.device)
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float().to(self.device)
        se = self.model.state_embed(board, stats)          # (1, state_dim)
        a = torch.from_numpy(np.asarray(action_vecs, dtype=np.float32)).to(self.device)
        act_emb = self.model.action_mlp(a)                 # (N, action_emb)
        x = torch.cat([se.expand(act_emb.shape[0], -1), act_emb], dim=1)
        return self.model.q_head(x).squeeze(-1).cpu().numpy().astype(np.float64)


def load_qnet_scorer(model_path: str, feature_extractor, device: str = "cpu") -> QNetScorer:
    """从 Q 网 checkpoint 建部署打分器（device 默认 cpu，与 flat 搜索逐位一致）。

    checkpoint 里记录了 `encoder_ckpt`（Q 网训练时冻结的那个值网 encoder）⇒ 从这里重载，
    不依赖调用方的 `AZAI_CKPT`/`AZAI_VALUE_CKPT` 覆盖。
    """
    dev = torch.device(device)
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    encoder_ckpt = sd.get("encoder_ckpt", "checkpoints/posnet_A_k5_m32.pt")
    from my_ai.az_intent.az_selfplay import load_three_models
    _, _, value_model = load_three_models(encoder_ckpt)
    model = QActionNet(value_model).to(dev)
    model.action_mlp.load_state_dict(sd["action_mlp"])
    model.q_head.load_state_dict(sd["q_head"])
    model.eval()
    return QNetScorer(model, feature_extractor, dev)


# ----------------------------------------------------------------------------- 数据

def _iter_games(data_dir: Path):
    for f in sorted(data_dir.glob("az_selfplay_seed9*.cand.jsonl")):
        stem = f.name.replace(".cand.jsonl", "")
        yield stem, f


def build_cache(data_dir: str, cache: str) -> None:
    data = Path(data_dir)
    out = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    games = list(_iter_games(data))
    print(f"[qnet:build] {len(games)} 局", flush=True)

    B, S, A, P = [], [], [], []
    L = {m: [] for m in LABEL_MODES}
    C, D = [], []
    counts, names = [], []
    op_stat = {}
    for stem, cf in games:
        npz = cf.with_name(stem + ".npz")
        ents = cf.with_name(stem + ".ents.jsonl")
        z = np.load(npz)
        board = z["board"]; stats = z["stats"]
        player = z["player"]; terminal = z["value_target"]
        el = [json.loads(l) for l in open(ents, encoding="utf-8") if l.strip()]
        cl = [json.loads(l) for l in open(cf, encoding="utf-8") if l.strip()]
        n = len(terminal)
        act = np.zeros((n, N_CLASS), dtype=np.float16)
        coins = np.zeros(n, dtype=np.int16)
        cd = np.zeros(n, dtype=np.int8)
        for i, (e, c) in enumerate(zip(el, cl)):
            pl = int(e["player"])
            coins[i] = int(e["coins"][pl])
            wcd = e.get("weapon_cd") or [[0] * 5, [0] * 5]
            cd[i] = int(wcd[pl][1]) if len(wcd[pl]) > 1 else 0
            ops = c.get("chosen") or []
            act[i] = encode_ops(ops).astype(np.float16)
            for op in ops:
                op_stat[int(op[0])] = op_stat.get(int(op[0]), 0) + 1
        B.append(board); S.append(stats); A.append(act); P.append(player)
        C.append(coins); D.append(cd)
        for m in LABEL_MODES:
            lab = terminal if m == "terminal" else compute_labels(
                np.asarray(stats, dtype=np.float32), np.asarray(player, dtype=np.int64), m)
            L[m].append(lab.astype(np.float32))
        counts.append(n); names.append(stem)

    N = int(sum(counts))
    Cb, H, W = B[0].shape[1:]
    Sd = S[0].shape[1]
    print(f"[qnet:build] 总决策点 {N}  board=({Cb},{H},{W}) stats={Sd}", flush=True)
    print(f"[qnet:build] chosen op_type 分布: {dict(sorted(op_stat.items()))}", flush=True)

    for nm, arr, dt, shape in (
        ("board", B, np.float16, (N, Cb, H, W)),
        ("stats", S, np.float16, (N, Sd)),
        ("action", A, np.float16, (N, N_CLASS)),
        ("player", P, np.int8, (N,)),
        ("coins", C, np.int16, (N,)),
        ("cd_lightning", D, np.int8, (N,)),
    ):
        mm = np.lib.format.open_memmap(out / f"{nm}.npy", mode="w+", dtype=dt, shape=shape)
        off = 0
        for blk in arr:
            mm[off:off + len(blk)] = blk
            off += len(blk)
        mm.flush(); del mm
        print(f"[qnet:build] 写完 {nm} ({shape})", flush=True)
    for m in LABEL_MODES:
        mm = np.lib.format.open_memmap(out / f"label_{m}.npy", mode="w+", dtype=np.float32, shape=(N,))
        off = 0
        for blk in L[m]:
            mm[off:off + len(blk)] = blk; off += len(blk)
        mm.flush(); del mm
        print(f"[qnet:build] 写完 label_{m}（std={float(np.asarray(L[m][0]).std()):.4f}...）", flush=True)
    np.savez(out / "meta.npz", counts=np.asarray(counts, dtype=np.int64),
             names=np.asarray(names))
    print(f"[qnet:build] 完成 → {out}/{{board,stats,action,player,coins,cd_lightning,label_*}}.npy + meta.npz",
          flush=True)


def load_cache(cache: str):
    out = Path(cache)
    return (
        np.load(out / "board.npy", mmap_mode="r"),
        np.load(out / "stats.npy", mmap_mode="r"),
        np.load(out / "action.npy", mmap_mode="r"),
        np.load(out / "player.npy", mmap_mode="r"),
        np.load(out / "coins.npy", mmap_mode="r"),
        np.load(out / "cd_lightning.npy", mmap_mode="r"),
        np.load(out / "meta.npz", allow_pickle=True),
    )


def load_label(cache: str, mode: str):
    return np.load(Path(cache) / f"label_{mode}.npy", mmap_mode="r")


def _game_offsets(counts: np.ndarray):
    return np.concatenate([[0], np.cumsum(counts)])


# ----------------------------------------------------------------------------- 训练

@torch.no_grad()
def evaluate(model, b, s, a, lab, idx, dev, batch_size):
    model.eval()
    se = np.zeros(len(idx), dtype=np.float64)
    pr = np.zeros(len(idx), dtype=np.float64)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev)
        ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev)
        aa = torch.from_numpy(np.asarray(a[j], dtype=np.float32)).to(dev)
        out = model(bb, ss, aa)
        pr[i:i + len(j)] = out.detach().cpu().numpy()
        se[i:i + len(j)] = np.asarray(lab[j], dtype=np.float64)
    err = (pr - se) ** 2
    r = {"n": len(idx), "mse": float(err.mean()), "var": float(se.var())}
    r["corr"] = float(np.corrcoef(pr, se)[0, 1]) if len(idx) > 2 else float("nan")
    return r


def train_main(args) -> None:
    dev = torch.device(args.device if args.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    b, s, a, player, coins, cd, meta = load_cache(args.cache)
    counts = meta["counts"]
    offs = _game_offsets(counts)
    N = int(counts.sum())
    lab = load_label(args.cache, args.label_mode)
    print(f"[qnet:train] device={dev}  label={args.label_mode}  cache={args.cache}  N={N}  "
          f"{len(counts)} 局  label std={np.asarray(lab).std():.4f}", flush=True)

    from my_ai.az_intent.az_selfplay import load_three_models
    _, _, value_model = load_three_models(args.ckpt)
    model = QActionNet(value_model).to(dev)
    n_tr = sum(p.numel() for p in model.trainable_params())
    print(f"[qnet:train] 冻结值网 encoder；可训参数 {n_tr:,}（action_mlp + q_head）", flush=True)

    rng = np.random.default_rng(args.seed)
    gi = rng.permutation(len(counts))
    val_g = set(int(x) for x in gi[:args.val_games])
    val_idx = np.concatenate([np.arange(offs[g], offs[g + 1]) for g in sorted(val_g)])
    tr_idx = np.concatenate([np.arange(offs[g], offs[g + 1])
                             for g in range(len(counts)) if g not in val_g])
    print(f"[qnet:train] train {len(tr_idx)} / val {len(val_idx)}（{args.val_games} 局）", flush=True)

    opt = torch.optim.Adam(model.trainable_params(), lr=args.lr)
    base = evaluate(model, b, s, a, lab, val_idx, dev, args.batch_size)
    print("[qnet:train] **训练前** val MSE=%.5f (常数基线 %.5f) r=%+.3f"
          % (base["mse"], base["var"], base["corr"]), flush=True)

    best = (base["mse"], -1)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.default_rng(args.seed + ep).permutation(len(tr_idx))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            j = tr_idx[order[i:i + args.batch_size]]
            bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev, non_blocking=True)
            ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev, non_blocking=True)
            aa = torch.from_numpy(np.asarray(a[j], dtype=np.float32)).to(dev, non_blocking=True)
            vv = torch.from_numpy(np.asarray(lab[j], dtype=np.float32)).to(dev, non_blocking=True)
            out = model(bb, ss, aa)
            loss = nn.functional.mse_loss(out, vv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        r = evaluate(model, b, s, a, lab, val_idx, dev, args.batch_size)
        flag = ""
        if r["mse"] < best[0]:
            best = (r["mse"], ep)
            if args.out:
                torch.save({"action_mlp": model.action_mlp.state_dict(),
                            "q_head": model.q_head.state_dict(),
                            "encoder_ckpt": args.ckpt, "state_dim": model.state_dim,
                            "label_mode": args.label_mode,
                            "epoch": ep, "train_mse": tot / max(nb, 1), "val_mse": r["mse"],
                            "val_corr": r["corr"]}, args.out)
            flag = "  <- 保存"
        print("[qnet:train] epoch %2d/%d  train MSE=%.5f | val MSE=%.5f (基线 %.5f) r=%+.3f  (%.0fs)%s"
              % (ep, args.epochs, tot / max(nb, 1), r["mse"], r["var"], r["corr"],
                 time.time() - t0, flag), flush=True)
    print(f"[qnet:train] 最佳 val MSE=%.5f @epoch %d  ->  {args.out}" % (best[0], best[1]), flush=True)


# ----------------------------------------------------------------------------- 探偏导

def _load_qnet(ckpt_path, model_path, dev):
    from my_ai.az_intent.az_selfplay import load_three_models
    _, _, value_model = load_three_models(ckpt_path)
    model = QActionNet(value_model).to(dev)
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    model.action_mlp.load_state_dict(sd["action_mlp"])
    model.q_head.load_state_dict(sd["q_head"])
    model.eval()
    return model


def _grad_batch(model, bb, ss, aa):
    """给定 board/stats/action（action requires_grad）→ 返回 Q 与 ∂Q/∂action。"""
    state_emb = model.state_embed(bb, ss)
    a = model.action_mlp(aa)
    x = torch.cat([state_emb, a], dim=1)
    q = model.q_head(x).squeeze(-1)
    g = torch.autograd.grad(q.sum(), aa)[0]
    return q.detach(), g


def _mixed_deriv_batch(model, bb, ss, aa):
    """混合二阶导：∂²Q/∂(动作数量)∂(闪电冷却 stats[CD_IDX])。

    让 stats 也 requires_grad，encoder 直接前向（不经 no_grad，其参数冻结但输入可微），
    先对 action 求一阶（create_graph），再对 stats 的 cd 分量求导。返回 (d2build, d2upg)。
    """
    state_emb = model.encoder(bb, ss)["state_emb"]          # 梯度可流回 ss
    a = model.action_mlp(aa)
    q = model.q_head(torch.cat([state_emb, a], dim=1)).squeeze(-1)
    g_a = torch.autograd.grad(q.sum(), aa, create_graph=True, retain_graph=True)[0]  # (B,24)
    d2_build = torch.autograd.grad(g_a[:, 0].sum(), ss, retain_graph=True)[0][:, CD_IDX]
    g_upg = g_a[:, list(UPGRADE_SEEN_CLASSES)].mean(dim=1)
    d2_upg = torch.autograd.grad(g_upg.sum(), ss)[0][:, CD_IDX]
    return d2_build.detach(), d2_upg.detach()


def probe_main(args) -> None:
    dev = torch.device(args.device if args.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    b, s, a, player, coins, cd, meta = load_cache(args.cache)
    counts = meta["counts"]
    N = int(counts.sum())
    model = _load_qnet(args.ckpt, args.model, dev)
    print(f"[qnet:probe] device={dev}  label={args.label_mode}  model={args.model}  N={N}", flush=True)

    coins_np = np.asarray(coins)
    cd_np = np.asarray(cd)
    chosen_vec = np.asarray(a)
    is_build = chosen_vec[:, 0] > 0
    is_upg = chosen_vec[:, 1:16].sum(axis=1) > 0
    is_downgrade = chosen_vec[:, DOWNGRADE] > 0
    is_lightning = chosen_vec[:, LIGHTNING] > 0

    ref = np.zeros((1, N_CLASS), dtype=np.float32); ref[0, HOLD] = 1.0

    _rng = np.random.default_rng(args.seed + 1000)

    def sample(idx, k):
        if len(idx) <= k:
            return idx
        return idx[_rng.choice(len(idx), k, replace=False)]

    def grads_for(idx):
        n = len(idx)
        G = np.zeros((n, N_CLASS), dtype=np.float32)
        Q = np.zeros(n, dtype=np.float32)
        for i in range(0, n, args.batch_size):
            j = idx[i:i + args.batch_size]
            bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev)
            ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev)
            aa = torch.from_numpy(np.tile(ref, (len(j), 1))).to(dev).requires_grad_(True)
            q, g = _grad_batch(model, bb, ss, aa)
            Q[i:i + len(j)] = q.detach().cpu().numpy()
            G[i:i + len(j)] = g.detach().cpu().numpy()
        return Q, G

    # 场景读数：Q@HOLD 均值（每种 chosen 场景的"基线价值"）
    print("[qnet:probe] 场景样本数 + Q@HOLD 均值:", flush=True)
    for tag, m in (("建塔(chosen含build)", is_build), ("升级(chosen含upgrade)", is_upg),
                   ("降级(chosen含downgrade)", is_downgrade),
                   ("闪电(chosen含lightning)", is_lightning)):
        idx = sample(np.nonzero(m)[0], args.per_scene)
        if len(idx) == 0:
            continue
        Q, G = grads_for(idx)
        print(f"    {tag}: 全量{int(m.sum())} 抽{len(idx)}  Q@HOLD mean={Q.mean():+.4f}", flush=True)

    # ── 核心探针：∂Q/∂(建塔/升级/降级/闪电) 随金币分桶 + 相对排序 ─────────────
    # 用户口径（2026-11-02）：正负不一定要"负"，只要相对排序对——闪电 > 建塔 > 升级。
    buckets = [("0-19", 0, 20), ("20-44", 20, 45), ("45-59", 45, 60),
               ("60-74", 60, 75), ("75-89", 75, 90), ("90-119", 90, 120), ("120+", 120, 10**9)]
    print("\n===== ∂Q/∂a 随金币分桶（HOLD 参考点；升级=见类 1/5/9/13 均值）=====", flush=True)
    print(f"{'金币桶':>8} {'n':>6} | {'建塔(0)':>9} {'升级(见)':>9} {'降级(16)':>9} {'闪电(17)':>9} | "
          f"{'闪>建塔':>6} {'闪>建>升':>7} {'闪排名#1':>7}", flush=True)
    res = {"buckets": {}, "scenario": {}}

    def ordering(G):
        """每局面：闪电/建塔/升级(见类均值)/降级/HOLD 的偏导相对关系。"""
        g_light = G[:, LIGHTNING]
        g_build = G[:, 0]
        g_upg = G[:, list(UPGRADE_SEEN_CLASSES)].mean(axis=1)
        g_dg = G[:, DOWNGRADE]
        g_hold = G[:, HOLD]
        # 名次：值越大名次越前（1=最高）
        four = np.stack([g_light, g_build, g_upg, g_hold], axis=1)   # (n,4)
        rank = (-four).argsort(axis=1).argsort(axis=1) + 1            # (n,4) 1..4
        light_first = (rank[:, 0] == 1)
        lbg = (g_light > g_build) & (g_build > g_upg)                 # 闪电>建塔>升级
        return {
            "light_build_frac": float((g_light > g_build).mean()),
            "light_build_upg_frac": float(lbg.mean()),
            "light_first_frac": float(light_first.mean()),
            "mean_rank_light": float(rank[:, 0].mean()),
            "mean_rank_build": float(rank[:, 1].mean()),
            "mean_rank_upg": float(rank[:, 2].mean()),
        }

    for name, lo, hi in buckets:
        m = (coins_np >= lo) & (coins_np < hi)
        idx = sample(np.nonzero(m)[0], args.per_scene * 3)
        if len(idx) == 0:
            print(f"{name:>8} {0:>6} |（无样本）", flush=True)
            continue
        Q, G = grads_for(idx)
        g_build = G[:, 0]
        g_upg_seen = G[:, list(UPGRADE_SEEN_CLASSES)].mean(axis=1)
        g_dg = G[:, DOWNGRADE]
        g_light = G[:, LIGHTNING]
        ord_ = ordering(G)
        res["buckets"][name] = {
            "n": len(idx), "build": float(g_build.mean()), "build_pos": float((g_build > 0).mean()),
            "upg_seen": float(g_upg_seen.mean()), "downgrade": float(g_dg.mean()),
            "lightning": float(g_light.mean()), "q_hold": float(Q.mean()), **ord_,
        }
        print(f"{name:>8} {len(idx):>6} | {g_build.mean():+9.5f} {g_upg_seen.mean():+9.5f} "
              f"{g_dg.mean():+9.5f} {g_light.mean():+9.5f} | "
              f"{ord_['light_build_frac']*100:5.0f}% {ord_['light_build_upg_frac']*100:6.0f}% "
              f"{ord_['light_first_frac']*100:6.0f}%", flush=True)

    # ── 冷却分层：∂Q/∂(建塔/升级) 随闪电冷却 cd 怎么变 ──────────────────────
    # 推测："闪电已好(cd=0) 时花钱建塔/升级更不值（挤占现在就放闪电的机会）"。
    print("\n===== ∂Q/∂a 随闪电冷却 cd 分桶（HOLD 参考点）=====", flush=True)
    cd_buckets = [("cd=0(已好)", 0, 0), ("cd1-15(刚放)", 1, 15), ("cd16-35(冷却中)", 16, 35)]
    res["cd_buckets"] = {}
    for name, clo, chi in cd_buckets:
        m = (cd_np >= clo) & (cd_np <= chi)
        idx = sample(np.nonzero(m)[0], args.per_scene * 3)
        if len(idx) == 0:
            print(f"  {name}: 无样本", flush=True)
            continue
        Q, G = grads_for(idx)
        g_build = G[:, 0]
        g_upg = G[:, list(UPGRADE_SEEN_CLASSES)].mean(axis=1)
        g_light = G[:, LIGHTNING]
        res["cd_buckets"][name] = {"n": len(idx), "build": float(g_build.mean()),
                                   "upg_seen": float(g_upg.mean()), "lightning": float(g_light.mean())}
        print(f"  {name:>14} n={len(idx):>6}  建塔(0)={g_build.mean():+.5f}  "
              f"升级(见)={g_upg.mean():+.5f}  闪电(17)={g_light.mean():+.5f}", flush=True)

    # ── 联合分层：金币≥90 子集里，cd=0 vs cd>0 ──────────────────────────────
    print("\n===== 金币≥90 子集：cd=0 vs cd>0（建塔/升级偏导）=====", flush=True)
    res["cd_joint"] = {}
    for name, m in (("ge90&cd=0", (coins_np >= 90) & (cd_np == 0)),
                    ("ge90&cd>0", (coins_np >= 90) & (cd_np > 0))):
        idx = sample(np.nonzero(m)[0], args.per_scene * 3)
        if len(idx) == 0:
            print(f"  {name}: 无样本", flush=True)
            continue
        Q, G = grads_for(idx)
        g_build = G[:, 0]
        g_upg = G[:, list(UPGRADE_SEEN_CLASSES)].mean(axis=1)
        res["cd_joint"][name] = {"n": len(idx), "build": float(g_build.mean()),
                                 "upg_seen": float(g_upg.mean())}
        print(f"  {name:>12} n={len(idx):>6}  建塔(0)={g_build.mean():+.5f}  "
              f"升级(见)={g_upg.mean():+.5f}", flush=True)

    # ── 混合二阶导 ∂²Q/∂(建塔|升级)∂(cd)（可选）──────────────────────────────
    # stats[CD_IDX]=闪电cd/cooldown。符号>0 ⇒ cd 越大（越冷却）花钱边际越高 = 闪电好时花钱更不值。
    print("\n===== 混合二阶导 ∂²Q/∂(动作)∂(cd)（全量抽样）=====", flush=True)
    mix_idx = np.random.default_rng(1).choice(N, min(N, 8000), replace=False)
    d2_builds = np.zeros(len(mix_idx), dtype=np.float32)
    d2_upgs = np.zeros(len(mix_idx), dtype=np.float32)
    for i in range(0, len(mix_idx), args.batch_size):
        j = mix_idx[i:i + args.batch_size]
        bb = torch.from_numpy(np.asarray(b[j], dtype=np.float32)).to(dev)
        ss = torch.from_numpy(np.asarray(s[j], dtype=np.float32)).to(dev).requires_grad_(True)
        aa = torch.from_numpy(np.tile(ref, (len(j), 1))).to(dev).requires_grad_(True)
        d2b, d2u = _mixed_deriv_batch(model, bb, ss, aa)
        d2_builds[i:i + len(j)] = d2b.cpu().numpy()
        d2_upgs[i:i + len(j)] = d2u.cpu().numpy()
    res["mixed"] = {"n": len(mix_idx), "d2build_mean": float(d2_builds.mean()),
                    "d2build_pos": float((d2_builds > 0).mean()),
                    "d2upg_mean": float(d2_upgs.mean()), "d2upg_pos": float((d2_upgs > 0).mean())}
    print(f"  ∂²Q/∂(建塔)∂(cd)  mean={d2_builds.mean():+.5f}  正占比 {(d2_builds>0).mean()*100:.0f}%", flush=True)
    print(f"  ∂²Q/∂(升级)∂(cd)  mean={d2_upgs.mean():+.5f}  正占比 {(d2_upgs>0).mean()*100:.0f}%", flush=True)

    # 全量抽 2 万做 §3 每类偏导表 + 全局排序
    all_idx = np.random.default_rng(0).choice(N, min(N, 20000), replace=False)
    Q, G = grads_for(all_idx)
    res["all"] = {"n": len(all_idx), "mean": G.mean(axis=0), "sign_frac": (G > 0).mean(axis=0),
                  **ordering(G)}

    np.savez(args.out, all_mean=res["all"]["mean"], all_sign_frac=res["all"]["sign_frac"],
             buckets=res["buckets"], cd_buckets=res["cd_buckets"], cd_joint=res["cd_joint"],
             mixed=res["mixed"], all_ordering={k: res["all"][k] for k in
             ("light_build_frac", "light_build_upg_frac", "light_first_frac",
              "mean_rank_light", "mean_rank_build", "mean_rank_upg")})
    print(f"[qnet:probe] 结果 → {args.out}", flush=True)

    mean = res["all"]["mean"]; sf = res["all"]["sign_frac"]
    print("\n===== §3 每类动作 ∂Q/∂a（全量抽样 n=%d）=====" % res["all"]["n"])
    for c in range(N_CLASS):
        nm = CLASS_NAME.get(c, f"塔型{c}")
        mark = "" if (c in (0, 16, 17, 18, 19, 20, 23) or c in UPGRADE_SEEN_CLASSES) else "（无训练信号=噪声）"
        print(f"  类{c:2d} {nm:16s}  ∂Q/∂a = {mean[c]:+.5f}  正占比 {sf[c]:.2f}{mark}")
    print("\n===== 全局相对排序（闪电/建塔/升级/HOLD，全量抽样）=====")
    print(f"  闪电>建塔: {res['all']['light_build_frac']*100:.1f}%   "
          f"闪电>建塔>升级: {res['all']['light_build_upg_frac']*100:.1f}%   "
          f"闪电排名#1: {res['all']['light_first_frac']*100:.1f}%")
    print(f"  平均名次：闪电 {res['all']['mean_rank_light']:.2f} / 建塔 {res['all']['mean_rank_build']:.2f} "
          f"/ 升级 {res['all']['mean_rank_upg']:.2f}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--data", default="training_history/collect_v2_s1")
    ap.add_argument("--cache", default="training_history/qnet_action/cache")
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--model", default="training_history/qnet_action/qnet_action.pt")
    ap.add_argument("--out", default="training_history/qnet_action/probe.npz")
    ap.add_argument("--label-mode", default="terminal", choices=list(LABEL_MODES))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-games", type=int, default=20)
    ap.add_argument("--per-scene", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.build:
        build_cache(args.data, args.cache)
    elif args.train:
        train_main(args)
    elif args.probe:
        probe_main(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
