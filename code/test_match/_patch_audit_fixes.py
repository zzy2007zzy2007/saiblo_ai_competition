"""批量修复异常审计里的 7 项：把静默失败改成「可观测」的失败。

原则：**不改语义**（返回值/分支保持原样），只加 计数 + 首次打印。
这样历史行为不变，但以后再出现"某类动作/机制恒定为零"时能立刻看到原因。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
patched: list[str] = []


def patch(rel: str, old: str, new: str, tag: str) -> None:
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    if old not in t:
        print(f"  ✗ {tag}: 未找到锚点")
        return
    if new in t:
        print(f"  = {tag}: 已修过")
        return
    p.write_text(t.replace(old, new, 1), encoding="utf-8")
    patched.append(tag)
    print(f"  ✓ {tag}")


# ── A) mc_light 分支的宽 except ────────────────────────────────────────────
patch(
    "code/my_ai/az_intent/bundle_mcts.py",
    """                    except Exception:  # noqa: BLE001
                        pinned.append(_top)""",
    """                    except Exception as _exc:  # noqa: BLE001
                        # 2026-10-22（审计）：mc_light 抛异常 ⇒ 静默等于基线，原来**零观测**。
                        self.mc_light_fallback += 1
                        if self.mc_light_fallback == 1:
                            self.mc_last_error = self.mc_last_error or f"mc_light {type(_exc).__name__}: {_exc}"
                            print(f"[mc_light] ⚠️ 异常回退（首次）：{type(_exc).__name__}: {_exc}", flush=True)
                        pinned.append(_top)""",
    "A mc_light 异常可观测",
)

patch(
    "code/my_ai/az_intent/bundle_mcts.py",
    """        self.mc_last_error = None""",
    """        self.mc_last_error = None
        self.mc_light_fallback = 0      # 2026-10-22：mc_light 分支的异常回退计数（原来不可见）
        self.mc_upg_skipped = 0         # 2026-10-22：显式升级候选构造失败次数（原来只有 print）""",
    "A mc_light 计数器初始化",
)

# ── B) q_classes：静默跳过 → 可观测 ───────────────────────────────────────
patch(
    "code/my_ai/az_intent/az_selfplay.py",
    """                except Exception:  # noqa: BLE001  主干没暴露 state_emb 时静默跳过
                    pass""",
    """                except Exception as _exc:  # noqa: BLE001
                    # 2026-10-22（审计）：这里静默 ⇒ pos_pin="q" 会悄悄退化成 argmax，零观测。
                    if not out.get("_q_warned"):
                        out["_q_warned"] = True
                        print(f"[q_classes] ⚠️ 类条件 Q 值不可用（首次）：{type(_exc).__name__}: {_exc}",
                              flush=True)""",
    "B q_classes 可观测",
)

# ── D) AZAI_BAN_CLASSES 解析失败 → 打印 ──────────────────────────────────
patch(
    "code/my_ai/decoder.py",
    """            for _c in _ban.split(","):
                try:
                    _i = int(_c)
                except ValueError:
                    continue
                if 0 <= _i < mask.shape[0]:
                    mask[_i] = False""",
    """            _applied = 0
            for _c in _ban.split(","):
                try:
                    _i = int(_c)
                except ValueError:
                    # 2026-10-22（审计）：原来静默 continue ⇒ 写错格式("17 "、全角逗号)时
                    # **该类根本没被禁**，而训练曲线看起来一切正常。
                    print(f"[ban] ⚠️ 忽略非法类号 {_c!r}（AZAI_BAN_CLASSES={_ban!r}）", flush=True)
                    continue
                if 0 <= _i < mask.shape[0]:
                    mask[_i] = False
                    _applied += 1""",
    "D BAN_CLASSES 解析可观测",
)

# ── E) tail_fn 失败 → 可观测 ────────────────────────────────────────────
patch(
    "code/my_ai/az_intent/mc_quiet.py",
    """            try:
                _t = float(tail_fn(st, player))
            except Exception:  # noqa: BLE001
                _t = None""",
    """            try:
                _t = float(tail_fn(st, player))
            except Exception as _exc:  # noqa: BLE001
                # 2026-10-22（审计）：原来静默 ⇒ 补尾整段不生效、评分口径悄悄退回纯 HP 差。
                global _TAIL_WARNED
                if not _TAIL_WARNED:
                    _TAIL_WARNED = True
                    print(f"[tail] ⚠️ tail_fn 失败（首次）：{type(_exc).__name__}: {_exc}", flush=True)
                _t = None""",
    "E tail_fn 可观测",
)

patch(
    "code/my_ai/az_intent/mc_quiet.py",
    """DEFAULT_MAX_ROUNDS""",
    """_TAIL_WARNED = False   # 2026-10-22：tail_fn 失败只提示一次
DEFAULT_MAX_ROUNDS""",
    "E tail 警告标志",
)

# ── F) _op_label：失败可观测（保持返回值不变，避免破坏 dump schema）────────
patch(
    "code/test_match/az_bridge_ai.py",
    """    except Exception:  # noqa: BLE001
        pass
    return (23, -1, -1)""",
    """    except Exception as _exc:  # noqa: BLE001
        # 2026-10-22（审计）：这里静默返回 (23,-1,-1)=HOLD ⇒ 升级/降级 op 会被标成"空过"，
        # **污染训练标签**。实测正常 op 不受影响（升级→(1,4,2)、降级→(16,4,2)），
        # 但一旦发生必须看得见。
        global _OP_LABEL_WARNED
        if not _OP_LABEL_WARNED:
            _OP_LABEL_WARNED = True
            print(f"[label] ⚠️ _op_label 失败（首次）：{type(_exc).__name__}: {_exc}", flush=True)
    return (23, -1, -1)""",
    "F _op_label 可观测",
)

patch(
    "code/test_match/az_bridge_ai.py",
    """STATS = {"rounds": 0,""",
    """_OP_LABEL_WARNED = False   # 2026-10-22：op 标签解析失败只提示一次
STATS = {"rounds": 0,""",
    "F label 警告标志",
)

# ── G) res.visit 拼写错误（实际字段是 visit_policy）────────────────────
patch(
    "code/test_match/az_bridge_ai.py",
    """            vis = getattr(res, "visit", None)""",
    """            # 2026-10-22（审计）：BundleSearchResult 的真实字段是 **visit_policy**，
            # 原来写 "visit" ⇒ 恒为 None ⇒ dump 里"选中类"永远取候选 0（而这份 dump 是值网训练数据）
            vis = getattr(res, "visit_policy", None) or getattr(res, "visit", None)""",
    "G visit_policy 字段名",
)

print(f"\n完成 {len(patched)} 处：{', '.join(patched)}")
