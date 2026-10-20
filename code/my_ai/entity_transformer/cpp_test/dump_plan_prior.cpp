// dump_plan_prior.cpp —— 在我们自己的对局状态上，用**亚军自己的代码**算出动作先验（heuristic）。
//
// 目的（主线最后一环）：验证"我们的架构能不能表达亚军的先验"。
//   先验 = 亚军对每个候选计划的手写即时打分 heuristic（纯函数、无 rollout）：
//     heuristic = base_heuristic + lure_heuristic + lightning_heuristic
//                 - operation_penalty - followup_penalty
//   （亚军源码 `lure_strategy_v4_root_plans.hpp:223-224`）
//
// 输入：我们 raw dump 的 `ents_seed*.jsonl`（每行一条决策的完整状态）
//   {"round":..,"player":..,"towers":[[id,player,x,y,type,hp],..],
//    "ants":[[id,player,x,y,hp,max_hp,kind,age,behavior],..],
//    "effects":[[weapon_type,player,x,y,remaining_turns],..],
//    "weapon_cd":[[..5..],[..5..]], "weapon_usage":[u0,u1],
//    "coins":[c0,c1], "hp_us":..,"hp_opp":.., ...}
// 输出：JSONL，每行 {round, player, cand_idx, name, base_name, lure_name,
//                    lightning_name, heuristic, base_heuristic, lure_heuristic,
//                    lightning_heuristic, operation_penalty, followup_penalty}
//
// ── ⭐ 2026-10-20 重大发现：SDK 里已有现成工具，不必自己写 ──────────────
//   `其他版本ai/saiblo-30th-AI/Game1/antgame_cpp_sdk/examples/sdk_lure_inspector.cpp`
//   （119.7 KB，2746 行）是一个**完整的检查器**，已包含：
//     line 1262:  rs::DefenseSimulator defense_root = rs::make_defense_simulator(state, native, player);
//     line 1268:  const ls::RootPlanSet root_plans = ls::generate_root_plans(state, &defense_root, player);
//   以及 `int main()`（line 2711）：**从 stdin 读一个 JSON request**，按 `mode` 分派：
//     round_summary / alignment_check / rollout_alignment_batch /
//     rollout_entity_alignment / **plan_rollouts** / **plan_trace** / move_debug
//   ⇒ `plan_trace` / `plan_rollouts` 正是"从状态导出候选计划 + 分数"。
//
//   还差的两步：
//     (a) **编译**：没有 CMake 目标、也没有预编译 exe。要按 `docs/runnerup_build_note.md`
//         的方式编（MinGW + libantgame_cpp_sdk.a + ASCII 路径）；
//     (b) **搞清 request 格式**：读 `inspect_plan_trace_request` / `inspect_plan_rollouts_request`
//         两个函数，看它们从 request 里取哪些字段（是 replay 记录？还是可直接给 PublicState？）。
//   若它能直接吃 `PublicState`（或 replay），我们就能：
//     ① 把 5000 局的状态转成它的输入 → 得到亚军真实候选与 heuristic；
//     ② 用它自带的 alignment_check 与 seeds 732–900 的日志对拍，验证等价性。
//
// ── 下面保留我最初手写工具的骨架（如果 inspector 不可用，可回退到这条路）──────


//   antgame::sdk::PublicState           在 antgame_cpp_sdk/include/antgame_sdk/sdk.hpp:504
//     字段与我们的 raw dump 一一对应：round_index / coins / die_count / old_count /
//     super_weapon_usage / weapon_cooldowns / bases / towers / ants / active_effects
//     构造：explicit PublicState(uint64_t seed_in, movement_policy="enhanced", cold_handle_rule_illegal=false)
//     方法：clone() / to_public_round_state() / sync_public_round_state(..) / can_apply_operation(..)
//   antgame::sdk::NativeSimulator       亚军主程序用它做对局推进（ai_cpp_lure_v4.cpp:29/37）
//     NativeSimulator simulator(seed);  simulator.sync_public_round_state(public_state.to_public_round_state());
//   rs::DefenseSimulator                亚军自带的**单边**防御模拟器（用于 rollout 与候选生成）
//     亚军代码里形如 `rs::DefenseSimulator simulator = root.clone();`
//     （lure_strategy_v4_rollout_score.hpp:92）——root 的来源需要确认：
//       grep rs::DefenseSimulator 的构造点 / to_defense_simulator 之类的转换函数
//   候选生成入口：
//     antgame::ai::generate_root_plans(const PublicState&, const rs::DefenseSimulator*, int player)
//     → RootPlanSet { plans: vector<CombinedPlan{ name, base_name, lure_name,
//                        lightning_name, ops, heuristic, base_heuristic, lure_heuristic,
//                        lightning_heuristic, operation_penalty, followup_penalty, ... }> }
//
// ── 待完成（TODO）─────────────────────────────────────────────────────
//  [ ] 1. 找到 `rs::DefenseSimulator` 从 `PublicState` 构造的确切函数
//         （候选生成里 `generate_lure_candidates(state, simulator, player)` 需要它；
//          若暂时不好接，可先传 nullptr——`generate_root_plans` 内部对 simulator 可能允许为空，
//          需先确认，否则先只出 base/lightning 候选，lure 候选另算）
//  [ ] 2. 解析 `ents_*.jsonl` → 填充 PublicState（towers/ants/effects/coins/hp/weapon_cd/usage/round）
//  [ ] 3. 调 `generate_root_plans` → 遍历 `out.plans` → 写 JSONL
//  [ ] 4. **等价性校验**：拿 `match_results/ai0_runnerup_seed*.stderr.log`（seeds 732–900）
//         里的 `heuristic / base_heuristic / lure_heuristic / lightning_heuristic /
//         operation_penalty / followup_penalty` 逐条比对——但这些日志缺状态，
//         所以校验方式是：**重跑同 seed 并开 raw dump**（亚军确定性）后逐条对齐；
//         或者更省：在亚军自己的 exe 上开一个 `--dump-plan-prior` 开关，让它内部直接写。
//  [ ] 5. 跑 5000 局（`lowq_e2` 的 5000 个 ents.jsonl）→ 生成训练集
//
// ── 编译（参考亚军自己的构建方式）────────────────────────────────────
//   g++ -std=c++17 -O2 -I <antgame_cpp_sdk>/include -I <cpp_lure_v4>/include \
//       dump_plan_prior.cpp <antgame_cpp_sdk>/lib/libantgame_cpp_sdk.a -o dump_plan_prior.exe
//   ⚠️ 已知坑（见 docs/runnerup_build_note.md）：MinGW ld 处理不了中文路径 ⇒
//      库和 -o 输出都放 ASCII 目录（如 D:\dsh_build_tmp），TMP/TEMP 也必须是 ASCII。

#include <cstdio>
int main(int argc, char** argv) {
    std::fprintf(stderr,
        "[dump_plan_prior] 尚未实现 —— 见文件头 TODO 1..5。\n"
        "   下一步先确认 rs::DefenseSimulator 的构造来源，再填 PublicState。\n");
    return 1;
}
