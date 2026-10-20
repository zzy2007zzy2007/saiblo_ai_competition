// dump_plan_prior.cpp —— 把「我们的状态 JSON」直接灌进亚军的 generate_root_plans，
// 导出每个候选计划的 heuristic（= 亚军的**先验**，纯函数，不含 rollout）。
//
// 为什么不用 replay：`load_replay_round_start` 只能从 seed 逐回合推演，而我们手上
// 只有中局状态的快照、没有 op 历史（详见 docs/plan_prior_distill.md 第 2 节）。
//
// 协议（行式，UTF-8）：
//   stdin  每行一个状态：见 docs/plan_prior_distill.md 第 3 节
//   stdout 每行一个结果：{"ok":1,...,"plans":[...]}
//
// 编译见 code/my_ai/distill_top2/cpp_test/build_dump_plan_prior.ps1

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include "antgame_ai/lure_strategy_v4.hpp"
#include "antgame_sdk/native_sim.hpp"
#include "antgame_sdk/sdk.hpp"

using json = nlohmann::json;

namespace ls = ::antgame::sdk::lure_strategy_detail;
namespace rs = ::antgame::sdk::random_search_detail;

using ::antgame::sdk::Ant;
using ::antgame::sdk::AntBehavior;
using ::antgame::sdk::AntKind;
using ::antgame::sdk::AntStatus;
using ::antgame::sdk::NativeSimulator;
using ::antgame::sdk::Operation;
using ::antgame::sdk::PublicRoundState;
using ::antgame::sdk::PublicState;
using ::antgame::sdk::SuperWeaponType;
using ::antgame::sdk::Tower;
using ::antgame::sdk::TowerType;
using ::antgame::sdk::WeaponEffect;

namespace {

int row_int(const json &row, std::size_t index, int fallback = 0) {
    if (!row.is_array() || index >= row.size() || row.at(index).is_null()) {
        return fallback;
    }
    return row.at(index).get<int>();
}

std::vector<int> int_list(const json &value) {
    std::vector<int> out;
    if (!value.is_array()) {
        return out;
    }
    out.reserve(value.size());
    for (const auto &item : value) {
        out.push_back(item.is_null() ? 0 : item.get<int>());
    }
    return out;
}

// 我们的 dump schema：towers = [id, player, x, y, type, hp, cd]
// 同时兼容 inspector 的对象 schema（{id,player,pos:{x,y},type,cd,hp}），方便互换调试。
Tower parse_tower(const json &row) {
    Tower tower;
    if (row.is_object()) {
        tower.tower_id = row.value("id", -1);
        tower.player = row.value("player", -1);
        tower.x = row.value("pos", json::object()).value("x", row.value("x", -1));
        tower.y = row.value("pos", json::object()).value("y", row.value("y", -1));
        tower.tower_type = static_cast<TowerType>(row.value("type", 0));
        tower.cooldown = row.value("cd", row.value("cooldown", 0));
        tower.hp = row.value("hp", 0);
        return tower;
    }
    tower.tower_id = row_int(row, 0, -1);
    tower.player = row_int(row, 1, -1);
    tower.x = row_int(row, 2, -1);
    tower.y = row_int(row, 3, -1);
    tower.tower_type = static_cast<TowerType>(row_int(row, 4, 0));
    tower.hp = row_int(row, 5, 0);
    tower.cooldown = row_int(row, 6, 0);
    return tower;
}

// 我们的 dump schema：ants = [id, player, x, y, hp, max_hp, kind, age, level, behavior, status, frozen]
// ⚠️ 注意 kind 在 age/level 之前，与 sdk.hpp 的 Ant 字段顺序不同，必须按下标取。
Ant parse_ant(const json &row) {
    Ant ant;
    if (row.is_object()) {
        ant.ant_id = row.value("id", -1);
        ant.player = row.value("player", -1);
        ant.x = row.value("pos", json::object()).value("x", row.value("x", -1));
        ant.y = row.value("pos", json::object()).value("y", row.value("y", -1));
        ant.hp = row.value("hp", 0);
        ant.level = row.value("level", 0);
        ant.age = row.value("age", 0);
        ant.status = static_cast<AntStatus>(row.value("status", 0));
        ant.behavior = static_cast<AntBehavior>(row.value("behavior", 0));
        ant.kind = static_cast<AntKind>(row.value("kind", 0));
        ant.last_move = row.value("move", row.value("last_move", -1));
        return ant;
    }
    ant.ant_id = row_int(row, 0, -1);
    ant.player = row_int(row, 1, -1);
    ant.x = row_int(row, 2, -1);
    ant.y = row_int(row, 3, -1);
    ant.hp = row_int(row, 4, 0);
    // index 5 = max_hp（可由 kind/level 推出，不用）
    ant.kind = static_cast<AntKind>(row_int(row, 6, 0));
    ant.age = row_int(row, 7, 0);
    ant.level = row_int(row, 8, 0);
    ant.behavior = static_cast<AntBehavior>(row_int(row, 9, 0));
    ant.status = static_cast<AntStatus>(row_int(row, 10, 0));
    // index 11 = frozen（status 里已经编码，忽略）
    return ant;
}

WeaponEffect parse_effect(const json &row) {
    WeaponEffect effect;
    if (row.is_object()) {
        effect.weapon_type = static_cast<SuperWeaponType>(row.value("type", row.value("weapon_type", 1)));
        effect.player = row.value("player", -1);
        effect.x = row.value("x", -1);
        effect.y = row.value("y", -1);
        effect.remaining_turns = row.value("duration", row.value("remaining_turns", 0));
        return effect;
    }
    effect.weapon_type = static_cast<SuperWeaponType>(row_int(row, 0, 1));
    effect.player = row_int(row, 1, -1);
    effect.x = row_int(row, 2, -1);
    effect.y = row_int(row, 3, -1);
    effect.remaining_turns = row_int(row, 4, 0);
    return effect;
}

json ops_to_json(const std::vector<Operation> &operations) {
    json out = json::array();
    for (const auto &operation : operations) {
        json row;
        row["type"] = static_cast<int>(operation.op_type);
        row["arg0"] = operation.arg0;
        row["arg1"] = operation.arg1;
        out.push_back(std::move(row));
    }
    return out;
}

bool build_state(const json &req, PublicRoundState &round_state, PublicState &state, int &player,
                 std::uint64_t &seed, std::string &error) {
    seed = req.value("seed", 0ULL);
    player = req.value("player", 0);
    round_state.round_index = req.value("round_index", req.value("round", 0));
    (void)error;

    const auto coins = int_list(req.value("coins", json::array()));
    if (coins.size() >= 2) {
        round_state.coins = {coins[0], coins[1]};
    }
    const auto camps = int_list(req.value("camps", json::array()));
    if (camps.size() >= 2) {
        round_state.camps_hp = {camps[0], camps[1]};
    }
    const auto speed_lv = int_list(req.value("speedLv", json::array()));
    if (speed_lv.size() >= 2) {
        round_state.speed_lv = {speed_lv[0], speed_lv[1]};
    }
    const auto anthp_lv = int_list(req.value("anthpLv", json::array()));
    if (anthp_lv.size() >= 2) {
        round_state.anthp_lv = {anthp_lv[0], anthp_lv[1]};
    }

    // bases = [[hp, generation_level, ant_level], ...]（我们的 dump）
    if (req.contains("bases") && req.at("bases").is_array()) {
        int index = 0;
        for (const auto &row : req.at("bases")) {
            if (index >= 2) {
                break;
            }
            if (row.is_object()) {
                round_state.camps_hp[index] = row.value("hp", round_state.camps_hp[index]);
                round_state.speed_lv[index] = row.value("generation_level", 0);
                round_state.anthp_lv[index] = row.value("ant_level", 0);
            } else if (row.is_array()) {
                round_state.camps_hp[index] = row_int(row, 0, round_state.camps_hp[index]);
                round_state.speed_lv[index] = row_int(row, 1, 0);
                round_state.anthp_lv[index] = row_int(row, 2, 0);
            }
            ++index;
        }
    }

    if (req.contains("weapon_cooldowns") && req.at("weapon_cooldowns").is_array()) {
        int index = 0;
        for (const auto &row : req.at("weapon_cooldowns")) {
            if (index >= 2) {
                break;
            }
            const auto values = int_list(row);
            if (values.size() == 4) {
                round_state.weapon_cooldowns[index] = {0, values[0], values[1], values[2], values[3]};
            } else if (values.size() >= 5) {
                round_state.weapon_cooldowns[index] = {values[0], values[1], values[2], values[3], values[4]};
            }
            ++index;
        }
    }

    if (req.contains("towers") && req.at("towers").is_array()) {
        for (const auto &row : req.at("towers")) {
            round_state.towers.push_back(parse_tower(row));
        }
    }
    if (req.contains("ants") && req.at("ants").is_array()) {
        for (const auto &row : req.at("ants")) {
            round_state.ants.push_back(parse_ant(row));
        }
    }
    const char *effect_keys[] = {"active_effects", "effects", "activeEffects"};
    for (const char *key : effect_keys) {
        if (req.contains(key) && req.at(key).is_array()) {
            for (const auto &row : req.at(key)) {
                round_state.active_effects.push_back(parse_effect(row));
            }
            break;
        }
    }

    // ⚠️ to_public_round_state() 返回的是 PublicRoundState（sdk.hpp:1000），
    //    不是 PublicState；PublicState 要靠 sync_public_round_state() 灌（sdk.hpp:1016）。
    try {
        state = PublicState(seed);
        state.sync_public_round_state(round_state);
    } catch (const std::exception &exc) {
        error = std::string("sync failed: ") + exc.what();
        return false;
    }

    state.seed = seed;
    state.round_index = round_state.round_index;

    // 这几个 PublicState 独有字段不走 round_state，直接覆盖。
    const auto usage = int_list(req.value("super_weapon_usage", json::array()));
    if (usage.size() >= 2) {
        state.super_weapon_usage = {usage[0], usage[1]};
    }
    const auto die_count = int_list(req.value("die_count", json::array()));
    if (die_count.size() >= 2) {
        state.die_count = {die_count[0], die_count[1]};
    }
    const auto old_count = int_list(req.value("old_count", json::array()));
    if (old_count.size() >= 2) {
        state.old_count = {old_count[0], old_count[1]};
    }
    return true;
}

json handle_one(const json &req) {
    json out;
    PublicRoundState round_state;
    PublicState state;
    int player = 0;
    std::uint64_t seed = 0;
    std::string error;
    if (!build_state(req, round_state, state, player, seed, error)) {
        out["ok"] = 0;
        out["error"] = error;
        return out;
    }
    if (player < 0 || player > 1) {
        out["ok"] = 0;
        out["error"] = "player must be 0 or 1";
        return out;
    }

    // 与 sdk_lure_inspector.cpp:1262-1269 完全一致的取 plan 方式（那一处已经验证过
    // 能复现亚军日志里的候选集），只是不做 rollout。
    NativeSimulator native(seed);
    native.sync_public_round_state(round_state);
    rs::DefenseSimulator defense_root = rs::make_defense_simulator(state, &native, player);
    defense_root.ignore_enemy_spawns = true;
    const ls::RootPlanSet root_plans = ls::generate_root_plans(state, &defense_root, player);

    out["ok"] = 1;
    out["round_index"] = round_state.round_index;
    out["player"] = player;
    out["seed"] = seed;
    out["n_base"] = root_plans.base_count;
    out["n_lure"] = root_plans.lure_count;
    out["n_lightning"] = root_plans.lightning_count;
    out["raw_plan_count"] = root_plans.raw_plan_count;
    out["n_plans"] = static_cast<int>(root_plans.plans.size());
    json plans = json::array();
    for (const auto &plan : root_plans.plans) {
        json row;
        row["key"] = plan.key;
        row["name"] = plan.name;
        row["base_name"] = plan.base_name;
        row["lure_name"] = plan.lure_name;
        row["lightning_name"] = plan.lightning_name;
        row["heuristic"] = plan.heuristic;
        row["base_heuristic"] = plan.base_heuristic;
        row["lure_heuristic"] = plan.lure_heuristic;
        row["lightning_heuristic"] = plan.lightning_heuristic;
        row["operation_penalty"] = plan.operation_penalty;
        row["followup_penalty"] = plan.followup_penalty;
        row["lightning_static_bonus"] = plan.lightning_static_bonus;
        row["has_lightning"] = plan.has_lightning;
        row["horizon"] = plan.horizon;
        row["followup"] = ls::followup_text(plan.followup);
        row["ops"] = ops_to_json(plan.ops);
        plans.push_back(std::move(row));
    }
    out["plans"] = std::move(plans);
    return out;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    std::size_t handled = 0;
    std::size_t failed = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json result;
        try {
            result = handle_one(json::parse(line));
        } catch (const std::exception &exc) {
            result = json{{"ok", 0}, {"error", std::string("exception: ") + exc.what()}};
        }
        if (result.value("ok", 0) != 1) {
            ++failed;
        }
        ++handled;
        std::cout << result.dump() << '\n';
    }
    std::cout.flush();
    std::cerr << "[dump_plan_prior] handled=" << handled << " failed=" << failed << '\n';
    return 0;
}
