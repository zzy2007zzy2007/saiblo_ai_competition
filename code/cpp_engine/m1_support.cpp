// M1 embeddable-engine support: judger-free init, deep clone, state snapshots.
// Defined as Game member functions (can touch privates); compiled only in the
// code/cpp_engine copy, NOT in the official game/.
#include "game.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
// Mirrors the constants in game.cpp (file-local there; deterministic LCG).
constexpr unsigned long long RNG_MASK = (1ULL << 48) - 1;
constexpr unsigned long long RNG_MULTIPLIER = 25214903917ULL;
constexpr int INIT_CAMP_HP = 50;
}  // namespace

void Game::init_game(unsigned long long seed, const std::string &policy_str,
                     bool cold_handle_illegal) {
    // Reset every state container (safe on re-init).
    is_end = false;
    winner = -1;
    round = 0;
    ant_id = 0;
    barrack_id = 0;
    tower_id = 0;
    err_msg = "";
    state[0] = AI_state::OK;
    state[1] = AI_state::OK;
    op[0].clear();
    op[1].clear();
    item[0].clear();
    item[1].clear();
    defensive_towers.clear();
    ants.clear();
    risk_fields_dirty = true;
    enhanced_move_phase_active = false;
    enhanced_move_cache_dirty = true;
    enhanced_worker_costs = {};
    enhanced_combat_base_costs = {};
    enhanced_traffic_field = {};
    enhanced_reservations = {};
    enhanced_tower_plans = {};
    enhanced_tower_claims = {};
    enhanced_move_cells.clear();
    enhanced_move_tower_targets.clear();
    map = Map();

    player0.ant_target_x = PLAYER_1_BASE_CAMP_X;
    player0.ant_target_y = PLAYER_1_BASE_CAMP_Y;
    player1.ant_target_x = PLAYER_0_BASE_CAMP_X;
    player1.ant_target_y = PLAYER_0_BASE_CAMP_Y;

    movement_policy =
        (policy_str == "legacy") ? MovementPolicy::Legacy : MovementPolicy::Enhanced;
    cold_handle_rule_illegal = cold_handle_illegal;
    random_seed = seed;
    rng_state = (random_seed ^ RNG_MULTIPLIER) & RNG_MASK;
    map.init_pheromon(random_seed);

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < ItemType::Count; j++)
            item[i].push_back(Item(0, 0, 0, 0));

    base_camp0 = Headquarter(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y, 0, 0, 0, INIT_CAMP_HP);
    base_camp1 = Headquarter(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y, 1, 0, 0, INIT_CAMP_HP);
    map.map[PLAYER_0_BASE_CAMP_X][PLAYER_0_BASE_CAMP_Y].base_camp = &base_camp0;
    map.map[PLAYER_1_BASE_CAMP_X][PLAYER_1_BASE_CAMP_Y].base_camp = &base_camp1;
}

std::vector<std::array<int, 7>> Game::tower_snapshot() const {
    std::vector<std::array<int, 7>> out;
    out.reserve(defensive_towers.size());
    for (const auto &t : defensive_towers) {
        if (t.destroy())
            continue;
        out.push_back({t.get_id(), t.get_x(), t.get_y(), t.get_player(),
                       static_cast<int>(t.get_type()), t.get_hp(), t.get_hp_limit()});
    }
    return out;
}

std::vector<std::array<int, 6>> Game::ant_snapshot() const {
    std::vector<std::array<int, 6>> out;
    out.reserve(ants.size());
    for (const auto &a : ants) {
        if (a.get_hp() <= 0)
            continue;  // dead ants are not part of the live state
        out.push_back({a.get_id(), a.get_x(), a.get_y(), a.get_player(),
                       a.get_hp(), static_cast<int>(a.get_kind())});
    }
    return out;
}

std::array<int, 4> Game::weapon_cds(int player) const {
    return {item[player][ItemType::LightingStorm].cd,
            item[player][ItemType::EMPBlaster].cd,
            item[player][ItemType::Deflectors].cd,
            item[player][ItemType::EmergencyEvasion].cd};
}

std::vector<std::array<int, 12>> Game::ant_details() const {
    std::vector<std::array<int, 12>> out;
    out.reserve(ants.size());
    for (const auto &a : ants) {
        if (a.get_hp() <= 0)
            continue;
        out.push_back({a.get_id(), a.get_x(), a.get_y(), a.get_player(),
                       a.get_hp(), static_cast<int>(a.get_kind()),
                       a.get_age(), a.get_level(),
                       static_cast<int>(a.get_status()),
                       a.get_hp_limit(), a.is_frozen ? 1 : 0,
                       static_cast<int>(a.behavior)});
    }
    return out;
}

std::vector<std::array<int, 11>> Game::tower_details() const {
    std::vector<std::array<int, 11>> out;
    out.reserve(defensive_towers.size());
    for (const auto &t : defensive_towers) {
        if (t.destroy())
            continue;
        out.push_back({t.get_id(), t.get_x(), t.get_y(), t.get_player(),
                       static_cast<int>(t.get_type()), t.get_hp(), t.get_hp_limit(),
                       t.get_level(), t.get_range(), t.get_damage(), t.get_cd()});
    }
    return out;
}

std::vector<double> Game::pheromone_flat() {
    auto ph = map.get_pheromone();  // [2][MAP_SIZE][MAP_SIZE]
    std::vector<double> out;
    out.reserve(2 * MAP_SIZE * MAP_SIZE);
    for (int p = 0; p < 2; ++p)
        for (int x = 0; x < MAP_SIZE; ++x)
            for (int y = 0; y < MAP_SIZE; ++y)
                out.push_back(ph[p][x][y]);
    return out;
}

std::vector<std::array<int, 5>> Game::active_effects() const {
    std::vector<std::array<int, 5>> out;
    for (int p = 0; p < 2; ++p)
        for (int t = 0; t < ItemType::Count; ++t) {
            const Item &it = item[p][t];
            if (it.duration > 0)
                out.push_back({t, p, it.x, it.y, it.duration});
        }
    return out;
}

int Game::tower_build_cost(int tower_count) {
    // mirrors coin.cpp's tower_build_cost_for_count (INITIAL_TOWER_BUILD_PRICE = 15)
    tower_count = std::max(tower_count, 0);
    int cost = 15;
    for (int index = 0; index < tower_count / 2; ++index)
        cost *= 3;
    if (tower_count % 2 == 1)
        cost *= 2;
    return cost;
}

namespace {
bool is_base_upgrade_op(Operation::Type type) {
    return type == Operation::Type::BarrackUpgrade ||
           type == Operation::Type::AntUpgrade;
}
}  // namespace

std::vector<Operation> Game::apply_operation_list_cold(
    int player, const std::vector<Operation> &op_list) {
    // Replicates the official cold_handle_rule_illegal path in
    // round_read_from_judger: per-op apply, illegal skipped, but used_tower
    // and camp_upgraded_flag persist across the whole list.
    std::vector<Operation> accepted_ops;
    std::vector<int> used_tower;
    bool camp_upgraded_flag = false;
    for (const auto &operation : op_list) {
        const auto type = operation.get_operation_type();
        if ((type == Operation::Type::TowerUpgrade ||
             type == Operation::Type::TowerDestroy) &&
            std::find(used_tower.begin(), used_tower.end(), operation.get_id()) !=
                used_tower.end()) {
            continue;
        }
        if (is_base_upgrade_op(type) && camp_upgraded_flag) {
            continue;
        }
        const int pending_tower_id = tower_id;
        std::string operation_error;
        OperationErrorKind error_kind = OperationErrorKind::None;
        if (!apply_operation(std::vector<Operation>{operation}, player,
                             operation_error, &error_kind)) {
            continue;  // protocol errors can't arise outside the judger
        }
        accepted_ops.push_back(operation);
        if (type == Operation::Type::TowerBuild)
            used_tower.push_back(pending_tower_id);
        else if (type == Operation::Type::TowerUpgrade ||
                 type == Operation::Type::TowerDestroy)
            used_tower.push_back(operation.get_id());
        if (is_base_upgrade_op(type))
            camp_upgraded_flag = true;
    }
    return accepted_ops;
}

namespace {
// mirror of coin.cpp's file-local tower_build_cost_for_count (INITIAL=15)
int tower_build_cost_for_count(int tower_count) {
    tower_count = std::max(tower_count, 0);
    int cost = 15;
    for (int index = 0; index < tower_count / 2; ++index)
        cost *= 3;
    if (tower_count % 2 == 1)
        cost *= 2;
    return cost;
}
constexpr int TOWER_MAX_LEVEL = 2;  // mirror of game.cpp's define
bool op_eq(const Operation &a, const Operation &b) {
    return a.get_operation_type() == b.get_operation_type() &&
           a.get_id() == b.get_id() && a.get_args() == b.get_args() &&
           a.get_pos_x() == b.get_pos_x() && a.get_pos_y() == b.get_pos_y();
}
int tower_upgrade_cost(int level) { return level == 0 ? 60 : 200; }
int item_cost(ItemType it) {
    const int cost[4] = {90, 135, 60, 60};
    return cost[it];
}
int base_upgrade_cost(int level) {
    const int cost[2] = {200, 250};
    return cost[level];
}
int tower_destroy_income(const DefenseTower &t, int tower_count) {
    switch (t.get_level()) {
    case 0:
        return static_cast<int>(
            (9LL * tower_build_cost_for_count(tower_count - 1) *
             std::max(t.get_hp(), 0)) /
            (10LL * std::max(t.get_hp_limit(), 1)));
    case 1:
        return static_cast<int>((9LL * 60 * std::max(t.get_hp(), 0)) /
                                (10LL * std::max(t.get_hp_limit(), 1)));
    case 2:
        return static_cast<int>((9LL * 200 * std::max(t.get_hp(), 0)) /
                                (10LL * std::max(t.get_hp_limit(), 1)));
    }
    return 0;
}
}  // namespace

bool Game::can_apply_dry(int player, const Operation &op,
                         const std::vector<Operation> &pending) {
    // Clone-free cold-path simulation: iterate pending + op, check each against
    // the simulated state (gold / tower count / used_tower / camp / cooldowns),
    // apply only the accepted ones.  Returns whether the final op was accepted.
    int gold = player == 0 ? player0.coin.get_coin() : player1.coin.get_coin();
    int tower_count = tower_count_for_player(player);
    int next_tower_id = tower_id;
    std::vector<int> used_tower;
    bool camp_upgraded = false;
    std::array<int, 4> cds;
    for (int t = 0; t < 4; ++t)
        cds[t] = item[player][t].cd;

    auto emp_shielded = [&](int px, int py) {
        const Item &it = item[1 - player][ItemType::EMPBlaster];
        return it.duration && distance(Pos(px, py), Pos(it.x, it.y)) <= 3;
    };

    auto check_apply = [&](const Operation &p) -> bool {
        const auto pt = p.get_operation_type();
        int x = p.get_pos_x(), y = p.get_pos_y();
        switch (pt) {
        case Operation::Type::TowerBuild:
            if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE)
                return false;
            if (map.map[x][y].base_camp != nullptr)
                return false;
            if (map.map[x][y].tower != nullptr)
                return false;
            if (map.map[x][y].player != player)
                return false;
            if (emp_shielded(x, y))
                return false;
            if (gold < tower_build_cost_for_count(tower_count))
                return false;
            used_tower.push_back(next_tower_id++);
            gold -= tower_build_cost_for_count(tower_count);
            tower_count++;
            return true;
        case Operation::Type::TowerUpgrade: {
            int id = p.get_id();
            if (id < 0 || id >= (int)defensive_towers.size() ||
                defensive_towers[id].destroy() ||
                defensive_towers[id].get_player() != player)
                return false;
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
                return false;
            const DefenseTower &t = defensive_towers[id];
            if (emp_shielded(t.get_x(), t.get_y()))
                return false;
            if (t.get_level() == TOWER_MAX_LEVEL)
                return false;
            if (!t.upgrade_type_check(TowerType(p.get_args())))
                return false;
            if (gold < tower_upgrade_cost(t.get_level()))
                return false;
            used_tower.push_back(id);
            gold -= tower_upgrade_cost(t.get_level());
            return true;
        }
        case Operation::Type::TowerDestroy: {
            int id = p.get_id();
            if (id < 0 || id >= (int)defensive_towers.size() ||
                defensive_towers[id].destroy() ||
                defensive_towers[id].get_player() != player)
                return false;
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
                return false;
            const DefenseTower &t = defensive_towers[id];
            if (emp_shielded(t.get_x(), t.get_y()))
                return false;
            used_tower.push_back(id);
            gold += tower_destroy_income(t, tower_count);
            if (t.get_type() == TowerType::Basic)
                tower_count--;
            return true;
        }
        case Operation::Type::LightingStorm:
        case Operation::Type::EMPBlaster:
        case Operation::Type::Deflectors:
        case Operation::Type::EmergencyEvasion: {
            if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE)
                return false;
            ItemType it = pt == Operation::Type::LightingStorm
                              ? ItemType::LightingStorm
                              : pt == Operation::Type::EMPBlaster
                                    ? ItemType::EMPBlaster
                                    : pt == Operation::Type::Deflectors
                                          ? ItemType::Deflectors
                                          : ItemType::EmergencyEvasion;
            if (cds[it] > 0)
                return false;
            if (gold < item_cost(it))
                return false;
            gold -= item_cost(it);
            cds[it] = get_item_cd(it);
            return true;
        }
        case Operation::Type::BarrackUpgrade:
        case Operation::Type::AntUpgrade: {
            if (camp_upgraded)
                return false;
            int level = player == 0 ? base_camp0.get_cd_level()
                                    : base_camp1.get_cd_level();
            if (level == 2)
                return false;
            if (gold < base_upgrade_cost(level))
                return false;
            gold -= base_upgrade_cost(level);
            camp_upgraded = true;
            return true;
        }
        default:
            return false;
        }
    };

    std::vector<Operation> all = pending;
    all.push_back(op);
    bool last_accepted = false;
    for (size_t i = 0; i < all.size(); ++i) {
        bool acc = check_apply(all[i]);
        if (i + 1 == all.size())
            last_accepted = acc;
    }
    return last_accepted;
}

bool Game::can_apply_cold(int player, const Operation &op,
                          const std::vector<Operation> &pending) {
    // pending + op must be applied in ONE cold-list call so used_tower /
    // camp_upgraded persist across the whole round sequence.
    Game copy = deep_clone();
    std::vector<Operation> all = pending;
    all.push_back(op);
    std::vector<Operation> accepted = copy.apply_operation_list_cold(player, all);
    // op was accepted iff it appears in `accepted` more times than in `pending`
    // (a duplicate op in pending that got skipped is not counted twice).
    int in_pending = 0, in_accepted = 0;
    for (const auto &p : pending)
        if (op_eq(p, op)) in_pending++;
    for (const auto &a : accepted)
        if (op_eq(a, op)) in_accepted++;
    return in_accepted > in_pending;
}

Game Game::deep_clone() const {
    Game copy = *this;  // memberwise copy: containers deep-copied, Map pointers stale

    // Re-point the base camps to the copy's own Headquarter members.
    copy.map.map[PLAYER_0_BASE_CAMP_X][PLAYER_0_BASE_CAMP_Y].base_camp = &copy.base_camp0;
    copy.map.map[PLAYER_1_BASE_CAMP_X][PLAYER_1_BASE_CAMP_Y].base_camp = &copy.base_camp1;

    // Re-point every Map cell's DefenseTower* to the copy's towers (by id).
    std::unordered_map<int, DefenseTower *> id_to_ptr;
    id_to_ptr.reserve(copy.defensive_towers.size());
    for (auto &t : copy.defensive_towers)
        id_to_ptr[t.get_id()] = &t;
    for (int i = 0; i < MAP_SIZE; i++)
        for (int j = 0; j < MAP_SIZE; j++) {
            if (copy.map.map[i][j].tower == nullptr)
                continue;
            auto it = id_to_ptr.find(copy.map.map[i][j].tower->get_id());
            copy.map.map[i][j].tower =
                (it != id_to_ptr.end()) ? it->second : nullptr;
        }
    return copy;
}
