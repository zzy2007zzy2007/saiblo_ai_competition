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
