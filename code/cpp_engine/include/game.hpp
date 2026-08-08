#ifndef __GAME_H__
#define __GAME_H__

#include "ant.h"
#include "item.h"
#include "building.h"
#include "comm_judger.h"
#include "map.h"
#include "operation.h"
#include "output.h"
#include "player.h"
#include <array>
#include <deque>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#define MAX_ROUND 512

class Game {
  public:
    enum class MovementPolicy {
        Legacy,
        Enhanced,
    };

    enum class OperationErrorKind {
        None,
        Rule,
        Protocol,
    };

  private:
    using RiskField = multi_dim_array_t<double, 2, MAP_SIZE, MAP_SIZE>;
    using ScalarField = multi_dim_array_t<double, MAP_SIZE, MAP_SIZE>;
    struct PathPlan {
        ScalarField total_cost{};
        ScalarField damage_cost{};
    };
    struct TowerPathPlan {
        int tower_id = -1;
        PathPlan plan{};
    };
    bool is_end;
    int winner;
    int round;
    const std::string mini_replay = "mini_replay.txt";
    int ant_id = 0;
    int barrack_id = 0;
    int tower_id = 0;
    std::string err_msg = "";

    // state of AI
    enum AI_state {
        OK,
        INITIAL_ERROR,
        RUN_ERROR,
        TIMEOUT_ERROR,
        OUTPUT_LIMIT,
        ILLEGAL_OPERATION,
        HUMAN_PLAYER
    } state[2];

    std::string record_file;
    from_judger_round judger_round_info;
    to_judger output_to_judger;
    unsigned long long random_seed;
    unsigned long long rng_state = 0;
    Map map;
    // player[2]
    Player player0, player1;
    Headquarter base_camp0, base_camp1;
    std::vector<Operation> op[2];
    std::vector<Item> item[2];
    // Map caches DefenseTower* per cell, so tower storage must keep addresses stable.
    std::deque<DefenseTower> defensive_towers;
    std::vector<Ant> ants;
    RiskField damage_risk_field{};
    RiskField control_risk_field{};
    RiskField effect_pull_field{};
    bool risk_fields_dirty = true;
    MovementPolicy movement_policy = MovementPolicy::Enhanced;
    bool cold_handle_rule_illegal = false;
    bool enhanced_move_phase_active = false;
    bool enhanced_move_cache_dirty = true;
    std::array<ScalarField, 2> enhanced_worker_costs{};
    std::array<ScalarField, 2> enhanced_combat_base_costs{};
    std::array<ScalarField, 2> enhanced_traffic_field{};
    std::array<ScalarField, 2> enhanced_reservations{};
    std::array<std::vector<TowerPathPlan>, 2> enhanced_tower_plans{};
    std::array<std::unordered_map<int, int>, 2> enhanced_tower_claims{};
    std::unordered_map<int, std::pair<int, int>> enhanced_move_cells{};
    std::unordered_map<int, int> enhanced_move_tower_targets{};

    Output output;

    void attack_ants();   // defensive towers attack ants
    void move_ants();     // get direction and move ants
    bool manage_ants();   // update game_data_output & manage ants by status
    void generate_ants(); // generate new ants
    void increase_ant_age();
    void update_items();      // update duration of item
    void update_coin();      // update coin by basic income and penalty
    void update_pheromone(); // update pheromone for each ant
    bool judge_base_camp();  // judge winner by base_camps' hp
    void judge_winner(); // judge winner when round is no less than 512
    unsigned long long next_random();
    double random_float();
    int random_index(int bound);
    std::vector<std::tuple<int, int, int>> legal_move_candidates(const Ant &ant) const;
    int choose_random_legal_move(const Ant &ant);
    int choose_ant_move(const Ant &ant);
    int choose_ant_move_legacy(const Ant &ant);
    int choose_ant_move_enhanced(const Ant &ant);
    bool ant_can_walk_to(int x, int y) const;
    bool ant_can_target_cell(const Ant &ant, int x, int y) const;
    double crowding_penalty(const Ant &ant, int x, int y) const;
    double move_progress_score(const Ant &ant, int x, int y, const Pos &target) const;
    double move_pheromone_score(const Ant &ant, int x, int y) const;
    double expected_damage_cost(const Ant &ant, int x, int y) const;
    double control_risk_cost(const Ant &ant, int x, int y) const;
    void mark_risk_fields_dirty();
    void refresh_static_risk_fields();
    std::vector<double> directional_field_scores(
        const Ant &ant, const std::vector<std::tuple<int, int, int>> &candidates,
        const RiskField &field) const;
    double tower_pull_score(const Ant &ant, int x, int y,
                            const DefenseTower *tower_target) const;
    Pos move_target_for_ant(const Ant &ant) const;
    void invalidate_enhanced_move_cache();
    void begin_move_phase();
    void end_move_phase();
    double cell_damage_hp(int player, int x, int y) const;
    void compute_enhanced_traffic_field();
    PathPlan reverse_weighted_plan(
        int player, const std::vector<std::pair<int, int>> &sources,
        double damage_weight, double control_weight, double traffic_weight,
        double effect_weight) const;
    void prepare_enhanced_move_cache(bool reset_reservations);
    void ensure_enhanced_move_cache();
    double tower_attack_value(const Ant &ant, const DefenseTower &tower, double arrival_hp) const;
    const TowerPathPlan *tower_plan_for(int player, int tower_id) const;
    void record_enhanced_reservation(const Ant &ant, int move);
    void teleport_ants();
    void resolve_random_move_steps(Ant &ant, int steps);
    void drift_items();
    std::pair<int, int> random_bewitch_target(const Ant &ant);
    int half_plane_delta(int player, int x, int y) const;
    bool ant_in_own_half(const Ant &ant) const;
    DefenseTower *enemy_tower_at(int player, int x, int y);
    const DefenseTower *enemy_tower_at(int player, int x, int y) const;
    void grant_emergency_evasion(Ant &ant, int stacks,
                                 bool grant_control_free_on_deplete = true);
    void attack_tower_from_ant(Ant &ant, DefenseTower &tower);
    void resolve_ant_step(Ant &ant, int move);
    void spawn_ant_from_tower(const DefenseTower &tower, Ant::Kind kind);
    void process_producer_towers();
    void apply_control(Ant &ant, Ant::Behavior behavior,
                       const std::pair<int, int> *target = nullptr);
    void maybe_control_free(Ant &ant, bool was_active, bool is_active);
    void prepare_ants_for_attack();
    void apply_lightning_storm(Item &it, int player);
    void damage_ant_by_tower(DefenseTower &tower, Ant &ant);
    int tower_count_for_player(int player) const;

  public:
    Game();
    ~Game();

    std::string get_record_file();
    void set_AI_state_IO(int player);
    void init();
    bool is_ended();
    bool next_round();

    // bool is_operation_valid(const OperationSet& op) const;
    bool apply_operation(const std::vector<Operation> &op_list,
                         int player, std::string &err_msg,
                         OperationErrorKind *error_kind = nullptr); // 进行选手在回合的操作
    void dump_round_state(/* const std::string& filename */);
    void
    dump_last_round(/* const std::string& filename */ const std::string &msg);
    void request_end_state();
    void receive_end_state();
    void send_end_info();
    void dump_result(const std::string &filename);
    // void dump_mini_replay(const std::string &filename);
    // void show(int t){map.show(t);}
    //  read message from judger
    template <typename T> void read_from_judger(T &des);
    void listen(int player);
    bool round_read_from_judger(int player);

    // ── M1 embeddable-engine interface (added in the code/cpp_engine copy) ──
    // Defined in m1_support.cpp.  No judger protocol, no replay I/O.
    void init_game(unsigned long long seed, const std::string &movement_policy,
                   bool cold_handle_illegal);
    int get_round() const { return round; }
    int get_winner() const { return winner; }
    int get_base_hp(int player) const {
        return player == 0 ? base_camp0.get_hp() : base_camp1.get_hp();
    }
    // tower snapshot: (id, x, y, player, type, hp, hp_limit) of alive towers
    std::vector<std::array<int, 7>> tower_snapshot() const;
    // ant snapshot: (id, x, y, player, hp, kind) of alive ants
    std::vector<std::array<int, 6>> ant_snapshot() const;
    bool valid_cell(int x, int y) const { return map.is_valid(x, y); }
    int coin(int player) const { return player == 0 ? player0.coin.get_coin()
                                                    : player1.coin.get_coin(); }
    unsigned long long rng_state_now() const { return rng_state; }
    std::array<int, 2> base_levels(int player) const {
        return {player == 0 ? base_camp0.get_cd_level() : base_camp1.get_cd_level(),
                player == 0 ? base_camp0.get_ant_level() : base_camp1.get_ant_level()};
    }
    // weapon cooldowns for a player: [lightning, emp, deflector, evasion]
    std::array<int, 4> weapon_cds(int player) const;
    // tower details: (id, x, y, player, type, hp, hp_limit, level, range, damage, cd)
    std::vector<std::array<int, 11>> tower_details() const;
    // ant details: (id, x, y, player, hp, kind, age, level, status, max_hp, frozen, behavior)
    std::vector<std::array<int, 12>> ant_details() const;
    // pheromone field as a flat 2*MAP_SIZE*MAP_SIZE float array
    std::vector<double> pheromone_flat();
    // active weapon effects: (weapon_type, player, x, y, remaining_turns) for items with duration > 0
    std::vector<std::array<int, 5>> active_effects() const;
    int super_weapon_usage(int player) const {
        return player == 0 ? player0.super_weapons_usage : player1.super_weapons_usage;
    }
    static int tower_build_cost(int tower_count);
    // Apply ops with the official cold_handle_rule_illegal semantics: per-op,
    // illegal ones skipped, but used_tower / camp-upgraded flags persist across
    // the whole list (matches Game::round_read_from_judger's cold path).
    std::vector<Operation> apply_operation_list_cold(
        int player, const std::vector<Operation> &op_list);
    // Can ``op`` be accepted after ``pending``?  Uses the REAL cold path on a
    // deep clone (no state mutation), so it is exactly as faithful as the
    // engine — but in ONE C++ call instead of Python-side clone+apply.
    bool can_apply_cold(int player, const Operation &op,
                        const std::vector<Operation> &pending);
    // Clone-free legality check (replicates the cold path's checks with
    // pending's gold/used_tower/camp simulated) — cheap at any state size.
    bool can_apply_dry(int player, const Operation &op,
                       const std::vector<Operation> &pending);
    Game deep_clone() const;
};

#endif
