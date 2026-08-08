// M1: thin pybind11 binding of the embeddable C++ engine for MCTS.
// Exposes a Python-friendly interface (op tuples like the Python engine),
// mirrors the official protocol-token -> Operation field mapping.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "game.hpp"

#include <array>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
// Python (op_type, arg0, arg1) -> C++ Operation, matching comm_judger.cpp's
// protocol-token parsing:
//   11 build / 21 lightning / 22 emp / 23 deflector / 24 evasion : pos = (a0, a1)
//   12 upgrade: id = a0, args = a1 ;  13 downgrade: id = a0 ;  else: bare type
Operation make_operation(int type, int a0, int a1) {
    switch (type) {
    case 11:
    case 21:
    case 22:
    case 23:
    case 24:
        return Operation(type, -1, -1, a0, a1);
    case 12:
        return Operation(12, a0, a1, -1, -1);
    case 13:
        return Operation(13, a0, -1, -1, -1);
    default:
        return Operation(type, -1, -1, -1, -1);
    }
}
}  // namespace

PYBIND11_MODULE(native_game, m) {
    m.doc() = "Embeddable C++ AntWar engine for MCTS (code/cpp_engine copy)";

    py::class_<Game>(m, "NativeGame")
        .def(py::init<>())
        .def("init_game", &Game::init_game,
             py::arg("seed"), py::arg("movement_policy") = "enhanced",
             py::arg("cold") = true)
        .def("apply_operation_list",
             [](Game &g, int player, const std::vector<std::array<int, 3>> &ops) {
                 // Replicates the official cold_handle_rule_illegal path
                 // (round_read_from_judger): per-op apply with illegal ops
                 // skipped, while used_tower / camp-upgraded flags persist
                 // across the whole list.
                 std::vector<Operation> list;
                 list.reserve(ops.size());
                 for (const auto &o : ops)
                     list.push_back(make_operation(o[0], o[1], o[2]));
                 auto accepted = g.apply_operation_list_cold(player, list);
                 return std::make_tuple(true, accepted.size());
             },
             py::arg("player"), py::arg("ops"))
        .def("advance_round", &Game::next_round)
        .def("round", &Game::get_round)
        .def("is_ended", &Game::is_ended)
        .def("winner", &Game::get_winner)
        .def("base_hp", &Game::get_base_hp)
        .def("tower_snapshot", &Game::tower_snapshot)
        .def("ant_snapshot", &Game::ant_snapshot)
        .def("valid_cell", &Game::valid_cell)
        .def("coin", &Game::coin)
        .def("rng_state_now", &Game::rng_state_now)
        .def("weapon_cds", &Game::weapon_cds)
        .def("base_levels", &Game::base_levels)
        .def("tower_details", &Game::tower_details)
        .def("ant_details", &Game::ant_details)
        .def("pheromone_flat", &Game::pheromone_flat)
        .def("active_effects", &Game::active_effects)
        .def("super_weapon_usage", &Game::super_weapon_usage)
        .def_static("tower_build_cost", &Game::tower_build_cost)
        .def("can_apply_cold",
             [](Game &g, int player, const std::array<int, 3> &op,
                const std::vector<std::array<int, 3>> &pending) {
                 std::vector<Operation> pend;
                 pend.reserve(pending.size());
                 for (const auto &o : pending)
                     pend.push_back(make_operation(o[0], o[1], o[2]));
                 return g.can_apply_cold(player,
                                         make_operation(op[0], op[1], op[2]),
                                         pend);
             },
             py::arg("player"), py::arg("op"), py::arg("pending"))
        .def("can_apply_dry",
             [](Game &g, int player, const std::array<int, 3> &op,
                const std::vector<std::array<int, 3>> &pending) {
                 std::vector<Operation> pend;
                 pend.reserve(pending.size());
                 for (const auto &o : pending)
                     pend.push_back(make_operation(o[0], o[1], o[2]));
                 return g.can_apply_dry(player,
                                        make_operation(op[0], op[1], op[2]),
                                        pend);
             },
             py::arg("player"), py::arg("op"), py::arg("pending"))
        .def("clone", &Game::deep_clone);
}
