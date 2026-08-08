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
                 // Per-op calls replicate the Python engine's cold handling:
                 // an illegal op is skipped (returns false, no state change)
                 // and the remaining ops still apply.  The C++ apply_operation
                 // takes a whole list and aborts on the first failure, so we
                 // must call it once per operation.
                 int failed = 0;
                 std::string first_err;
                 for (const auto &o : ops) {
                     std::vector<Operation> one{make_operation(o[0], o[1], o[2])};
                     std::string err;
                     if (!g.apply_operation(one, player, err)) {
                         failed++;
                         if (first_err.empty())
                             first_err = err;
                     }
                 }
                 return std::make_tuple(failed == 0, first_err);
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
        .def("clone", &Game::deep_clone);
}
