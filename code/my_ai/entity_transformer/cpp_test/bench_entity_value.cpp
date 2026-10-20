// 速度基准：同一局面重复 N 次前向，报每次耗时
#include <chrono>
#include <cstdio>
#include <vector>

#include "antgame_ai/entity_value_net.hpp"

using antgame::sdk::lure_strategy_detail::EntityValueNet;

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <weights.bin> [iters]\n", argv[0]); return 2; }
    const int iters = argc >= 3 ? std::atoi(argv[2]) : 10000;
    EntityValueNet net;
    if (argc >= 4) net.d = std::atoi(argv[3]);
    if (argc >= 5) net.layers_n = std::atoi(argv[4]);
    if (argc >= 6) net.ffn = std::atoi(argv[5]);
    if (argc >= 7) net.heads = std::atoi(argv[6]);
    net.dh = net.d / net.heads;
    if (!net.load(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    net.value_mean = 10481.754f;
    net.value_std = 1750.318f;

    // 一个典型中后期局面：2 座塔 + 10 只蚁
    std::vector<int> tx = {4, 5}, ty = {9, 15}, tt = {0, 0};
    std::vector<float> thr = {1.0f, 0.6f};
    std::vector<int> ax = {3, 7, 10, 12, 14, 6, 9, 11, 13, 8};
    std::vector<int> ay = {5, 8, 11, 4, 13, 7, 15, 9, 2, 12};
    std::vector<float> ahr = {1.0f, 0.5f, 0.8f, 1.0f, 0.3f, 0.9f, 1.0f, 0.6f, 0.7f, 1.0f};
    std::vector<int> ak = {0, 0, 1, 0, 1, 0, 0, 1, 0, 0};
    std::vector<int> al = {0, 1, 0, 2, 0, 1, 0, 0, 1, 0};
    std::vector<int> aa = {10, 20, 15, 30, 5, 40, 25, 8, 12, 18};
    std::vector<int> ab = {0, 0, 1, 0, 2, 0, 0, 0, 1, 0};

    float sink = 0.f;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        sink += net.forward(50, 100, tx, ty, tt, thr, ax, ay, ahr, ak, al, aa, ab);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    std::printf("iters=%d  per_forward=%.1f us  (sink=%.1f)\n", iters, us, sink);
    return 0;
}
