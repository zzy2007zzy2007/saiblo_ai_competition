// 对拍：C++ 手工前向 vs PyTorch（样本由 export_test_samples.py 导出）
#include <cstdio>
#include <cstdint>
#include <vector>

#include "antgame_ai/entity_value_net.hpp"

using antgame::sdk::lure_strategy_detail::EntityValueNet;

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <weights.bin> <samples.bin>\n", argv[0]);
        return 2;
    }
    EntityValueNet net;
    if (!net.load(argv[1])) {
        std::fprintf(stderr, "load weights failed\n");
        return 1;
    }
    // 归一化常数（来自 ckpt 的 value_mean/value_std）
    net.value_mean = 10481.754f;
    net.value_std = 1750.318f;
    FILE *f = std::fopen(argv[2], "rb");
    if (!f) {
        std::fprintf(stderr, "open samples failed\n");
        return 1;
    }
    int n = 0;
    if (std::fread(&n, 4, 1, f) != 1) return 1;
    double sum_abs = 0.0, sum_sq = 0.0;
    for (int s = 0; s < n; ++s) {
        int ntw = 0;
        std::fread(&ntw, 4, 1, f);
        std::vector<int> tx(ntw), ty(ntw), tt(ntw);
        std::vector<float> thr(ntw);
        for (int i = 0; i < ntw; ++i) std::fread(&tx[i], 4, 1, f), std::fread(&ty[i], 4, 1, f),
                                     std::fread(&tt[i], 4, 1, f), std::fread(&thr[i], 4, 1, f);
        int nan = 0;
        std::fread(&nan, 4, 1, f);
        std::vector<int> ax(nan), ay(nan), ak(nan), al(nan), aa(nan), ab(nan);
        std::vector<float> ahr(nan);
        for (int i = 0; i < nan; ++i) std::fread(&ax[i], 4, 1, f), std::fread(&ay[i], 4, 1, f),
                                     std::fread(&ahr[i], 4, 1, f), std::fread(&ak[i], 4, 1, f),
                                     std::fread(&al[i], 4, 1, f), std::fread(&aa[i], 4, 1, f),
                                     std::fread(&ab[i], 4, 1, f);
        int base_hp = 0, coins = 0;
        float expected = 0.f, pred = 0.f;
        std::fread(&base_hp, 4, 1, f);
        std::fread(&coins, 4, 1, f);
        std::fread(&expected, 4, 1, f);
        std::fread(&pred, 4, 1, f);
        const float got = net.forward(base_hp, coins, tx, ty, tt, thr, ax, ay, ahr, ak, al, aa, ab);
        const double diff = got - pred;      // 和 PyTorch 预测比（验证 C++ 前向一致性）
        const double diff_t = got - expected; // 和亚军目标比（看模型误差）
        sum_abs += std::fabs(diff);
        sum_sq += diff * diff;
        if (s < 8) std::printf("  [%2d] pytorch=%.1f cpp=%.1f (diff=%.2f) | target=%.1f\n",
                               s, pred, got, diff, expected);
    }
    std::printf("[check] n=%d  MAE=%.2f  RMSE=%.2f\n", n, sum_abs / n, std::sqrt(sum_sq / n));
    std::fclose(f);
    return 0;
}

// 单独的速度基准：把第一个样本的前向重复 N 次，报每次耗时
extern "C" int run_bench(const char *weights, int iters);
