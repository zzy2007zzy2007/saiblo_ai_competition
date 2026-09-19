# ppo_v9 训练进度检查 Prompt

将以下内容发送给 AI，一次性获取所有关键指标。使用时替换 `<run_id>`。

```
请查看 server10 上 ppo_v9 的训练进度和阶段性结果。

要求：
1. 输出 Markdown 表格，数字右对齐，不要输出原始日志
2. 每个表格后附一句简要趋势判断（↗改善 / ↘恶化 / →持平 / ⚠️关注）
3. 如果某项数据不完整，标注"（无数据）"而非跳过

---

## 一、进程状态

运行 `ps aux | grep train | grep -v grep`，确认进程存活、PID、启动时间。

## 二、逐代进化概览

从 `ppo_v9/outputs/<run_id>/training/ga_training_*.log` 中提取每代的 `Generation.*best_elo` 行，
输出表格：| 代 | best_elo | mean_elo | 种群 diversity | 种子 cos 均值 | 耗时 |

## 三、种子多样性详细

从日志中提取所有 `Seeds diversity` 行，
输出表格：| 代 | cos 范围 [min, max] | cos 均值 | 趋势判断 |

同时检查是否有 `Hard dup` 或 `Soft thr` 或 `dedup fallback` 日志行。
输出：| 代 | 去重触发次数 | 触发类型 |

## 四、基线验证对战（最新 3 代）

读取 `ppo_v9/outputs/<run_id>/generations/gen_XXXX/evaluation.json`，
对每代输出 Best Seed 对应的表格：
| Baseline Agent | 胜 | 负 | 平 | 场次 | 胜率 | 平均回合 |

同时输出跨代趋势表（只列 Best Seed 胜率和回合数）：
| Baseline Agent | Gen N-2 胜率 | Gen N-1 胜率 | Gen N 胜率 | Gen N 回合 | 趋势 |

## 五、Selection vs Baseline 耗时分解

从日志 Phase 切换时间戳计算每代的：
| 代 | Selection 耗时 | Baseline 耗时 | 总耗时 | Selection 占比 |

格式：mm:ss（如 63:15），占比百分比右对齐。

## 六、对手池状态

读取 `ppo_v9/outputs/<run_id>/league/pool.json`，
输出：池大小、最新加入的 N 个对手 ID、被淘汰记录。

## 七、异常与风险

- 检查 ga_training_error_*.log 是否非空
- 检查日志中是否有 ERROR / CRITICAL / Traceback
- 用 nvidia-smi 检查 GPU 显存是否正常释放
- 统计 best_elo 连续未突破的代数（如 >20 代预警收敛过早）

## 八、总结（3-5 句话）

聚焦：
- ELO 演进趋势（best / mean）
- 多样性是否恶化
- rule_zzy25 是否被突破
- 最大瓶颈（Selection 耗时 / 收敛过早 / 其他）
```
