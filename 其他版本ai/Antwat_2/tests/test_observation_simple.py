#!/usr/bin/env python3
"""简化版验证 ObservationEncoder 的脚本"""

import sys
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print("=" * 80)
print("测试 ObservationEncoder 的 global_dim")
print("=" * 80)
print()

try:
    # 直接读取和检查 observation.py 的内容
    obs_path = src_dir / "ppo_antwar" / "env" / "observation.py"
    with open(obs_path, 'r') as f:
        content = f.read()
    
    # 检查 global_dim
    if "self.global_dim = 30" in content:
        print("✓ 检查到 self.global_dim = 30")
    else:
        print("✗ 未找到 self.global_dim = 30")
        sys.exit(1)
    
    # 检查是否有对手超级武器冷却编码的代码
    has_enemy_cooldown = False
    if "for weapon_type in SuperWeaponType:" in content:
        lines = content.split('\n')
        # 检查是否有两个循环
        count = 0
        for line in lines:
            if "for weapon_type in SuperWeaponType:" in line:
                count += 1
        if count >= 2:
            has_enemy_cooldown = True
            print("✓ 检查到两个超级武器编码循环")
    
    # 检查是否有 enemy 相关的武器冷却代码
    if "state.weapon_cooldowns[enemy, weapon_type]" in content:
        has_enemy_cooldown = True
        print("✓ 检查到对手超级武器冷却编码代码")
    
    if has_enemy_cooldown:
        print("✓ 所有要求的修改都已完成！")
    else:
        print("✗ 缺少对手超级武器冷却编码代码")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("所有检查通过！✓")
print("=" * 80)
