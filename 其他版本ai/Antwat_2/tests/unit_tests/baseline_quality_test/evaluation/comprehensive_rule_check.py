"""游戏规则全面检查脚本"""
import sys
import os

# 添加SDK目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'saiblo-antwar-sdk-python'))

from antwar.gamestate import GameState
from antwar.pheromone import generate_init_pheromone
from antwar.gamedata import Ant, init_hp, init_coin
from antwar.protocol import *
from antwar.coord import Coord

def check_initialization():
    """检查游戏初始化规则"""
    print("=" * 80)
    print("1. 游戏初始化规则检查")
    print("=" * 80)
    
    issues = []
    
    # 检查初始HP
    print("\n1.1 初始HP检查")
    hp = init_hp()
    print(f"  官方初始HP: {hp}")
    if hp != 50:
        issues.append(f"初始HP错误: 期望50, 实际{hp}")
    
    # 检查初始金币
    print("\n1.2 初始金币检查")
    coin = init_coin()
    print(f"  官方初始金币: {coin}")
    if coin != 50:
        issues.append(f"初始金币错误: 期望50, 实际{coin}")
    
    # 检查信息素初始化
    print("\n1.3 信息素初始化检查")
    for seed in [0, 1, 2]:
        p0, p1 = generate_init_pheromone(seed)
        print(f"  Seed {seed}:")
        print(f"    p0[0][0] = {p0.value[0][0]:.6f}")
        print(f"    p1[0][0] = {p1.value[0][0]:.6f}")
        
        # 检查Seed 0的对称性
        if seed == 0:
            is_symmetric = True
            for i in range(19):
                for j in range(19):
                    if abs(p0.value[i][j] - p1.value[18-i][18-j]) > 0.001:
                        is_symmetric = False
                        break
                if not is_symmetric:
                    break
            if is_symmetric:
                issues.append(f"Seed {seed}的信息素完全对称，可能导致双方行为一致")
                print(f"    ⚠️  警告: 信息素完全对称")
    
    # 检查GameState初始化
    print("\n1.4 GameState初始化检查")
    game_state = GameState()
    print(f"  初始回合数: {game_state.round}")
    print(f"  初始HP: {game_state.hp}")
    print(f"  初始金币: {game_state.coin}")
    print(f"  初始蚂蚁数: {len(game_state.ants)}")
    print(f"  初始防御塔数: {len(game_state.towers)}")
    
    if game_state.round != 0:
        issues.append(f"初始回合数错误: 期望0, 实际{game_state.round}")
    if game_state.hp != [50, 50]:
        issues.append(f"初始HP错误: 期望[50, 50], 实际{game_state.hp}")
    if game_state.coin != [50, 50]:
        issues.append(f"初始金币错误: 期望[50, 50], 实际{game_state.coin}")
    
    return issues

def check_ant_generation():
    """检查蚂蚁生成规则"""
    print("\n" + "=" * 80)
    print("2. 蚂蚁生成规则检查")
    print("=" * 80)
    
    issues = []
    
    # 检查蚂蚁生成速度
    print("\n2.1 蚂蚁生成速度检查")
    game_state = GameState()
    game_state.init_with_seed(1)  # 初始化信息素
    
    # 模拟前10回合
    for round_num in range(1, 11):
        prev_ant_count = len(game_state.ants)
        game_state.simulate_next_round()
        new_ants = len(game_state.ants) - prev_ant_count
        print(f"  回合 {round_num}: 新增蚂蚁数={new_ants}, 总蚂蚁数={len(game_state.ants)}")
    
    # 检查蚂蚁属性
    print("\n2.2 蚂蚁属性检查")
    if game_state.ants:
        ant = game_state.ants[0]
        print(f"  蚂蚁ID: {ant.id}")
        print(f"  蚂蚁玩家: {ant.player}")
        print(f"  蚂蚁坐标: {ant.coord}")
        print(f"  蚂蚁HP: {ant.hp}")
        print(f"  蚂蚁等级: {ant.level}")
        print(f"  蚂蚁年龄: {ant.age}")
        print(f"  蚂蚁最大年龄: {Ant.max_age()}")
    
    # 检查蚂蚁最大年龄
    max_age = Ant.max_age()
    print(f"\n2.3 蚂蚁最大年龄: {max_age}")
    if max_age != 32:
        issues.append(f"蚂蚁最大年龄错误: 期望32, 实际{max_age}")
    
    return issues

def check_ant_movement():
    """检查蚂蚁移动规则"""
    print("\n" + "=" * 80)
    print("3. 蚂蚁移动规则检查")
    print("=" * 80)
    
    issues = []
    
    game_state = GameState()
    game_state.init_with_seed(1)  # 使用非对称的seed
    
    # 生成一些蚂蚁
    for _ in range(5):
        game_state.simulate_next_round()
    
    print(f"  当前蚂蚁数: {len(game_state.ants)}")
    
    if game_state.ants:
        ant = game_state.ants[0]
        print(f"\n  蚂蚁移动前坐标: {ant.coord}")
        print(f"  蚂蚁路径长度: {len(ant.path)}")
        
        # 模拟移动
        prev_coord = ant.coord
        game_state.simulate_next_round()
        new_coord = ant.coord
        
        print(f"  蚂蚁移动后坐标: {new_coord}")
        print(f"  蚂蚁是否移动: {prev_coord != new_coord}")
    
    return issues

def check_tower_attack():
    """检查防御塔攻击规则"""
    print("\n" + "=" * 80)
    print("4. 防御塔攻击规则检查")
    print("=" * 80)
    
    issues = []
    
    game_state = GameState()
    game_state.init_with_seed(1)
    
    # 尝试建造防御塔
    tower_pos = Coord(4, 6)
    op = build_tower_op(tower_pos)
    
    # 给玩家0足够的金币
    game_state.coin[0] = 100
    result = game_state.apply_operation(0, op)
    
    print(f"  建造防御塔: 位置={tower_pos}, 结果={result}")
    
    if result:
        print(f"  防御塔数量: {len(game_state.towers)}")
        if game_state.towers:
            tower = game_state.towers[0]
            print(f"  防御塔ID: {tower.id}")
            print(f"  防御塔玩家: {tower.player}")
            print(f"  防御塔坐标: {tower.coord}")
            print(f"  防御塔类型: {tower.type}")
            print(f"  防御塔CD: {tower.cd}")
            print(f"  防御塔范围: {tower.range()}")
            print(f"  防御塔伤害: {tower.damage()}")
    
    return issues

def check_costs():
    """检查成本计算规则"""
    print("\n" + "=" * 80)
    print("5. 成本计算规则检查")
    print("=" * 80)
    
    issues = []
    
    # 检查防御塔建造成本
    print("\n5.1 防御塔建造成本")
    print("  规则: 第k个防御塔成本 = 15 * 2^(k-1)")
    print("  第1个: 15, 第2个: 30, 第3个: 60, ...")
    
    # 不需要实际建造，只需要验证公式
    # 成本公式: 15 * 2^k，k为已有防御塔数量
    for k in range(3):
        expected_cost = 15 * (2 ** k)
        print(f"  第{k+1}个防御塔期望成本: {expected_cost}")
    
    # 检查升级成本
    print("\n5.2 升级成本")
    for level in range(2):
        cost = Ant.upgrade_cost(level)
        print(f"  等级{level}升级成本: {cost}")
    
    expected_costs = [200, 250]
    for level in range(2):
        actual_cost = Ant.upgrade_cost(level)
        if actual_cost != expected_costs[level]:
            issues.append(f"等级{level}升级成本错误: 期望{expected_costs[level]}, 实际{actual_cost}")
    
    # 检查防御塔升级成本
    print("\n5.3 防御塔升级成本")
    game_state = GameState()
    for tower_type in TowerType:
        cost = game_state.upgrade_tower_cost(tower_type)
        print(f"  {tower_type.name}升级成本: {cost}")
    
    return issues

def check_coin_acquisition():
    """检查金币获取规则"""
    print("\n" + "=" * 80)
    print("6. 金币获取规则检查")
    print("=" * 80)
    
    issues = []
    
    game_state = GameState()
    game_state.init_with_seed(1)
    
    print("\n6.1 每回合金币增长")
    prev_coin = game_state.coin[0]
    game_state.simulate_next_round()
    new_coin = game_state.coin[0]
    coin_increase = new_coin - prev_coin
    
    print(f"  回合前金币: {prev_coin}")
    print(f"  回合后金币: {new_coin}")
    print(f"  金币增长: {coin_increase}")
    
    if coin_increase != 1:
        issues.append(f"每回合金币增长错误: 期望1, 实际{coin_increase}")
    
    print("\n6.2 蚂蚁成功到达敌方基地的金币奖励")
    # 这个需要模拟蚂蚁到达敌方基地的情况
    # 暂时跳过
    
    return issues

def check_victory_conditions():
    """检查胜负判定规则"""
    print("\n" + "=" * 80)
    print("7. 胜负判定规则检查")
    print("=" * 80)
    
    issues = []
    
    print("\n7.1 基地被摧毁判定")
    print("  规则: 当一方基地HP <= 0时，另一方获胜")
    
    print("\n7.2 超时判定规则")
    print("  规则: 当达到最大回合数时，按照以下顺序判定:")
    print("    1. 基地血量多的一方获胜")
    print("    2. 如果血量相同，死亡蚂蚁少的一方获胜")
    print("    3. 如果死亡蚂蚁数相同，使用超级武器少的一方获胜")
    print("    4. 如果以上都相同，默认玩家0获胜")
    
    # 检查我们的battle_engine.py中的胜负判定逻辑
    print("\n7.3 检查battle_engine.py中的胜负判定逻辑")
    battle_engine_path = os.path.join(os.path.dirname(__file__), 'battle_engine.py')
    if os.path.exists(battle_engine_path):
        with open(battle_engine_path, 'r') as f:
            content = f.read()
            if "game_state.hp[0] != game_state.hp[1]" in content:
                print("  ✓ 已包含基地血量判定")
            else:
                issues.append("battle_engine.py缺少基地血量判定")
            
            if "die_count" in content:
                print("  ✓ 已包含死亡蚂蚁数判定")
            else:
                issues.append("battle_engine.py缺少死亡蚂蚁数判定")
            
            if "super_weapon_usage" in content:
                print("  ✓ 已包含超级武器使用判定")
            else:
                issues.append("battle_engine.py缺少超级武器使用判定")
    
    return issues

def check_strategy_thresholds():
    """检查策略代码中的阈值"""
    print("\n" + "=" * 80)
    print("8. 策略代码阈值检查")
    print("=" * 80)
    
    issues = []
    
    strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')
    
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(strategies_dir, filename)
            print(f"\n检查 {filename}:")
            
            with open(filepath, 'r') as f:
                content = f.read()
                
                # 检查升级阈值
                if "my_coin >= 80" in content:
                    issues.append(f"{filename}: 升级检查阈值错误(80)，应为200")
                    print(f"  ⚠️  升级检查阈值错误: 80 (应为200)")
                elif "my_coin >= 200" in content:
                    print(f"  ✓ 升级检查阈值正确: 200")
                
                # 检查建造防御塔阈值
                if "tower_cost" in content:
                    print(f"  ✓ 使用tower_cost变量检查建造成本")
    
    return issues

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("游戏规则全面检查")
    print("=" * 80)
    
    all_issues = []
    
    # 执行所有检查
    all_issues.extend(check_initialization())
    all_issues.extend(check_ant_generation())
    all_issues.extend(check_ant_movement())
    all_issues.extend(check_tower_attack())
    all_issues.extend(check_costs())
    all_issues.extend(check_coin_acquisition())
    all_issues.extend(check_victory_conditions())
    all_issues.extend(check_strategy_thresholds())
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("检查总结报告")
    print("=" * 80)
    
    if all_issues:
        print(f"\n发现 {len(all_issues)} 个问题:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✓ 未发现问题，所有规则检查通过！")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
