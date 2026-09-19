"""简化的ActionCatalog - 不使用一步前瞻搜索"""
from SDK.utils.actions import ActionCatalog, ActionBundle
from SDK.backend.state import BackendState


class SimpleActionCatalog(ActionCatalog):
    """简化的ActionCatalog，禁用一步前瞻搜索以提高性能"""
    
    def build(self, state: BackendState, player: int) -> list[ActionBundle]:
        """构建候选动作列表，但不进行一步前瞻搜索"""
        bundles: list[ActionBundle] = [ActionBundle(name="hold", score=0.0, tags=("noop",))]
        bundles.extend(self._build_candidates(state, player))
        bundles.extend(self._upgrade_candidates(state, player))
        bundles.extend(self._downgrade_candidates(state, player))
        bundles.extend(self._base_upgrade_candidates(state, player))
        bundles.extend(self._superweapon_candidates(state, player))
        bundles.extend(self._paired_candidates(state, player, bundles[1:]))
        
        # 去重
        unique: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}
        for bundle in bundles:
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            if key not in unique or bundle.score > unique[key]:
                unique[key] = bundle
        
        # 排序并返回（不进行一步前瞻搜索）
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        return ordered[:self.max_actions]
