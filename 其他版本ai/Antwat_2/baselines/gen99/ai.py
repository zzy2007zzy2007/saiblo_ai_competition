#!/usr/bin/env python3
"""
Modified AI using actions_modifiable.py with parameter loading
"""

import os
try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.utils.actions_modifiable import ActionCatalog
from SDK.utils.actions import ActionBundle
from SDK.backend.state import BackendState


class ModifiedAgent(BaseAgent):
    def __init__(self, seed: int | None = None, max_actions: int = 32):
        super().__init__(seed=seed, max_actions=max_actions)
        # 加载参数文件并替换 catalog
        params_file = os.path.join(os.path.dirname(__file__), 'params.json')
        self.catalog = ActionCatalog(
            max_actions=max_actions, 
            feature_extractor=self.feature_extractor, 
            params_file=params_file
        )
    
    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if len(bundles) <= 1:
            return bundles[0]
        
        # Evaluate all possible actions (no truncation)
        shortlist = bundles[1 : len(bundles)]
        best = max(shortlist, key=lambda bundle: (bundle.score, -len(bundle.operations)), default=None)
        return best or bundles[0]


class AI(ModifiedAgent):
    pass


def create_agent():
    return AI()
