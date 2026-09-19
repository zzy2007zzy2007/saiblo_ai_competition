#!/usr/bin/env python3
"""调试FeatureExtractor输出"""
import sys
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.utils.features import FeatureExtractor
import numpy as np

print("=" * 50)
print("调试 FeatureExtractor")
print("=" * 50)

backend = load_backend()
state = backend.initial_state(seed=42)
game_state = PythonBackendState(state)

feature_extractor = FeatureExtractor()
action_mask = np.ones(32, dtype=np.float32)
observation = feature_extractor.encode_observation(state, 0, action_mask)

print(f"observation type: {type(observation)}")
print(f"observation keys: {observation.keys() if isinstance(observation, dict) else 'not dict'}")

flattened = feature_extractor.flatten_observation(observation)
print(f"flattened type: {type(flattened)}")

if hasattr(flattened, 'astype'):
    print(f"has astype, shape: {flattened.shape}")
    print(f"dtype: {flattened.dtype}")
else:
    print(f"NO astype!")
    if hasattr(flattened, '__len__'):
        print(f"len: {len(flattened)}")
        if len(flattened) > 0:
            print(f"first element type: {type(flattened[0])}")
            if isinstance(flattened[0], list):
                print(f"flattened is a list of lists!")
                flattened = np.array(flattened, dtype=np.float32)
                print(f"converted to numpy array, shape: {flattened.shape}")