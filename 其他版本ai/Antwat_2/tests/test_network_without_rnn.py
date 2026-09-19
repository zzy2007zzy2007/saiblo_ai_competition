import sys
import torch

sys.path.insert(0, 'src')

from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork


def test_network_without_rnn():
    print("Testing AntWarPolicyValueNetwork with use_rnn=False...")
    
    # Create network with use_rnn=False
    network = AntWarPolicyValueNetwork(
        board_shape=(28, 19, 19),
        global_dim=30,
        action_dim=96,
        hidden_dim=256,
        use_rnn=False
    )
    
    print(f"Network created successfully!")
    print(f"use_rnn: {network.use_rnn}")
    print(f"RNN module exists: {hasattr(network, 'rnn') and network.rnn is not None}")
    
    # Test forward pass
    batch_size = 2
    board = torch.randn(batch_size, 28, 19, 19)
    global_features = torch.randn(batch_size, 30)
    action_mask = torch.ones(batch_size, 96)
    
    print(f"\nTesting forward pass...")
    action_logits, value, rnn_states = network(board, global_features, action_mask)
    
    print(f"action_logits shape: {action_logits.shape}")
    print(f"value shape: {value.shape}")
    print(f"rnn_states: {rnn_states}")
    
    assert action_logits.shape == (batch_size, 96)
    assert value.shape == (batch_size, 1)
    assert rnn_states is None
    
    print(f"\nTesting get_action...")
    action, action_log_prob, value_out, rnn_states_out = network.get_action(
        board, global_features, action_mask, deterministic=True
    )
    
    print(f"action shape: {action.shape}")
    print(f"action_log_prob shape: {action_log_prob.shape}")
    print(f"value_out shape: {value_out.shape}")
    print(f"rnn_states_out: {rnn_states_out}")
    
    assert action.shape == (batch_size,)
    assert action_log_prob.shape == (batch_size,)
    assert value_out.shape == (batch_size, 1)
    assert rnn_states_out is None
    
    print(f"\nTesting evaluate_actions...")
    actions = torch.randint(0, 96, (batch_size,))
    action_log_prob_eval, value_eval, dist_entropy = network.evaluate_actions(
        board, global_features, actions, action_mask
    )
    
    print(f"action_log_prob_eval shape: {action_log_prob_eval.shape}")
    print(f"value_eval shape: {value_eval.shape}")
    print(f"dist_entropy shape: {dist_entropy.shape}")
    
    assert action_log_prob_eval.shape == (batch_size,)
    assert value_eval.shape == (batch_size,)
    assert dist_entropy.shape == (batch_size,)
    
    print("\n✅ All tests passed!")


def test_network_with_rnn():
    print("\n\nTesting AntWarPolicyValueNetwork with use_rnn=True (default)...")
    
    # Create network with use_rnn=True (default)
    network = AntWarPolicyValueNetwork(
        board_shape=(28, 19, 19),
        global_dim=30,
        action_dim=96,
        hidden_dim=256,
        use_rnn=True
    )
    
    print(f"Network created successfully!")
    print(f"use_rnn: {network.use_rnn}")
    print(f"RNN module exists: {hasattr(network, 'rnn') and network.rnn is not None}")
    
    # Test forward pass
    batch_size = 2
    board = torch.randn(batch_size, 28, 19, 19)
    global_features = torch.randn(batch_size, 30)
    action_mask = torch.ones(batch_size, 96)
    
    print(f"\nTesting forward pass...")
    action_logits, value, rnn_states = network(board, global_features, action_mask)
    
    print(f"action_logits shape: {action_logits.shape}")
    print(f"value shape: {value.shape}")
    print(f"rnn_states exists: {rnn_states is not None}")
    
    assert action_logits.shape == (batch_size, 96)
    assert value.shape == (batch_size, 1)
    assert rnn_states is not None
    
    print("\n✅ RNN mode test passed!")


if __name__ == "__main__":
    test_network_without_rnn()
    test_network_with_rnn()
