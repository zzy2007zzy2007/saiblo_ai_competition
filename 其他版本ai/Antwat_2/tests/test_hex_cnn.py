import torch
import sys
sys.path.insert(0, '/Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src')

from ppo_antwar.network.antwar_net import HexCNNEncoder, AntWarPolicyValueNetwork

def test_hex_cnn_encoder():
    print("Testing HexCNNEncoder...")
    
    batch_size = 2
    channels = 28
    height = 19
    width = 19
    
    x = torch.randn(batch_size, channels, height, width)
    
    encoder = HexCNNEncoder(input_channels=channels, hidden_channels=128, map_size=height)
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(x)
    
    expected_dim = encoder.get_output_dim()
    assert output.shape == (batch_size, expected_dim), f"Expected shape { (batch_size, expected_dim) }, got { output.shape }"
    print(f"✓ HexCNNEncoder forward pass successful. Output shape: { output.shape }")

def test_antwar_policy_value_network():
    print("\nTesting AntWarPolicyValueNetwork...")
    
    batch_size = 2
    board_shape = (28, 19, 19)
    global_dim = 30
    action_dim = 96
    
    board = torch.randn(batch_size, *board_shape)
    global_features = torch.randn(batch_size, global_dim)
    action_mask = torch.ones(batch_size, action_dim)
    
    network = AntWarPolicyValueNetwork(
        board_shape=board_shape,
        global_dim=global_dim,
        action_dim=action_dim,
        hidden_dim=256,
        use_hex_cnn=True
    )
    network.eval()
    
    with torch.no_grad():
        action_logits, value, rnn_states = network(board, global_features, action_mask)
    
    assert action_logits.shape == (batch_size, action_dim), f"Expected action logits shape { (batch_size, action_dim) }, got { action_logits.shape }"
    assert value.shape == (batch_size, 1), f"Expected value shape { (batch_size, 1) }, got { value.shape }"
    print(f"✓ AntWarPolicyValueNetwork forward pass successful")
    print(f"  - Action logits shape: { action_logits.shape }")
    print(f"  - Value shape: { value.shape }")

def test_get_hex_neighbor_sum():
    print("\nTesting get_hex_neighbor_sum...")
    
    batch_size = 1
    channels = 3
    height = 5
    width = 5
    
    x = torch.zeros(batch_size, channels, height, width)
    x[0, 0, 2, 2] = 1.0
    
    encoder = HexCNNEncoder(input_channels=channels, hidden_channels=128, map_size=height)
    
    with torch.no_grad():
        neighbor_tensor = encoder.get_hex_neighbor_sum(x)
    
    expected_shape = (batch_size, channels * 6, height, width)
    assert neighbor_tensor.shape == expected_shape, f"Expected shape { expected_shape }, got { neighbor_tensor.shape }"
    print(f"✓ get_hex_neighbor_sum output shape: { neighbor_tensor.shape }")

if __name__ == "__main__":
    test_get_hex_neighbor_sum()
    test_hex_cnn_encoder()
    test_antwar_policy_value_network()
    print("\n✅ All tests passed!")
