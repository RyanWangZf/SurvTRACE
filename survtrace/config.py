from easydict import EasyDict

STConfig = EasyDict(
    {
        'data': 'metabric',
        'num_durations': 5,
        'horizons': [.25, .5, .75],
        'seed': 1234,
        'checkpoint': './checkpoints/survtrace.pt',
        'vocab_size': 8,
        'hidden_size': 16,
        'intermediate_size': 64,
        'num_hidden_layers': 3,
        'num_attention_heads': 2,
        'hidden_dropout_prob': 0.0,
        'num_feature': 9,
        'num_numerical_feature': 5,
        'num_categorical_feature': 4,
        'out_feature':3,
        'hidden_act': 'gelu',
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'early_stop_patience': 5,
        'initializer_range': 0.001,
        'layer_norm_eps': 1e-12,
        'chunk_size_feed_forward': 0,
        'output_attentions': False,
        'output_hidden_states': False,
        'tie_word_embeddings': True,
        'pruned_heads': {},
    }
)