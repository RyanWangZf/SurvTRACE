from easydict import EasyDict

STConfig = EasyDict(
    {
        'data': 'metabric', # dataset name, in 'metabric', 'support', or 'seer'
        'num_durations': 5, # num of discrete intervals for prediction, e.g., num_dur = 5 means the whole period is discretized to be 5 intervals
        'horizons': [.25, .5, .75], # the discrete intervals are cut at 0%, 25%, 50%, 75%, 100%
        'seed': 1234,
        'checkpoint': './checkpoints/survtrace.pt',
        'vocab_size': 8, # num of all possible values of categorical features
        'hidden_size': 16, # embedding size
        'intermediate_size': 64, # intermediate layer size in transformer layer
        'num_hidden_layers': 3, # num of transformers
        'num_attention_heads': 2, # num of attention heads in transformer layer
        'hidden_dropout_prob': 0.0,
        'num_feature': 9, # num of covariates of patients, should be set during load_data
        'num_numerical_feature': 5, # num of numerical covariates of patients, should be set during load_data
        'num_categorical_feature': 4, # num of categorical covariates of patients, should be set during load_data
        'out_feature':3, # equals to the length of 'horizons', indicating the output dim of the logit layer of survtrace
        'num_event': 1, # only set when using SurvTraceMulti for competing risks
        'hidden_act': 'gelu',
        'attention_probs_dropout_prob': 0.1,
        'early_stop_patience': 5,
        'initializer_range': 0.001,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512, # # no use
        'chunk_size_feed_forward': 0, # no use
        'output_attentions': False, # no use
        'output_hidden_states': False, # no use 
        'tie_word_embeddings': True, # no use
        'pruned_heads': {}, # no use
    }
)
