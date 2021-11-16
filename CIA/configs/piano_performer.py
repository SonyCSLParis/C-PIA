from pathlib import Path

num_events_context = 256
local_window_size = 64
config = {
    "dataset": "piano",
    # --- Dataloader ---
    "dataloader_generator_kwargs": dict(
        sequences_size=1024,
        transformations={
            "time_dilation": True,
            "velocity_shift": True,
            "transposition": True,
        },
        offset_beginning=-(local_window_size - 1),
        offset_end=-local_window_size,
    ),
    # --- DataProcessor ---
    "data_processor_type": "piano_prefix",  # piano_prefix | piano_prefixEnd
    "data_processor_kwargs": dict(
        embedding_size=64,
        num_events_local_window=local_window_size,
        num_events_context=num_events_context,
        reverse_prefix=False,  # only for prefixEnd
    ),  # Can be different from the encoder's data processor
    # --- Positional Embedding ---
    "positional_embedding_dict": dict(
        sinusoidal_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
        ),
        sinusoidal_remaining_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
        ),
        channel_embedding=dict(positional_embedding_size=12, num_channels=4),
        sinusoidal_progress_bar_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.0,
        ),
    ),
    # --- Start Of Sequence embeddings
    "sos_embedding_dict": dict(
        learnt_sos_embedding=dict(
            embedding_size=512  # sum must be equal to d_model_decoder
        )
    ),
    # --- Handler type ---
    "handler_type": "channel",  # event | channel
    # --- Decoder ---
    "decoder_kwargs": dict(
        # autoregressive_decoding only needed if handler_type == 'event
        autoregressive_decoding=None,  # fullcat | mlp | None
        type="performer",
        d_model=512,
        n_head=8,
        local_attn_heads=4,
        fast_local_attn=False,
        local_window_size=local_window_size,  # works with batch_size = 8
        num_decoder_layers=12,
        dropout=0.1,
        label_smoothing=False,
        features={
            "type": "elu",  # favor | elu | None (for standard Transformer)
            # 'args': dict(n_features=256),  # favor args
            "args": dict(),  # elu args
        },
        execute_type="reversible",  # gated (recommended) | reversible | sequential
        layer_pe=None,
    ),
    # ======== Training ========
    "lr": 1e-4,
    "batch_size": 4,
    "num_batches": 512,
    "num_epochs": 3000000,
    # ======== model ID ========
    "timestamp": None,
    "savename": Path(__file__).stem,
}
