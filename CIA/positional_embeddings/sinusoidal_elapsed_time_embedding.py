from CIA.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
from CIA.utils import flatten
import torch
import math


class SinusoidalElapsedTimeEmbedding(BasePositionalEmbedding):
    def __init__(
        self,
        positional_embedding_size,
        num_channels,
        dataloader_generator,
        data_processor,
        dropout,
        expand_channels,
        **kwargs
    ):
        super(SinusoidalElapsedTimeEmbedding, self).__init__(
            expand_channels=expand_channels
        )
        assert positional_embedding_size % 2 == 0
        self.data_processor = data_processor
        self.dataloader_generator = dataloader_generator
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels

    def forward(self, x_embed, i, metadata_dict):
        assert i == 0
        batch_size, num_events, _ = x_embed.size()

        # add embedding_dim to elapsed time
        elapsed_time = self.data_processor.compute_elapsed_time(metadata_dict)
        elapsed_time = elapsed_time.unsqueeze(2)
        # TODO scale?! only 10?!
        elapsed_time = elapsed_time * 100
        if self.expand_channels:
            elapsed_time = elapsed_time.repeat_interleave(self.num_channels, dim=1)
        else:
            elapsed_time = elapsed_time

        # sinusoids
        pe = torch.zeros(batch_size, num_events, self.positional_embedding_size)
        pos_embedding = pe.to(device=x_embed.device)
        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float()
            * (-math.log(10000.0) / self.positional_embedding_size)
        )
        div_term = div_term.to(device=x_embed.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pos_embedding[:, :, 0::2] = torch.sin(elapsed_time * div_term)
        pos_embedding[:, :, 1::2] = torch.cos(elapsed_time * div_term)

        pos_embedding = self.dropout(pos_embedding)
        x_embed = torch.cat([x_embed, pos_embedding], dim=2)
        return x_embed

    def forward_step(self, x, i=0, metadata_dict={}):
        if not self.expand_channels:
            raise NotImplementedError

        # time_shift must be the last feature
        assert (
            self.dataloader_generator.features.index("time_shift")
            == len(self.dataloader_generator.features) - 1
        )

        assert (
            "original_token" in metadata_dict
        ), 'Dictionnary metadata_dict must contain entry "original_token" in order to compute the elapsed time'

        batch_size = x.size(0)
        raise NotImplementedError("quelle valeur pour elapsed_time ici")
        elapsed_time = None

        pe = torch.zeros(batch_size, self.positional_embedding_size)
        pe = pe.to(device=x.device)

        div_term = torch.exp(
            torch.arange(0, self.positional_embedding_size, 2).float()
            * (-math.log(10000.0) / self.positional_embedding_size)
        )
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, 1::2] = torch.cos(elapsed_time * div_term)

        # dropout only on pe
        pe = self.dropout(pe)
        x_embed = torch.cat([x, pe], dim=1)
        return x_embed
