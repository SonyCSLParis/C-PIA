from CIA.model.positional_embeddings.get_pe_input import get_pe_input
from torch import nn
from CIA.utils import flatten, categorical_crossentropy
import torch


class CausalEventsModelFullCat(nn.Module):
    def __init__(
        self,
        data_processor,
        dataloader_generator,
        positional_embedding,
        sos_embedding,
        d_model,
        num_channels_decoder,
        num_events_decoder,
        label_smoothing,
        transformer,
        pe_input_type,
    ):
        super(CausalEventsModelFullCat, self).__init__()
        self.data_processor = data_processor
        # can be useful
        self.dataloader_generator = dataloader_generator

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels_target = len(self.num_tokens_per_channel)
        assert self.num_channels_target == num_channels_decoder
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens
        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Input and layer embeddings
        self.positional_embedding = positional_embedding
        # self.layer_pos_emb = layer_pos_emb
        # self.layer_pos_emb_local = layer_pos_emb_local
        self.pe_input_type = pe_input_type
        # linear to model dim
        self.linear_target = nn.Linear(
            self.data_processor.embedding_size * self.num_channels_target
            + self.positional_embedding.positional_embedding_size,
            self.d_model,
        )

        ########################################################
        # Start of sentence
        self.sos_embedding = sos_embedding

        ######################################################
        self.transformer = transformer
        self.label_smoothing = label_smoothing

        ######################################################
        # Output dimension adjustment
        # self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
        #                                     for num_tokens_of_channel in self.num_tokens_per_channel
        #                                     ]
        #                                    )
        d_last_layer = self.transformer.dim_last_layer
        self.last_mlps = nn.ModuleList(
            [
                # MLPs for autoregressive generation
                nn.Sequential(
                    nn.Linear(
                        d_last_layer + channel_id * self.data_processor.embedding_size,
                        self.d_model * 4,
                    ),
                    nn.LeakyReLU(),
                    nn.Linear(self.d_model * 4, self.d_model),
                )
                for channel_id, num_tokens_of_channel in enumerate(
                    self.num_tokens_per_channel
                )
            ]
        )

        self.pre_softmaxes = nn.ModuleList(
            [
                nn.Linear(
                    self.d_model
                    + d_last_layer
                    + channel_id * self.data_processor.embedding_size,
                    num_tokens_of_channel,
                )
                for channel_id, num_tokens_of_channel in enumerate(
                    self.num_tokens_per_channel
                )
            ]
        )

    def __repr__(self) -> str:
        return "CausalEventsDecoderFullCat"

    def prepare_sequence(self, target_seq, metadata_dict):
        # add input positional embeddings
        target_seq = self.positional_embedding(
            target_seq, i=0, metadata_dict=metadata_dict
        )
        target_seq = self.linear_target(target_seq)

        # compute input to layer positional embeddings
        if self.pe_input_type is not None:
            layer_pos_emb_input = get_pe_input(
                data_processor=self.data_processor,
                x_embed=target_seq,
                metadata_dict=metadata_dict,
                pe_input_type=self.pe_input_type,
                event_representation=True,
            )

        # shift target_seq by one
        dummy_input_target = self.sos_embedding(metadata_dict).unsqueeze(1)
        target_seq = torch.cat([dummy_input_target, target_seq], dim=1)
        target_seq = target_seq[:, :-1]

        if self.pe_input_type is not None:
            # For dummy input on layer positional attention, we can simply repeat the first embedding
            # which corresponds either to position 0 or elapsed time 0
            layer_pos_emb_input = torch.cat(
                [layer_pos_emb_input[:, 0:1], layer_pos_emb_input], dim=1
            )
            layer_pos_emb_input = layer_pos_emb_input[:, :-1]
        else:
            layer_pos_emb_input = None
        return target_seq, layer_pos_emb_input

    def compute_event_state(self, target, metadata_dict):
        # (batch_size, num_events, num_channels, dim)
        target_embedded = self.data_processor.embed(target)
        # WE do NOT flatten here
        target_seq = torch.cat(target_embedded.split(1, dim=2), dim=3).squeeze(2)
        target_seq, layer_pos_emb_input = self.prepare_sequence(
            target_seq, metadata_dict
        )

        # forward pass
        out = self.transformer(
            target_seq,
            pos_emb_input=layer_pos_emb_input,
            # TODO add parameter
            # TODO pb with inferring_states = True
            inferring_states=False,
            states=None,
        )
        output = out["x"]

        return output, target_embedded

    def forward(self, target, metadata_dict):
        """
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        # compute event_state
        # embed + add positional embedding + offset + transformer pass
        output, target_embedded = self.compute_event_state(target, metadata_dict)

        # auto regressive predictions from output
        weights_per_category = self.event_state_to_weights(
            output=output, target_embedded=target_embedded
        )

        # we can change loss mask
        if "loss_mask" in metadata_dict:
            loss_mask = 1 - metadata_dict["loss_mask"].long()
        else:
            loss_mask = torch.ones_like(target)

        # If prefix mode, we keep track of the two separate losses
        if "decoding_start" in metadata_dict:
            decoding_start = metadata_dict["decoding_start"]
            weights_prefix = [
                weight[:, :decoding_start] for weight in weights_per_category
            ]
            target_prefix = target[:, :decoding_start]
            loss_mask_prefix = loss_mask[:, :decoding_start]
            loss_prefix = categorical_crossentropy(
                value=weights_prefix,
                target=target_prefix,
                mask=loss_mask_prefix,
                label_smoothing=self.label_smoothing,
            )

            weights_inpainting = [
                weight[:, decoding_start:] for weight in weights_per_category
            ]
            target_inpainting = target[:, decoding_start:]
            loss_mask_inpainting = loss_mask[:, decoding_start:]
            loss_inpainting = categorical_crossentropy(
                value=weights_inpainting,
                target=target_inpainting,
                mask=loss_mask_inpainting,
                label_smoothing=self.label_smoothing,
            )

            # num_tokens_prefix = loss_mask_prefix.sum()
            # num_tokens_inpainting = loss_mask_inpainting.sum()
            # loss = (loss_prefix * num_tokens_prefix + loss_inpainting * num_tokens_inpainting) / \
            #     (num_tokens_prefix + num_tokens_inpainting)

            # TODO WARNING hardcoded values:
            # different weighting for finetuning
            loss = loss_prefix * 0.1 + loss_inpainting * 0.9

            return {
                "loss": loss,
                "weights_per_category": weights_per_category,
                "monitored_quantities": {
                    "loss": loss.item(),
                    "loss_prefix": loss_prefix.item(),
                    "loss_inpainting": loss_inpainting.item(),
                },
            }

        else:
            loss = categorical_crossentropy(
                value=weights_per_category,
                target=target,
                mask=loss_mask,
                label_smoothing=self.label_smoothing,
            )

            return {
                "loss": loss,
                "weights_per_category": weights_per_category,
                "monitored_quantities": {"loss": loss.item()},
            }

    def event_state_to_weights(self, output, target_embedded):
        weights_per_category = []
        for channel_id, (mlp, pre_softmax) in enumerate(
            zip(self.last_mlps, self.pre_softmaxes)
        ):
            # mimics residual connexion:
            weight = mlp(output)
            weight = pre_softmax(torch.cat([weight, output], dim=2))
            weights_per_category.append(weight)

            # concatenate channels to output
            output = torch.cat([output, target_embedded[:, :, channel_id]], dim=2)
        return weights_per_category

    def event_state_to_weight_step(self, output, target_embedded, channel_id):
        """[summary]

        Args:
            output (batch_size, feature_dim): event_state
            target_embedded (batch_size, num_channels, feature_dim): embeddings of the channels of the current event
            channel_id ([type]): channel BEING predicted
        """
        # concatenate all already-predected channels
        for i in range(channel_id):
            output = torch.cat([output, target_embedded[:, i]], dim=1)
        weight = torch.cat([self.last_mlps[channel_id](output), output], dim=1)
        weight = self.pre_softmaxes[channel_id](weight)
        return weight

    def forward_step(self, target, metadata_dict, i):
        """
        if i == 0, target is not used: SOS instead
        :param target: sequence of tokens (batch_size,)
        :param i:f
        :return:
        """
        raise NotImplementedError

    def infer_hidden_states(self, priming_seq, metadata_dict, decoding_start_index):
        target_embedded = self.data_processor.embed(priming_seq)
        target_seq = flatten(target_embedded)
        target_seq, layer_pos_emb = self.prepare_sequence(target_seq, metadata_dict)
        out = self.transformer(
            target_seq[:, : decoding_start_index + 1],
            pos_emb_input=layer_pos_emb[:, : decoding_start_index + 1],
            inferring_states=True,
            states=None,
        )
        # softmax
        # prediction for time_index decoding_start_index
        out_x = out["x"][:, -1]
        channel_index_output = decoding_start_index % self.num_channels_target
        weights = self.pre_softmaxes[channel_index_output](out_x)
        return {
            "loss": None,
            "weights": weights,
            "Zs": out["Zs"],
            "Ss": out["Ss"],
            "Zs_rot": out["Zs_rot"],
            "Ss_rot": out["Ss_rot"],
        }

    def recurrent_step(self, target, metadata_dict, states, decoding_index):
        # aaa = time.time()
        # CRUCIAL LINE
        # metadata_dict['original_token'] = target[:, decoding_index]

        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)
        target_seq, layer_pos_emb = self.prepare_sequence(target_seq, metadata_dict)
        # bbb = time.time()
        out = self.transformer(
            target_seq[:, decoding_index : decoding_index + 1],
            pos_emb_input=layer_pos_emb[:, decoding_index : decoding_index + 1],
            inferring_states=False,
            states=states,
        )
        # softmax
        # prediction for time_index decoding_start_index
        out_x = out["x"][:, 0]
        channel_index_output = decoding_index % self.num_channels_target
        weights = self.pre_softmaxes[channel_index_output](out_x)
        # ccc = time.time()
        # print(f'Time preprocess: {bbb-aaa}\t{(bbb-aaa)/(ccc-aaa)}\%')
        # print(f'Time generation: {ccc-bbb}\t{(ccc-bbb)/(ccc-aaa)}\%')
        return {
            "loss": None,
            "weights": weights,
            "Zs": out["Zs"],
            "Ss": out["Ss"],
            "Zs_rot": out["Zs_rot"],
            "Ss_rot": out["Ss_rot"],
        }
