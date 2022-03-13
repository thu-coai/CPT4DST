from prompt_files.p_tuning.modeling_prompt_t5 import *


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        # self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = self.wi_0(hidden_states)
        hidden_states = self.gelu_act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5PrefixStack(T5Stack):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prefix_hidden_states=None,
        prompt_mask=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value, prefix_hidden) in enumerate(zip(self.block, past_key_values, prefix_hidden_states)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if encoder_layer_head_mask is not None:
                    encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # prefix_tuning
            # prompt_mask: [bs, total_len]
            hidden_states[(prompt_mask==1)] = prefix_hidden

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )



# cannot use now
# class T5ForPrefixDST(T5ForPromptEncDecDST):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_dim = config.d_model
#
#         self.shared = nn.Embedding(config.vocab_size, config.d_model)
#         self.prefix_mlps = nn.ModuleList([MLP(config.d_model, config.d_model) for _ in range(config.num_layers)])
#         self.prompt_embedder = nn.Embedding(config.num_prompt_tokens, config.d_model)
#         self.vocab_size = config.vocab_size
#         self.prompt_size = config.num_prompt_tokens
#
#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5PrefixStack(encoder_config, None)
#
#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5PrefixStack(decoder_config, self.shared)
#
#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
#
#         self.init_weights()
#
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#
#
#     def assign_prompt_embedding(self, promptidss2tokenids):
#         # new_embeddings.weight.data[:num_tokens_to_copy, :]
#         for prompt_idx, token_idx in promptidss2tokenids.items():
#             self.prompt_embedder.weight.data[prompt_idx] = self.shared.weight.data[token_idx, :].unsqueeze(0)
#
#
#     def convert_input_ids_to_input_embeds(self, input_ids):
#         if input_ids is None:
#             return None
#         vocab_ids = torch.where(input_ids < self.vocab_size, input_ids, torch.zeros_like(input_ids))
#         vocab_embeds = self.shared(vocab_ids)
#         prompt_ids = input_ids - self.vocab_size
#         prompt_ids = torch.where(prompt_ids >= 0, prompt_ids, torch.zeros_like(prompt_ids))
#         prompt_embeds = self.prompt_embedder(prompt_ids)
#         inputs_embeds = torch.where((input_ids < self.vocab_size).unsqueeze(-1), vocab_embeds, prompt_embeds)
#         return inputs_embeds
#
#
#     def get_prompt_mask(self, input_ids):
#         return input_ids >= self.vocab_size
#
#
#     def get_prefix_hidden_states(self, prompt_embeds):
#         hidden_states = (prefix_embedder(prompt_embeds) for prefix_embedder in self.prefix_mlps)
#         return hidden_states
#
#
#     def _prepare_encoder_decoder_kwargs_for_generation(
#         self, input_ids: torch.LongTensor, model_kwargs
#     ):
#         # retrieve encoder hidden states
#         encoder = self.get_encoder()
#         encoder_kwargs = {
#             argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
#         }
#         inputs_embeds = self.convert_input_ids_to_input_embeds(input_ids)
#         input_prompt_mask = self.get_prompt_mask(input_ids)
#         input_prefix_states = self.get_prefix_hidden_states(inputs_embeds[input_prompt_mask])
#         model_kwargs["encoder_outputs"] = encoder(input_ids=None,
#                                                   inputs_embeds=inputs_embeds,
#                                                   return_dict=True,
#                                                   prompt_mask=input_prompt_mask,
#                                                   prefix_hidden_states=input_prefix_states,
#                                                   **encoder_kwargs)
#         return model_kwargs
#
#
#     def prepare_inputs_for_generation(
#         self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
#     ):
#
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             input_ids = input_ids[:, -1:]
#
#         return {
#             "decoder_input_ids": input_ids,
#             "past_key_values": past,
#             "encoder_outputs": encoder_outputs,
#             "attention_mask": attention_mask,
#             "use_cache": use_cache,
#         }
#
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
#             config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
#             labels in ``[0, ..., config.vocab_size]``
#
#         Returns:
#
#         Examples::
#
#             >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
#
#             >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
#             >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
#
#             >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
#             >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
#             >>> outputs = model(input_ids=input_ids, labels=labels)
#             >>> loss = outputs.loss
#             >>> logits = outputs.logits
#
#             >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
#             >>> outputs = model.generate(input_ids)
#         """
#
#         decoder_inputs_embeds = self.convert_input_ids_to_input_embeds(decoder_input_ids)
#         decoder_input_prompt_mask = self.get_prompt_mask(decoder_input_ids)
#         decoder_prefix_states = self.get_prefix_hidden_states(decoder_inputs_embeds[decoder_input_prompt_mask])
#
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask
#
#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             inputs_embeds = self.convert_input_ids_to_input_embeds(input_ids)
#             input_prompt_mask = self.get_prompt_mask(input_ids)
#             input_prefix_states = self.get_prefix_hidden_states(inputs_embeds[input_prompt_mask])
#
#             encoder_outputs = self.encoder(
#                 input_ids=None,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 prefix_hidden_states=input_prefix_states,
#                 prompt_mask=input_prompt_mask,
#             )
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#
#         hidden_states = encoder_outputs[0]
#
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#
#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)
#
#         # If decoding with past key value states, only the last tokens
#         # should be given as an input
#         if past_key_values is not None:
#             assert labels is None, "Decoder should not use cached key value states when training."
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids[:, -1:]
#             if decoder_inputs_embeds is not None:
#                 decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
#
#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=None,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             encoder_head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             prefix_hidden_states=decoder_prefix_states,
#             prompt_mask=decoder_input_prompt_mask,
#         )
#
#         sequence_output = decoder_outputs[0]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.encoder.first_device)
#             self.lm_head = self.lm_head.to(self.encoder.first_device)
#             sequence_output = sequence_output.to(self.lm_head.weight.device)
#
#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#             sequence_output = sequence_output * (self.model_dim ** -0.5)
#
#         lm_logits = self.lm_head(sequence_output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
#
#         if not return_dict:
#             output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#             return ((loss,) + output) if loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )