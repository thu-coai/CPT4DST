from prompt_files.t5_model.modeling_t5 import *
from prompt_files.t5_model.configuration_t5 import T5Config
from prompt_files.prompts_config import PROMPT_TOKENS, META_PROMPT_TOKENS
import numpy as np
from prompt_files.transformer_utils import logging
logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class T5ForPromptDSTConfig(T5Config):
    def __init__(
            self,
            **all_args,
    ):
        super().__init__(**all_args)
        self.num_prompt_tokens = len(PROMPT_TOKENS)
        self.num_meta_prompt_tokens = len(META_PROMPT_TOKENS)
        # self.num_mlm_prompt_tokens = len(MLM_PROMPT_TOKENS)
        self.same_pos_emb_for_prompts = False

        # finetune
        self.dropout_rate = 0.0


class T5PromptAttention(T5Attention):
    def compute_bias(self, query_length, key_length, position_bias_mask=None):
        """ Compute binned relative position bias
            main change:
                position_bias_mask: [batchsize, query_length, key_length], is 1 where position bias is not needed
                                    i.e. 1 where input_ids in prompt_tokens
        """

        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)

        if position_bias_mask is not None:
            # DONE: modify pos emb for prompt token
            prompt_bias, prompt_mask = position_bias_mask  # (bs, len, len, num_heads), (bs, len)
            prompt_bias_mask = prompt_mask.unsqueeze(1).repeat(1, prompt_mask.size(1), 1)  # (bs, len, len)
            prompt_bias_mask[prompt_mask] = True
            prompt_bias_mask = prompt_bias_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # (bs, num_heads, len, len)
            values = values.repeat(prompt_mask.size(0), 1, 1, 1)  # (bs, num_heads, len, len)
            values = torch.where(prompt_bias_mask == 0, values, prompt_bias.permute([0, 3, 1, 2]))
            # print(position_bias_mask.sum(-1))
            # print(position_bias_mask.shape)
            # from pprint import pprint
            # print(pprint(position_bias_mask.tolist()))
            # relative_position_bucket = relative_position_bucket.unsqueeze(0).repeat(position_bias_mask.shape[0], 1, 1)  # (bs, q, k)
            # relative_position_bucket = torch.where(position_bias_mask==0, relative_position_bucket, torch.ones_like(relative_position_bucket)*(self.relative_attention_num_buckets-1))  # ()
            # values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
            # values = values.permute([0, 3, 1, 2])
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        position_bias_mask=None
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """  projection """
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """  reshape """
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """ projects hidden states correctly to key/query states """
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, position_bias_mask)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5PromptLayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = T5PromptAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        position_bias_mask=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_bias_mask=position_bias_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5PromptBlock(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5PromptLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        position_bias_mask=None,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_bias_mask=position_bias_mask,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PromptStack(T5Stack):
    # mainly change:
    #   add position_bias parameter to forward()
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5PromptBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

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
        position_bias_mask=None,
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

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
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
                position_bias_mask=position_bias_mask,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

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



class T5ForPromptDST(T5ForConditionalGeneration):
    config_class = T5ForPromptDSTConfig

    def __init__(self, config:T5ForPromptDSTConfig):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.prompt_embedder = nn.Embedding(config.num_prompt_tokens, config.d_model)
        self.prompt_bias = nn.Embedding(config.num_prompt_tokens, config.num_heads, _weight=torch.zeros(config.num_prompt_tokens,config.num_heads))
        self.meta_prompt_embedder = nn.Embedding(config.num_meta_prompt_tokens, config.d_model)
        self.meta_prompt_bias = nn.Embedding(config.num_meta_prompt_tokens, config.num_heads, _weight=torch.zeros(config.num_meta_prompt_tokens,config.num_heads))
        # self.mlm_prompt_embedder = nn.Embedding(config.num_mlm_prompt_tokens, config.d_model)

        self.vocab_size = config.vocab_size
        self.prompt_size = config.num_prompt_tokens

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5PromptStack(encoder_config, None)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5PromptStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def assign_prompt_embedding(self, promptidss2tokenids):
        # new_embeddings.weight.data[:num_tokens_to_copy, :]
        for prompt_idx, token_idx in promptidss2tokenids.items():
            self.prompt_embedder.weight.data[prompt_idx, :] = self.shared.weight.data[token_idx, :]


    def initialize_prompt_by_trained_prompt(self, prompt_init_dict):
        for target_prompt_idx, init_prompt_idx in prompt_init_dict.items():
            self.prompt_embedder.weight.data[target_prompt_idx, :] = self.prompt_embedder.weight.data[init_prompt_idx, :]
            self.prompt_bias.weight.data[target_prompt_idx, :] = self.prompt_bias.weight.data[init_prompt_idx, :]

    def initialize_prompt_by_trained_metaprompt(self, prompt_init_dict):
        for target_prompt_idx, init_prompt_idx in prompt_init_dict.items():
            self.prompt_embedder.weight.data[target_prompt_idx, :] = self.meta_prompt_embedder.weight.data[init_prompt_idx, :]
            self.prompt_bias.weight.data[target_prompt_idx, :] = self.meta_prompt_bias.weight.data[init_prompt_idx, :]

    def initialize_metaprompt_by_trained_prompt(self, prompt_init_dict):
        for target_prompt_idx, init_prompt_idx in prompt_init_dict.items():
            self.meta_prompt_embedder.weight.data[target_prompt_idx, :] = self.prompt_embedder.weight.data[init_prompt_idx, :]
            self.meta_prompt_bias.weight.data[target_prompt_idx, :] = self.prompt_bias.weight.data[init_prompt_idx, :]

    def initialize_prompt_embedder(self, init_style):
        if init_style == 'random':
            return
        elif init_style == 'vocab_dist':
            vocab_mean = torch.mean(self.shared.weight.data, dim=0)
            vocab_var = torch.var(self.shared.weight.data, dim=0)
            init_weight = torch.zeros_like(self.prompt_embedder.weight.data)
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            for _dim in range(embed_size):
                init_weight[:, _dim] = torch.distributions.Normal(loc=vocab_mean[_dim], scale=vocab_var[_dim]).sample((num_prompt,))
            self.prompt_embedder.weight.data = init_weight

        elif init_style == 'vocab_sample':
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            sampled_vocab_idxs = np.random.choice(self.vocab_size, size=num_prompt, replace=True)
            init_weight = self.shared.weight.data.index_select(dim=0, index=torch.tensor(sampled_vocab_idxs))
            self.prompt_embedder.weight.data = init_weight

        elif init_style == 'vocab_sample_avg':
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            sampled_vocab_idxs = np.random.choice(self.vocab_size, size=10*num_prompt, replace=True)
            init_weight = self.shared.weight.data.index_select(dim=0, index=torch.tensor(sampled_vocab_idxs))
            init_weight = torch.split(init_weight, 10, dim=0)
            init_weight = [torch.mean(_, dim=0) for _ in init_weight]
            init_weight = torch.stack(init_weight, dim=0)
            self.prompt_embedder.weight.data = init_weight

        elif init_style == 'vocab_sample_same':
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            sampled_vocab_idxs = np.random.choice(self.vocab_size, size=1, replace=True)
            init_weight = self.shared.weight.data.index_select(dim=0, index=torch.tensor(sampled_vocab_idxs))
            init_weight = init_weight.repeat(num_prompt, 1)
            self.prompt_embedder.weight.data = init_weight

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ):
        # retrieve encoder hidden states
        encoder = self.get_encoder()
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
        }
        inputs_embeds = self.convert_input_ids_to_input_embeds(input_ids)
        enc_pos_attention_mask = None
        if self.config.same_pos_emb_for_prompts:
            enc_pos_attention_mask = self.prepare_pos_attn_mask_for_encoder_prompt(input_ids)
        model_kwargs["encoder_outputs"] = encoder(input_ids=None, inputs_embeds=inputs_embeds, return_dict=True,
                                                  position_bias_mask=enc_pos_attention_mask, **encoder_kwargs)
        return model_kwargs


    def convert_input_ids_to_input_embeds(self, input_ids):
        if input_ids is None:
            return None
        vocab_ids = torch.where(input_ids < self.vocab_size, input_ids, torch.zeros_like(input_ids))
        vocab_embeds = self.shared(vocab_ids)
        prompt_ids = input_ids - self.vocab_size
        prompt_ids = torch.where((prompt_ids >= 0) & (prompt_ids < self.config.num_prompt_tokens), prompt_ids,
                                 torch.zeros_like(prompt_ids))
        prompt_embeds = self.prompt_embedder(prompt_ids)
        inputs_embeds = torch.where((input_ids < self.vocab_size).unsqueeze(-1), vocab_embeds, prompt_embeds)

        # meta_prompt embeddings
        meta_prompt_ids = input_ids - self.vocab_size - self.config.num_prompt_tokens
        meta_prompt_ids = torch.where(meta_prompt_ids >= 0, meta_prompt_ids, torch.zeros_like(meta_prompt_ids))
        meta_prompt_embeds = self.meta_prompt_embedder(meta_prompt_ids)
        inputs_embeds = torch.where((input_ids < self.vocab_size+self.config.num_prompt_tokens).unsqueeze(-1), inputs_embeds, meta_prompt_embeds)

        # # mlm_prompt embeddings
        # mlm_prompt_ids = input_ids - self.vocab_size - self.config.num_prompt_tokens - self.config.num_meta_prompt_tokens
        # mlm_prompt_ids = torch.where(mlm_prompt_ids >= 0, mlm_prompt_ids, torch.zeros_like(mlm_prompt_ids))
        # mlm_prompt_embeds = self.mlm_prompt_embedder(mlm_prompt_ids)
        # inputs_embeds = torch.where((input_ids < self.vocab_size + self.config.num_prompt_tokens + self.config.num_meta_prompt_tokens).unsqueeze(-1),
        #                             inputs_embeds, mlm_prompt_embeds)

        return inputs_embeds


    def prepare_pos_attn_mask_for_encoder_prompt(self, input_ids):
        if input_ids is None:
            return None
        bs, seqlen = input_ids.shape

        prompt_ids = input_ids - self.vocab_size
        prompt_mask = (prompt_ids >= 0) & (prompt_ids < self.config.num_prompt_tokens)  # (bs, len)
        meta_prompt_ids = input_ids - self.vocab_size - self.config.num_prompt_tokens
        meta_prompt_mask = meta_prompt_ids >= 0
        bias_mask = prompt_ids >= 0

        prompt_ids = torch.where(prompt_mask, prompt_ids, torch.zeros_like(prompt_ids))  # (bs, len)
        prompt_bias = self.prompt_bias(prompt_ids)  # (bs, len, num_heads)
        prompt_bias[~prompt_mask] = 0
        meta_prompt_ids = torch.where(meta_prompt_mask, meta_prompt_ids, torch.zeros_like(meta_prompt_ids))
        meta_prompt_bias = self.meta_prompt_bias(meta_prompt_ids)
        meta_prompt_bias[~meta_prompt_mask] = 0

        bias = prompt_bias + meta_prompt_bias

        bias = bias.unsqueeze(1).repeat(1, seqlen, 1, 1)  # (bs, len, len, num_heads)
        bias[bias_mask] = 0  # (bs, len, len, num_heads)  no bias for prompt as query

        return bias, bias_mask


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            only_encoder=False
    ):
        enc_pos_attention_mask = None
        if self.config.same_pos_emb_for_prompts:
            enc_pos_attention_mask = self.prepare_pos_attn_mask_for_encoder_prompt(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.convert_input_ids_to_input_embeds(input_ids)
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.convert_input_ids_to_input_embeds(decoder_input_ids)


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_bias_mask=enc_pos_attention_mask,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if only_encoder:
            return Seq2SeqLMOutput(
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



class T5ForPromptEncDecDST(T5ForPromptDST):
    def set_same_prompt_pos_emb(self):
        self.config.same_pos_emb_for_prompts = True


    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            'decoder_inputs_embeds': self.convert_input_ids_to_input_embeds(input_ids),
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def save_prompt(self, save_directory):
        # only save prompt parameters
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)

        # Attach architecture to the config
        self.config.architectures = [self.__class__.__name__]

        # Save the config
        self.config.save_pretrained(save_directory)

        # Save the model
        state_dict = {k: v for k, v in self.state_dict().items() if 'prompt' in k}

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(state_dict, output_model_file)

        logger.info("Prompt parameters saved in {}".format(output_model_file))

    def load_prompt(self, load_directory):
        state_dict = torch.load(os.path.join(load_directory, 'pytorch_model.bin'))
        self.load_state_dict(state_dict)

        logger.info("Load prompt parameters from {}".format(load_directory))
