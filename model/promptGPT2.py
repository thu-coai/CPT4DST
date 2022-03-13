from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F


class GPT2Prompt(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_embedder = None
        self.prompt_len = config.prompt_len
        self.domain2prompt = {}  # {task_id: <${task_id}_domain_id>}
        self.vocab_size = self.transformer.wte.weight.data.shape[0]

    def add_prompt(self, tokenizer: GPT2Tokenizer, init_style=None, **promptids2tokenids):
        # add prompt when new domain comes
        new_task_id = len(self.domain2prompt)
        new_prompt = ['<{}_domain_{}>'.format(new_task_id, i) for i in range(self.prompt_len)]
        new_prompt_ids = [len(tokenizer) + i for i in range(self.prompt_len)]
        self.domain2prompt[new_task_id] = new_prompt_ids
        # assume tokenizer add tokens in order
        tokenizer.add_tokens(new_prompt)
        if not self.prompt_embedder:
            self.prompt_embedder = nn.Embedding(self.prompt_len, self.config.d_model)
        else:
            old_num_tokens = self.prompt_embedder.weight.shape[0]
            new_num_tokens = old_num_tokens + self.prompt_len
            new_prompt_embedder = nn.Embedding(new_num_tokens, self.config.d_model).to(
                self.device, dtype=self.prompt_embedder.weight.dtype
            )
            self._init_weights(new_prompt_embedder)
            new_prompt_embedder.weight.data[:old_num_tokens, :] = self.prompt_embedder.weight.data[:old_num_tokens, :]
            # TODO: check if memory release?
            self.prompt_embedder = new_prompt_embedder
            # initial
            if init_style == 'vocab_sample':
                sampled_vocab_idxs = np.random.choice(self.vocab_size, size=self.prompt_len, replace=True)
                init_weight = self.transformer.wte.weight.data.index_select(dim=0,
                                                                            index=torch.tensor(sampled_vocab_idxs))
                self.prompt_embedder.weight.data[old_num_tokens:, :] = init_weight

            for prompt_idx, token_idx in promptids2tokenids.items():
                self.prompt_embedder.weight.data[prompt_idx, :] = self.transformer.wte.weight.data[token_idx, :]

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # TODO
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_id=-1,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
