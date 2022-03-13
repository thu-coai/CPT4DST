from prompt_files.gpt2.modeling_gpt2 import *
from prompt_files.gpt2.configuration_gpt2 import GPT2Config
from prompt_files.p_tuning.prompts_config import PROMPT_TOKENS
import numpy as np


class GPTForPromptDSTConfig(GPT2Config):
    def __init__(
            self,
            **all_args,
    ):
        super().__init__(**all_args)
        self.num_prompt_tokens = all_args.pop('num_prompt_tokens', len(PROMPT_TOKENS))
        self.d_model = self.n_embd


class GPTForPromptDST(GPT2LMHeadModel):
    config_class = GPTForPromptDSTConfig

    def __init__(self, config: GPTForPromptDSTConfig):
        super().__init__(config)
        self.prompt_embedder = nn.Embedding(config.num_prompt_tokens, config.d_model)
        self.vocab_size = self.transformer.wte.weight.data.shape[0]
        self.prompt_size = config.num_prompt_tokens


    def assign_prompt_embedding(self, promptidss2tokenids):
        # new_embeddings.weight.data[:num_tokens_to_copy, :]
        for prompt_idx, token_idx in promptidss2tokenids.items():
            self.prompt_embedder.weight.data[prompt_idx, :] = self.transformer.wte.weight.data[token_idx, :]


    def initialize_prompt_embedder(self, init_style):
        if init_style == 'random':
            return
        elif init_style == 'vocab_dist':
            vocab_mean = torch.mean(self.transformer.wte.weight.data, dim=0)
            vocab_var = torch.var(self.transformer.wte.weight.data, dim=0)
            init_weight = torch.zeros_like(self.prompt_embedder.weight.data)
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            for _dim in range(embed_size):
                init_weight[:, _dim] = torch.distributions.Normal(loc=vocab_mean[_dim], scale=vocab_var[_dim]).sample((num_prompt,))
            self.prompt_embedder.weight.data = init_weight

        elif init_style == 'vocab_sample':
            num_prompt, embed_size = self.prompt_embedder.weight.data.shape
            sampled_vocab_idxs = np.random.choice(self.vocab_size, size=num_prompt, replace=True)
            init_weight = self.transformer.wte.weight.data.index_select(dim=0, index=torch.tensor(sampled_vocab_idxs))
            self.prompt_embedder.weight.data = init_weight


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
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
            "input_ids": None,
            "inputs_embeds": self.convert_input_ids_to_input_embeds(input_ids),
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def convert_input_ids_to_input_embeds(self, input_ids):
        if input_ids is None:
            return None
        vocab_ids = torch.where(input_ids < self.vocab_size, input_ids, torch.zeros_like(input_ids))
        vocab_embeds = self.transformer.wte(vocab_ids)
        prompt_ids = input_ids - self.vocab_size
        prompt_ids = torch.where(prompt_ids >= 0, prompt_ids, torch.zeros_like(prompt_ids))
        prompt_embeds = self.prompt_embedder(prompt_ids)
        inputs_embeds = torch.where((input_ids < self.vocab_size).unsqueeze(-1), vocab_embeds, prompt_embeds)
        return inputs_embeds


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
    ):
        if inputs_embeds is None:
            inputs_embeds = self.convert_input_ids_to_input_embeds(input_ids)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert inputs_embeds is not None
        transformer_outputs = self.transformer(
            None,
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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

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