from prompt_files.t5_model.modeling_t5 import *

class T5ForConditionalGenerationEmbeddingPrompt(T5ForConditionalGeneration):
    def convert_special_token_reprs(self, reprs):
        return reprs

    def get_decoder_input_embeds_by_ids(self, decoder_input_ids):
        decoder_input_embeds = self.decoder.embed_tokens(decoder_input_ids)  # [bs, len, dim]
        return decoder_input_embeds

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if 'decoder_special_token_reprs_for_inputs' in kwargs:
            special_token_reprs_for_inputs = kwargs['decoder_special_token_reprs_for_inputs']
            waiting_for_prompt_token = kwargs['decoder_waiting_for_prompt_token']
            generated_prompt_token_mask = decoder_input_ids == waiting_for_prompt_token
            decoder_inputs_embeds = self.get_decoder_input_embeds_by_ids(decoder_input_ids)

            fill_prompt_mask = torch.cumsum(generated_prompt_token_mask, dim=-1)
            fill_prompt_mask = fill_prompt_mask <= torch.tensor(
                [_.shape[0] for _ in special_token_reprs_for_inputs]).unsqueeze(-1).cuda()
            fill_prompt_mask = fill_prompt_mask * generated_prompt_token_mask
            fill_reprs_num = torch.sum(fill_prompt_mask.int(), dim=-1).tolist()
            prompts_to_fill = []

            for sample_idx, repr_num in enumerate(fill_reprs_num):
                prompts_to_fill.append(special_token_reprs_for_inputs[sample_idx][:repr_num, :])

            if len(prompts_to_fill) > 0:
                prompts_to_fill = torch.cat(prompts_to_fill, dim=0)
                decoder_inputs_embeds[fill_prompt_mask] = prompts_to_fill

            if past is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:, :]

            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": None,
                "decoder_inputs_embeds": decoder_inputs_embeds,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
        else:
            if past is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            return {
                "decoder_input_ids": decoder_input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }


class T5ForPrompt(T5ForConditionalGeneration):
    def get_decoder_input_embeds_by_ids(self, decoder_input_ids):
        decoder_input_embeds = self.decoder.embed_tokens(decoder_input_ids)  # [bs, len, dim]
        return decoder_input_embeds


    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        # print('='*50)
        # print(decoder_input_ids)
        if 'decoder_special_token_reprs_for_inputs' in kwargs:
            special_token_reprs_for_inputs = kwargs['decoder_special_token_reprs_for_inputs']
            waiting_for_prompt_token = kwargs['decoder_waiting_for_prompt_token']
            generated_prompt_token_mask = decoder_input_ids == waiting_for_prompt_token
            decoder_inputs_embeds = self.get_decoder_input_embeds_by_ids(decoder_input_ids)
            # print(decoder_inputs_embeds[:, :, 0])
            fill_prompt_mask = torch.cumsum(generated_prompt_token_mask, dim=-1)
            fill_prompt_mask = fill_prompt_mask <= torch.tensor([_.shape[0] for _ in special_token_reprs_for_inputs]).unsqueeze(-1).cuda()
            fill_prompt_mask = fill_prompt_mask * generated_prompt_token_mask
            # print(fill_prompt_mask)
            fill_reprs_num = torch.sum(fill_prompt_mask.int(), dim=-1).tolist()
            prompts_to_fill = []

            for sample_idx, repr_num in enumerate(fill_reprs_num):
                prompts_to_fill.append(special_token_reprs_for_inputs[sample_idx][:repr_num, :])

            if len(prompts_to_fill) > 0:
                prompts_to_fill = torch.cat(prompts_to_fill, dim=0)
                decoder_inputs_embeds[fill_prompt_mask] = prompts_to_fill
            # print(decoder_inputs_embeds[:, :, 0])

            if past is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:, :]

            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": None,
                "decoder_inputs_embeds": decoder_inputs_embeds,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
        else:
            if past is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            return {
                "decoder_input_ids": decoder_input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
            }


class T5ForConditionalGenerationAttentionPrompt(T5ForPrompt):
    def __init__(self, config):
        super().__init__(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )


    def create_special_token_reprs(self, outputs1, special_token_mask_wmask, args):
        # extract slot representations from outputs1
        # for mlm, directly returns reprs at special token positions
        encoder_hidden_states = outputs1.encoder_last_hidden_state  # [bs, encoder_len, dim]
        dec_enc_attentions = outputs1.cross_attentions  # (bs, nhead, decoder_len, encoder_len) * 6
        dec_enc_attentions = sum(dec_enc_attentions)  # [bs, nhead, decoder_len, encoder_len]
        dec_enc_attentions = torch.sum(dec_enc_attentions, dim=1)  # [bs, decoder_len, encoder_len]
        dec_enc_attention_vectors = dec_enc_attentions.unsqueeze(-1) * encoder_hidden_states.unsqueeze(
            1)  # [bs, decoder_len, encoder_len, None] * [bs, None, encoder_len, dim] = [bs, decoder_len, encoder_len, dim]
        dec_enc_attention_vectors = torch.sum(dec_enc_attention_vectors, dim=2)  # [bs, decoder_len, dim]
        special_token_reprs = dec_enc_attention_vectors[special_token_mask_wmask]  # [nmask, dim]
        return special_token_reprs


    def convert_special_token_reprs(self, reprs, normalize=False):
        # convert slot representations from template output to prompt input
        reprs = self.mlp(reprs)
        assert len(reprs.shape) == 2
        if normalize:
            reprs = torch.nn.functional.normalize(reprs, dim=-1)
        return reprs



class T5ForDSTAttentionPrompt(T5ForConditionalGenerationAttentionPrompt):
    def create_special_token_reprs(self, outputs1, special_token_mask_wmask, args):
        # for dst, returns average reprs at special token positions
        encoder_hidden_states = outputs1.encoder_last_hidden_state  # [bs, encoder_len, dim]
        dec_enc_attentions = outputs1.cross_attentions  # (bs, nhead, decoder_len, encoder_len) * 6
        dec_enc_attentions = sum(dec_enc_attentions)  # [bs, nhead, decoder_len, encoder_len]
        dec_enc_attentions = torch.sum(dec_enc_attentions, dim=1)  # [bs, decoder_len, encoder_len]
        dec_enc_attention_vectors = dec_enc_attentions.unsqueeze(-1) * encoder_hidden_states.unsqueeze(
            1)  # [bs, decoder_len, encoder_len, None] * [bs, None, encoder_len, dim] = [bs, decoder_len, encoder_len, dim]
        dec_enc_attention_vectors = torch.sum(dec_enc_attention_vectors, dim=2)  # [bs, decoder_len, dim]
        special_token_reprs = dec_enc_attention_vectors[special_token_mask_wmask]  # [nmask, dim]
        special_token_reprs = torch.mean(special_token_reprs, dim=0)
        special_token_reprs = special_token_reprs.unsqueeze(0).repeat(encoder_hidden_states.shape[0], 1)
        return special_token_reprs


class T5ForDSTAttentionPromptSelect(T5ForConditionalGenerationAttentionPrompt):
    # softly choose one template representation as slot prompt
    def create_special_token_reprs(self, outputs1, special_token_mask_wmask, args):
        # for dst, returns average reprs at special token positions
        encoder_hidden_states = outputs1.encoder_last_hidden_state  # [bs, encoder_len, dim]
        dec_enc_attentions = outputs1.cross_attentions  # (bs, nhead, decoder_len, encoder_len) * 6
        dec_enc_attentions = sum(dec_enc_attentions)  # [bs, nhead, decoder_len, encoder_len]
        dec_enc_attentions = torch.sum(dec_enc_attentions, dim=1)  # [bs, decoder_len, encoder_len]
        dec_enc_attention_vectors = dec_enc_attentions.unsqueeze(-1) * encoder_hidden_states.unsqueeze(
            1)  # [bs, decoder_len, encoder_len, None] * [bs, None, encoder_len, dim] = [bs, decoder_len, encoder_len, dim]
        dec_enc_attention_vectors = torch.sum(dec_enc_attention_vectors, dim=2)  # [bs, decoder_len, dim]
        special_token_reprs = dec_enc_attention_vectors[special_token_mask_wmask]  # [nmask, dim]
        special_token_reprs = torch.mean(special_token_reprs, dim=0)
        special_token_reprs = special_token_reprs.unsqueeze(0).repeat(encoder_hidden_states.shape[0], 1)
        return special_token_reprs


class T5ForDSTEmbeddingPrompt(T5ForPrompt):
    def __init__(self, config: T5Config, prompt_config={}):
        super().__init__(config)
        prompt_style = prompt_config.pop('prompt_style', 'direct')
        num_prompt = prompt_config.pop('num_prompt', 30)
        prompt_init = prompt_config.pop('prompt_init', None)
        self.prompt_transform = nn.Identity()
        if prompt_style == 'mlp':
            self.prompt_transform = torch.nn.Sequential(
                torch.nn.Linear(self.config.d_model, self.config.d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.d_model, self.config.d_model))
        elif prompt_style == 'linear':
            self.prompt_transform = torch.nn.Linear(self.config.d_model, self.config.d_model)

        if prompt_init is None:
            self.prompt_embedding = nn.Embedding(num_prompt, self.config.d_model)
            self.init_weights()
        else:
            self.prompt_embedding = nn.Embedding(num_prompt, self.config.d_model, _weight=prompt_init)


    def get_prompt_reprs_by_dsidx(self, dsidx, train_embedding=True):
        prompt_embeds = self.prompt_embedding(dsidx)
        if not train_embedding:
            prompt_embeds.detach()
        # prompt_embeds = prompt_embeds.view(prompt_embeds.shape[0], prompt_embeds.shape[2])
        prompt_embeds = self.prompt_transform(prompt_embeds)  # [num_ds, 1, dim]
        return prompt_embeds








