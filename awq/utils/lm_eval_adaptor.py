"""unused since it used outdated version of lm_eval"""

import transformers
import torch
from lm_eval.api.model import LM
from typing import List, Tuple
import fnmatch


class LMEvalAdaptor(LM):
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        # assert isinstance(self.tokenizer, (
        #     transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
        #     transformers.T5Tokenizer, transformers.T5TokenizerFast,
        # )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id
    
    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)
    
    def _loglikelihood_tokens(self, requests: List[Tuple[str, str]], **kwargs) -> List[Tuple[float, bool]]:
        results = []
        for string, cont_toks in requests:
            enc = self.tokenizer(string, return_tensors="pt").to(self.device)
            if cont_toks:
                target_enc = self.tokenizer(cont_toks, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**enc)
                    log_probs = torch.log_softmax(outputs.logits, dim=-1)
                    # Shift for proper target alignment
                    shift_logits = log_probs[..., :-1, :].contiguous()
                    shift_labels = target_enc["input_ids"][..., 1:].contiguous()

                    # Calculate log probabilities for continuation tokens.
                    log_probs_cont = shift_logits[:, -target_enc["input_ids"].shape[1]:, :].gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                    total_log_prob = log_probs_cont.sum().item()
                    results.append((total_log_prob, True)) # Assuming always valid for now
            else: # Handle cases where cont_toks is empty.
                with torch.no_grad():
                    outputs = self.model(**enc)
                    log_probs = torch.log_softmax(outputs.logits, dim=-1)
                    results.append((0.0, True)) # Or some other default value/handling

        return results

    def generate_until(self, requests: List[Tuple[str, str]], disable_tqdm: bool = False) -> List[str]:
        generations = []
        for context, until in requests:
            enc = self.tokenizer(context, return_tensors="pt").to(self.device)
            until_toks = self.tokenizer(until, return_tensors="pt").input_ids.tolist()[0] if until else []
            max_new_tokens = self.max_gen_toks

            with torch.no_grad():
                generated_ids = self.model.generate(
                    enc.input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eot_token_id, # Or appropriate EOS token ID.
                    pad_token_id=self.tokenizer.pad_token_id, # Important for padding
                    do_sample=False,  # You might want to adjust this
                )
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generations.append(generated_text)
        return generations

    def loglikelihood_rolling(self, requests: List[str]) -> List[float]:
        loglikelihoods = []
        for string in requests:
            enc = self.tokenizer(string, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**enc)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)

                # Calculate rolling log likelihood (per token)
                rolling_log_probs = []
                for i in range(1, enc.input_ids.shape[1]):
                    log_prob = log_probs[0, i-1, enc.input_ids[0, i]].item()
                    rolling_log_probs.append(log_prob)
                loglikelihoods.extend(rolling_log_probs)
        return loglikelihoods

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, "n_ctx"):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):
            return self.model.config.n_positions
        elif "bloom" in self.model_name:
            return 2048
        elif "llama" in self.model_name:
            return 2048  # TODO: did not check this
        elif "mpt" in self.model_name:
            return 2048
        elif "falcon" in self.model_name:
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if isinstance(
                self.model,
                transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
            ):
                dec_inps = torch.cat(
                    [
                        torch.tensor(
                            self.model.generation_config.decoder_start_token_id,
                        )
                        .tile(len(inps), 1)
                        .to(inps),
                        inps,
                    ],
                    dim=1,
                )

                kwargs = {
                    "decoder_input_ids": dec_inps,
                }
            else:
                kwargs = {}
            out = self.model(inps, **kwargs)[0]
            if (
                "opt" in self.model_name
            ):  # there are a few extra tokens in opt, which we should omit
                return out[:, :, :50257]
            else:
                return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
