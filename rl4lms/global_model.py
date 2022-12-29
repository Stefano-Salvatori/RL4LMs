from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from torch.cuda.amp import autocast


class SummarizationModel:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._max_new_tokens = None
        self._special_tokens = None
        self._use_led_global_attention = None
        self.num_beams = None

    def istantiate(
        self,
        base_model: str = "",
        max_new_tokens: int = 256,
        num_beams: int = 5,
        device: torch.device = None,
        load_from_state_dict: bool = False,
        load_path: str = None,
        special_tokens=["<sl>", "<\sl>"],
        use_led_global_attention: bool = False,
    ) -> None:
        assert not self.is_available(), "Model already istantiated"

        self._special_tokens = special_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)

        if load_from_state_dict:
            assert load_path is not None, "load_path can't be None if load_from_state_dict is True"
            # Add special tokens to the tokenizer
            self._tokenizer.add_special_tokens({"additional_special_tokens": self._special_tokens})
            # Istantiate base model from base configurations (e.g., MingZhong/DialogLED-large-5120)
            self._model = AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(base_model))
            # Load pretrained weights. We remove the 'language_backbone' string from the weights names since it is not present in the base architecture
            weights = {
                k.replace("language_backbone.", ""): v for k, v in torch.load(load_path, map_location="cpu").items()
            }
            self._model.resize_token_embeddings(len(self._tokenizer))
            self._model.load_state_dict(weights)
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        # self._model.half()
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        self._use_led_global_attention = use_led_global_attention
        self.num_beams = num_beams

    def summarize(self, texts: List[str]) -> List[str]:
        assert self.is_available(), "Model must be istantiated"
        with torch.no_grad():
            encoded_input = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self._device)
            if self._use_led_global_attention:
                global_attention_mask = torch.zeros_like(encoded_input.attention_mask)
                global_attention_mask[:, 0] = 1
                for special_token in self._special_tokens:
                    global_attention_mask[
                        encoded_input.input_ids == self._tokenizer.convert_tokens_to_ids(special_token)
                    ] = 1
                encoded_input["global_attention_mask"] = global_attention_mask
            outputs = self._model.generate(
                **encoded_input, max_new_tokens=self._max_new_tokens, num_beams=self.num_beams
            )
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def is_available(self) -> bool:
        return self._model is not None


# We create a global summarization model since we will use it both for computing the reward and computing the metrics
GLOBAL_SUMMARIZATION_MODEL = SummarizationModel()
