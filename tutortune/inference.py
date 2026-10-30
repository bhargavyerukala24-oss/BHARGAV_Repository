import json
import threading
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import SYSTEM
import torch
torch.set_num_threads(8)


class Tutor:
    def __init__(self, adapter="adapter"):
        self.lock = threading.Lock()
        metadata = json.loads(
            (Path(adapter) / "training_metadata.json").read_text()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(adapter)
        base = AutoModelForCausalLM.from_pretrained(
            metadata["base_model"],
            revision=metadata["base_revision"],
            torch_dtype=torch.float32,
        )
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.to(self.device).eval()
        self.model.config.use_cache = True

    def answer(self, question, use_adapter=True):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        if inputs["input_ids"].shape[-1] > 400:
            raise ValueError("Please shorten your question.")
        # Adapter state is shared; serialize generation and switching.
        with self.lock, torch.inference_mode():
            if use_adapter:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                with self.model.disable_adapter():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=160,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
