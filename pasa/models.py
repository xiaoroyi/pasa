# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_ACTIVE_AGENT = None
_ACTIVE_AGENT_LOCK = threading.Lock()


class Agent:
    def __init__(self, model_name, low_vram=None, model_device=None):
        self.model_name = model_name
        self.low_vram = (
            os.getenv("PASA_LOW_VRAM", "0") == "1"
            if low_vram is None
            else low_vram
        )
        self.model_device = model_device or os.getenv("PASA_MODEL_DEVICE", "cuda:0")
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        if not self.low_vram:
            self.load()

    def _device_map(self):
        if self.model_device.lower() in {"auto", ""}:
            return "auto"
        return {"": self.model_device}

    def load(self):
        if self.model is not None:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self._device_map(),
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def unload(self):
        if self.model is None:
            return
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self):
        global _ACTIVE_AGENT
        if not self.low_vram:
            return
        with _ACTIVE_AGENT_LOCK:
            if _ACTIVE_AGENT is not None and _ACTIVE_AGENT is not self:
                _ACTIVE_AGENT.unload()
            self.load()
            _ACTIVE_AGENT = self

    def _model_device(self):
        self._ensure_loaded()
        return next(self.model.parameters()).device

    def infer_score(self, prompts):
        if len(prompts) == 0:
            return []
        self._ensure_loaded()
        encoded_input = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        device = self._model_device()
        input_ids = encoded_input.input_ids.to(device)
        attention_mask = encoded_input.attention_mask.to(device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False
            )
            true_token_id = self.tokenizer.convert_tokens_to_ids('True')
            probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].cpu().numpy().tolist()
        return probs

    def infer(self, prompt, sample=False):
        self._ensure_loaded()
        text = self.tokenizer.apply_chat_template(
            [{
                "content": prompt.strip(),
                "role":    "user"
            }],
            tokenize=False,
            max_length=992,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self._model_device())
        if sample:
            model_inputs["do_sample"] = True
            model_inputs["temperature"] = 2.0
            model_inputs["top_p"] = 0.8

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def batch_infer(self, prompts, batch_size=8, sample=False):
        if len(prompts) == 0:
            return []
        self._ensure_loaded()
        texts = [self.tokenizer.apply_chat_template(
            [{
                "content": prompt.strip(),
                "role":    "user"
            }],
            tokenize=False,
            max_length=992,
            add_generation_prompt=True
        ) for prompt in prompts]
        responses = []
        for i in range(0, len(texts), batch_size):
            model_inputs = self.tokenizer(texts[i: i + batch_size], return_tensors="pt", truncation=True, padding=True).to(self._model_device())
            if sample:
                model_inputs["do_sample"] = True
                model_inputs["temperature"] = 2.0
                model_inputs["top_p"] = 0.8
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            for response in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True):
                responses.append(response)
        return responses
    
if __name__ == "__main__":
    selector = Agent("/mnt/hdfs/foundation/agent/heyc/checkpoints/pasa-7b-selector")
    promtp = "You are an elite researcher in the field of AI, conducting research on Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\nSearched Paper:\nTitle: Specialized Language Models with Cheap Inference from Limited Domain Data\nAbstract:  Abstract Large language models have emerged as a versatile tool but are challenging to apply to tasks lacking large inference budgets and large in-domain training sets. This work formalizes these constraints and distinguishes four important variables: the pretraining budget (for training before the target domain is known), the specialization budget (for training after the target domain is known), the inference budget, and the in-domain training set size. Across these settings, we compare different approaches from the machine learning literature. Limited by inference cost, we find better alternatives to the standard practice of training very large vanilla transformer models. In particular, we show that hyper-networks and mixture of experts have better perplexity for large pretraining budgets, while small models trained on importance sampled datasets are attractive for large specialization budgets. \n\nUser Query: Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n\n\nOutput format: Decision: True/False\nReason:... \nDecision:"
    print(selector.infer_score([promtp, promtp, promtp]))
