from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPVisionModel, LlamaConfig, LlamaForCausalLM

from .core import RANK, SCALE, root_probabilities
from .data import IGNORE, IMAGE_ID, MAX_LENGTH, read_json


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Factors(nn.Module):
    def __init__(self, size_in, size_out, device):
        super().__init__()
        self.a = nn.Parameter(torch.empty(RANK, size_in, device=device))
        self.b = nn.Parameter(torch.empty(size_out, RANK, device=device))
        self.reset()

    def reset(self):
        nn.init.normal_(self.a, std=0.01)
        nn.init.zeros_(self.b)

    def forward(self, inputs):
        return SCALE * F.linear(F.linear(inputs, self.a.to(inputs.dtype)), self.b.to(inputs.dtype))


class ExpertLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.experts = nn.ModuleList()
        self.probabilities = None

    def add(self):
        self.experts.append(Factors(self.linear.in_features, self.linear.out_features, self.linear.weight.device))

    def forward(self, x):
        output = self.linear(x)
        if self.probabilities is None:
            raise RuntimeError("Missing root weights")
        if isinstance(self.probabilities, int):
            return output + self.experts[self.probabilities](F.dropout(x, 0.05, self.training))
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        for root, probability in enumerate(self.probabilities.unbind(dim=1)):
            weight = probability.to(output.dtype).reshape(shape)
            output = output + weight * self.experts[root](x)
        return output


class InstructionEncoder:
    def __init__(self, path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = CLIPTextModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        return self.model(**inputs.to(self.device)).pooler_output.float()

    @torch.no_grad()
    def prototype(self, texts):
        total = None
        for start in range(0, len(texts), 128):
            batch = self.encode(texts[start:start + 128]).sum(dim=0)
            total = batch if total is None else total + batch
        if total is None or not torch.isfinite(total).all() or not total.norm() > 0:
            raise ValueError("Invalid task prototype")
        return F.normalize(total, dim=0).cpu()


def load_vision(path, model_path, intern, device, dtype):
    if intern:
        processor = CLIPImageProcessor(size=336, crop_size=336, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config.image_size = 336
        config.use_flash_attn = False
        with init_empty_weights():
            vision = AutoModel.from_config(config, trust_remote_code=True)
        prefix = "model.vision_tower.vision_tower."
        weights = read_json(model_path / "pytorch_model.bin.index.json")["weight_map"]
        expected = set(vision.state_dict())
        files = sorted({file for key, file in weights.items() if key.startswith(prefix)})
        loaded = set()
        for file in files:
            raw = torch.load(model_path / file, map_location="cpu", weights_only=True)
            for name in sorted(expected - loaded):
                key = prefix + name
                if weights.get(key) == file:
                    set_module_tensor_to_device(vision, name, device, value=raw[key], dtype=dtype)
                    loaded.add(name)
            del raw
        if loaded != expected:
            raise RuntimeError(f"Missing embedded InternViT weights: {sorted(expected - loaded)}")
    else:
        processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
        vision = CLIPVisionModel.from_pretrained(path, local_files_only=True, torch_dtype=dtype)
    return vision.to(device).requires_grad_(False).eval(), processor


def load_projector(path, input_size, output_size):
    projector = nn.Sequential(nn.Linear(input_size, output_size), nn.GELU(), nn.Linear(output_size, output_size))
    raw = torch.load(path / "mm_projector.bin", map_location="cpu", weights_only=True)
    state = {key.split("mm_projector.")[-1]: value for key, value in raw.items() if key.split("mm_projector.")[-1] in projector.state_dict()}
    projector.load_state_dict(state, strict=True)
    return projector


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class SEARS(nn.Module):
    def __init__(self, model_path, vision_path, settings, device, dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.layer = settings["layer"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
        self.llm = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=self.dtype, low_cpu_mem_usage=True, local_files_only=True).to(device)
        self.llm.requires_grad_(False)
        self.vision, self.processor = load_vision(vision_path, model_path, settings["model"] == "Internvl-chat-7b", device, self.dtype)
        self.projector_template = load_projector(model_path, self.vision.config.hidden_size, config.hidden_size)
        self.projector_template.requires_grad_(False)
        self.projectors = nn.ModuleList()
        modules = {}
        scope = TARGETS if settings["scope"] == "all" else TARGETS[-3:]
        for name, layer in list(self.llm.named_modules()):
            parent, _, suffix = name.rpartition(".")
            if isinstance(layer, nn.Linear) and suffix in scope:
                wrapper = ExpertLinear(layer)
                setattr(self.llm.get_submodule(parent), suffix, wrapper)
                modules[name] = wrapper
        if len(modules) != config.num_hidden_layers * len(scope):
            raise RuntimeError("Unexpected LoRA module layout")
        self.adapters = modules
        self.active = None

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def ensure_roots(self, count):
        while len(self.projectors) < count:
            self.projectors.append(deepcopy(self.projector_template).to(self.device))
            for module in self.adapters.values():
                module.add()

    def select_for_training(self, root):
        self.ensure_roots(root + 1)
        self.active = root
        for index, projector in enumerate(self.projectors):
            projector.requires_grad_(index == root)
        for module in self.adapters.values():
            module.probabilities = root
            for index, factors in enumerate(module.experts):
                factors.requires_grad_(index == root)

    def set_routing(self, features, state):
        weights = root_probabilities(features, torch.stack(state["prototypes"]).to(self.device), state["assignment"])
        self.active = weights
        for module in self.adapters.values():
            module.probabilities = weights

    def snapshot(self, root):
        return {
            "projector": cpu_state(self.projectors[root]),
            "factors": {name: (module.experts[root].a.detach().cpu().clone(), module.experts[root].b.detach().cpu().clone()) for name, module in self.adapters.items()},
        }

    @torch.no_grad()
    def restore(self, root, state):
        self.ensure_roots(root + 1)
        self.projectors[root].load_state_dict(state["projector"])
        for name, (a, b) in state["factors"].items():
            self.adapters[name].experts[root].a.copy_(a)
            self.adapters[name].experts[root].b.copy_(b)

    def project(self, patches, index):
        with torch.autocast("cuda", dtype=self.dtype):
            if isinstance(self.active, int):
                return self.projectors[self.active](patches)
            return sum(weight * self.projectors[root](patches) for root, weight in enumerate(self.active[index].to(self.dtype)))

    def pack(self, ids, labels, pixels):
        embeddings = self.llm.get_input_embeddings()
        positions = [index for index, image in enumerate(pixels) if image is not None]
        patches = {}
        if positions:
            images = torch.stack([pixels[index] for index in positions]).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                features = self.vision(pixel_values=images, output_hidden_states=True).hidden_states[self.layer][:, 1:]
            patches = dict(zip(positions, features))
        vectors, targets = [], []
        for index, (tokens, target) in enumerate(zip(ids, labels)):
            tokens = tokens.to(self.device)
            target = target.to(self.device)
            marker = (tokens == IMAGE_ID).nonzero().flatten()
            if marker.numel():
                if marker.numel() != 1 or index not in patches:
                    raise ValueError("Image tokens do not match the image")
                i = int(marker[0])
                image_tokens = self.project(patches[index], index)
                vector = torch.cat((embeddings(tokens[:i]), image_tokens, embeddings(tokens[i + 1:])))
                target = torch.cat((target[:i], target.new_full((len(image_tokens),), IGNORE), target[i + 1:]))
            else:
                vector = embeddings(tokens)
                if self.training:
                    dummy = vector.new_zeros((1, self.vision.config.hidden_size))
                    vector = vector + 0 * self.project(dummy, index).sum()
            vectors.append(vector[:MAX_LENGTH])
            targets.append(target[:MAX_LENGTH])
        length = torch.tensor([len(vector) for vector in vectors], device=self.device)
        mask = torch.arange(int(length.max()), device=self.device)[None, :] < length[:, None]
        return pad_sequence(vectors, batch_first=True), pad_sequence(targets, batch_first=True, padding_value=IGNORE), mask

    def forward(self, ids, labels, pixels):
        vectors, targets, mask = self.pack(ids, labels, pixels)
        if not vectors.requires_grad:
            vectors.requires_grad_(True)
        return self.llm(inputs_embeds=vectors, labels=targets, attention_mask=mask, use_cache=False).loss

    @torch.inference_mode()
    def predict(self, item, features, state, max_new_tokens):
        self.set_routing(features, state)
        vectors, _, mask = self.pack(*collate_one(item))
        tokens = self.llm.generate(inputs_embeds=vectors, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True).strip()


def collate_one(item):
    return ([item[0]], [item[1]], [item[2]])
