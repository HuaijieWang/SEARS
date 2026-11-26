import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE = "<image>"
IMAGE_ID = -200
IGNORE = -100
MAX_LENGTH = 2048
SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def question_id(row, index):
    return str(row.get("question_id", row.get("id", index)))


def turns(row):
    if "conversations" in row:
        return [(str(item["from"]), str(item["value"])) for item in row["conversations"]]
    user = row.get("text", row.get("question", row.get("instruction")))
    if user is None:
        raise KeyError("Missing instruction")
    return [("human", str(user))]


def instruction(row):
    return turns(row)[0][1].replace(IMAGE, "").strip()


def image_name(row):
    name = row.get("image", row.get("images"))
    if isinstance(name, list):
        if len(name) > 1:
            raise ValueError("Expected at most one image per item")
        return name[0] if name else None
    return name


def encode_text(tokenizer, value):
    parts = [tokenizer(piece, add_special_tokens=False).input_ids for piece in value.split(IMAGE)]
    return [token for i, part in enumerate(parts) for token in (([IMAGE_ID] if i else []) + part)]


def training_tokens(row, tokenizer):
    history = turns(row)
    if history[0][0] not in {"human", "user"}:
        history = history[1:]
    prompt = SYSTEM + " "
    for index, (speaker, content) in enumerate(history):
        user = speaker in {"human", "user"}
        if user != (index % 2 == 0):
            raise ValueError("Expected alternating user and assistant turns")
        if IMAGE in content:
            content = (IMAGE + "\n" + content.replace(IMAGE, "").strip()).strip()
        role = "USER" if user else "ASSISTANT"
        prompt += role + (": " + content + (" " if user else tokenizer.eos_token) if content else ":")

    if image_name(row):
        def encode(value):
            return [tokenizer.bos_token_id] + encode_text(tokenizer, value)
    else:
        def encode(value):
            return tokenizer(value).input_ids

    ids = torch.tensor(encode(prompt)[:MAX_LENGTH], dtype=torch.long)
    labels = ids.clone()
    cursor = 1
    labels[:cursor] = IGNORE
    separator = " ASSISTANT: "
    for round_text in prompt.split(tokenizer.eos_token):
        parts = round_text.split(separator)
        if not round_text or len(parts) != 2:
            break
        instruction_length = len(encode(parts[0] + separator)) - 2
        labels[cursor:cursor + instruction_length] = IGNORE
        cursor += len(encode(round_text))
    labels[cursor:] = IGNORE
    if cursor < MAX_LENGTH and cursor != int(ids.ne(tokenizer.unk_token_id).sum()):
        labels.fill_(IGNORE)
    return ids, labels


def prepare(row, tokenizer, training, answer_prompt=""):
    if training:
        return training_tokens(row, tokenizer)
    prompt = SYSTEM + " USER: " + (IMAGE + "\n" if image_name(row) else "") + instruction(row) + answer_prompt + " ASSISTANT:"
    ids = torch.tensor(([tokenizer.bos_token_id] + encode_text(tokenizer, prompt))[:MAX_LENGTH], dtype=torch.long)
    return ids, torch.full_like(ids, IGNORE)


def load_pixels(path, processor, pad):
    with Image.open(path) as file:
        image = file.convert("RGB")
    if pad:
        width, height = image.size
        side = max(width, height)
        color = tuple(int(255 * value) for value in processor.image_mean)
        image = ImageOps.expand(image, ((side - width) // 2, (side - height) // 2, (side - width + 1) // 2, (side - height + 1) // 2), fill=color)
    return processor(images=image, return_tensors="pt").pixel_values[0]


class TaskDataset(Dataset):
    def __init__(self, rows, images, tokenizer, processor, training, pad=True, answer_prompt=""):
        self.rows = rows
        self.images = images
        self.tokenizer = tokenizer
        self.processor = processor
        self.training = training
        self.pad = pad
        self.answer_prompt = answer_prompt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids, labels = prepare(row, self.tokenizer, self.training, self.answer_prompt)
        path = image_name(row)
        pixels = load_pixels(self.images / path, self.processor, self.pad) if path else None
        return ids, labels, pixels


def collate(items):
    return tuple(list(column) for column in zip(*items))


def training_order(rows, batch, accumulation, generator):
    lengths = [max(1, sum(len(text.split()) for _, text in turns(row))) for row in rows]
    image_rows = [index for index, row in enumerate(rows) if image_name(row)]
    text_rows = [index for index, row in enumerate(rows) if not image_name(row)]
    block_size = batch * accumulation

    def blocks(indices, sign):
        shuffled = [indices[index] for index in torch.randperm(len(indices), generator=generator).tolist()]
        grouped = []
        for start in range(0, len(shuffled), block_size):
            indices_by_length = sorted(shuffled[start:start + block_size], key=lambda index: sign * lengths[index], reverse=True)
            if len(indices_by_length) % accumulation:
                microbatches = [indices_by_length[index::accumulation] for index in range(accumulation)]
            else:
                capacity = len(indices_by_length) // accumulation
                microbatches = [[] for _ in range(accumulation)]
                totals = [0] * accumulation
                for index in indices_by_length:
                    slot = min((slot for slot in range(accumulation) if len(microbatches[slot]) < capacity), key=totals.__getitem__)
                    microbatches[slot].append(index)
                    totals[slot] += sign * lengths[index]
            grouped.append([index for microbatch in microbatches for index in microbatch])
        return grouped

    if image_rows and text_rows:
        image_blocks, text_blocks = blocks(image_rows, 1), blocks(text_rows, 1)
        remainder = sorted(image_blocks.pop() + text_blocks.pop())
        groups = image_blocks + text_blocks
        groups = [groups[index] for index in torch.randperm(len(groups), generator=generator).tolist()]
        groups.append(remainder)
    else:
        groups = blocks(image_rows, 1) if image_rows else blocks(text_rows, -1)
    return [index for block in groups for index in block]
