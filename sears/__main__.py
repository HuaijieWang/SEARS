import argparse
import json
import math
import os
from pathlib import Path

from .data import image_name, read_json


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCIENCEQA_ANSWER_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def arguments():
    settings = read_json(CONFIGS / "runs.json")
    parser = argparse.ArgumentParser(prog="SEARS")
    parser.add_argument("operation", choices=("train", "eval", "check"))
    parser.add_argument("run", choices=settings)
    args = parser.parse_args()
    return args, settings[args.run]


def paths(args, settings):
    data = Path(os.environ["SEARS_DATA_ROOT"]).resolve()
    instructions = Path(os.environ.get("SEARS_INSTRUCTION_ROOT", data)).resolve()
    models = Path(os.environ["SEARS_MODEL_ROOT"]).resolve()
    output = Path(os.environ.get("SEARS_OUTPUT_ROOT", "runs")).resolve() / args.run
    benchmark = read_json(CONFIGS / (settings["benchmark"] + ".json"))
    images = data / benchmark["images"]
    tasks = []
    instruction_dir = instructions / benchmark["instructions"]
    for name, train, test, metric, *annotation in benchmark["tasks"]:
        root = data if settings["benchmark"] == "cvit" else instruction_dir
        reference = root / annotation[0] if annotation else None
        tasks.append((name, instruction_dir / train, instruction_dir / test, metric, reference))
    return models, images, tasks, output


def weight_files(directory):
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            return [directory / file for file in sorted(set(read_json(index)["weight_map"].values()))]
    return [directory / ("model.safetensors" if (directory / "model.safetensors").is_file() else "pytorch_model.bin")]


def validate(operation, models, images, tasks, output, settings):
    clip = models / "clip-vit-large-patch14-336"
    chat = models / settings["model"]
    vision = models / settings["vision"]
    required = [images, clip / "config.json", clip / "preprocessor_config.json", vision / "config.json", chat / "config.json", chat / "tokenizer.model", chat / "mm_projector.bin"]
    required += weight_files(clip) + weight_files(chat)
    if settings["model"] == "Internvl-chat-7b":
        index = chat / "pytorch_model.bin.index.json"
        required.extend([index, vision / "configuration_intern_vit.py", vision / "modeling_intern_vit.py"])
        if index.is_file():
            required.extend(chat / file for file in sorted(set(read_json(index)["weight_map"].values())))
    if operation != "eval":
        required.extend(train for _, train, _, _, _ in tasks)
    if operation != "train":
        required.extend(test for _, _, test, _, _ in tasks)
        required.extend(reference for _, _, _, _, reference in tasks if reference)
        required.extend(ref.with_name("pid_splits.json") for _, _, _, metric, ref in tasks if metric == "scienceqa")
    if operation == "eval":
        required.extend(checkpoint(output, index) for index in range(1, len(tasks) + 1))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(missing))


def checkpoint(output, stage):
    return output / "checkpoints" / f"stage-{stage:02}.pt"


def check_image_paths(images, tasks):
    for _, train_file, test_file, _, _ in tasks:
        for split in (train_file, test_file):
            rows = read_json(split)
            if not rows:
                raise SystemExit(f"Empty split: {split}")
            sample = next((path for row in rows if (path := image_name(row))), None)
            if sample and not (images / sample).is_file():
                raise SystemExit(f"Missing image referenced by {split}: {images / sample}")


def train(model, encoder, settings, images, tasks, output):
    from .attention import training_attention

    model.llm.gradient_checkpointing_enable()
    with training_attention(model.llm):
        train_stages(model, encoder, settings, images, tasks, output)


def train_stages(model, encoder, settings, images, tasks, output):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from .core import adaptive_retention_synthesis, allocate
    from .data import TaskDataset, collate, instruction, training_order

    state = {"prototypes": [], "assignment": [], "roots": []}
    accumulation = settings["accumulate"]
    for stage, (name, train_file, _, _, _) in enumerate(tasks, 1):
        dataset = TaskDataset(read_json(train_file), images, model.tokenizer, model.processor, True)
        sampler_seed = torch.Generator().manual_seed(42)
        order = training_order(dataset.rows, settings["batch"], accumulation, sampler_seed)
        loader = DataLoader(dataset, batch_size=settings["batch"], sampler=order, generator=sampler_seed, num_workers=settings["workers"], collate_fn=collate, pin_memory=True)
        prototype = encoder.prototype([instruction(row) for row in dataset.rows])
        root = allocate(prototype, state["prototypes"], state["assignment"], len(model.projectors))
        reuse = root < len(model.projectors)
        old = model.snapshot(root) if reuse else None
        model.select_for_training(root)
        model.train()
        adapters = [p for module in model.adapters.values() for p in module.experts[root].parameters()]
        projector = list(model.projectors[root].parameters())
        optimizer = torch.optim.AdamW([{"params": adapters, "lr": 2e-4}, {"params": projector, "lr": 2e-5}], weight_decay=0.0)
        steps = math.ceil(len(loader) / accumulation)
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps * 0.03), steps)
        print(f"Stage {stage}: {name}; {'reuse' if reuse else 'allocate'} root {root + 1}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        for step, batch in enumerate(loader, 1):
            last = step % accumulation == 0 or step == len(loader)
            group_size = min(accumulation, len(loader) - ((step - 1) // accumulation) * accumulation)
            with torch.autocast("cuda", dtype=model.dtype):
                loss = model(*batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at stage {stage}, step {step}")
            (loss / group_size).backward()
            if last:
                torch.nn.utils.clip_grad_norm_(adapters + projector, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"{stage}:{step}/{len(loader)} loss={float(loss):.5f}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        del optimizer, scheduler
        state["prototypes"].append(prototype)
        state["assignment"].append(root)
        acquired = model.snapshot(root)
        if reuse:
            acquired["factors"] = adaptive_retention_synthesis(old["factors"], acquired["factors"], model.device)
            model.restore(root, acquired)
            state["roots"][root] = acquired
        else:
            state["roots"].append(acquired)
        target = checkpoint(output, stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(target)


def evaluate(model, encoder, images, tasks, output, run, settings):
    import torch

    from .data import TaskDataset, instruction
    from .metrics import aggregate, score

    matrix = []
    model.eval()
    for stage in range(1, len(tasks) + 1):
        state = torch.load(checkpoint(output, stage), map_location="cpu", weights_only=True)
        for root, expert in enumerate(state["roots"]):
            model.restore(root, expert)
        stage_results = []
        for name, _, test_file, metric, annotation in tasks[:stage]:
            rows = read_json(test_file)
            options = settings["eval"]
            if metric == "scienceqa":
                options = {**options, **options["scienceqa"]}
            prompt = SCIENCEQA_ANSWER_PROMPT if metric == "scienceqa" else ""
            dataset = TaskDataset(rows, images, model.tokenizer, model.processor, False, pad=options["pad_images"], answer_prompt=prompt)
            probabilities = [encoder.encode([instruction(row) for row in rows[start:start + 128]]).cpu() for start in range(0, len(rows), 128)]
            features = torch.cat(probabilities)
            predictions = [model.predict(item, feature.unsqueeze(0).to(model.device), state, options["max_new_tokens"]) for item, feature in zip(dataset, features)]
            value = score(metric, rows, predictions, annotation)
            stage_results.append(value)
            print(f"Stage {stage}: {name} {value:.4f}", flush=True)
        matrix.append(stage_results)
    result = {"tasks": [task[0] for task in tasks], "stage_scores": matrix, "metrics": aggregate(matrix, run)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), flush=True)


def main():
    args, settings = arguments()
    models, images, tasks, output = paths(args, settings)
    validate(args.operation, models, images, tasks, output, settings)
    if args.operation == "check":
        check_image_paths(images, tasks)
        print(f"{args.run}: split files, model weights, and sampled image paths found")
        return
    import torch
    from transformers import set_seed

    from .model import InstructionEncoder, SEARS

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("SEARS requires a CUDA GPU supporting bfloat16")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(42)
    dtype = torch.bfloat16 if args.operation == "train" else getattr(torch, settings["eval"]["dtype"])
    model = SEARS(models / settings["model"], models / settings["vision"], settings, device, dtype)
    encoder = InstructionEncoder(models / "clip-vit-large-patch14-336", device)
    if args.operation == "train":
        train(model, encoder, settings, images, tasks, output)
    else:
        evaluate(model, encoder, images, tasks, output, args.run, settings)


if __name__ == "__main__":
    main()
