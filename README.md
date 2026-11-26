# SEARS: Semantic Expert Allocation with Retention-guided Synthesis

## Data, pretrained models, and instructions

```bash
export SEARS_DATA_ROOT=/path/to/datasets
export SEARS_INSTRUCTION_ROOT=/path/to/instructions
export SEARS_MODEL_ROOT=/path/to/models
export SEARS_OUTPUT_ROOT=/path/to/runs
```

Set `SEARS_INSTRUCTION_ROOT` to `SEARS_DATA_ROOT` if the instruction files and images share the same tree. The values in the last two columns are relative to those roots. Preserve the image paths in the instruction JSON files.

| Benchmark | Image download | Instruction download | Image directory | Instruction directory |
| --- | --- | --- | --- | --- |
| UCIT | [MCITlib](https://github.com/Ghy0501/MCITlib#-benchmarks) · [UCIT image links and layout](https://github.com/Ghy0501/HiDe-LLaVA#ucit-benchmark) | [HaiyangGuo/UCIT: UCIT JSON files](https://huggingface.co/datasets/HaiyangGuo/UCIT/tree/main/UCIT) | `UCIT/` | `UCIT/` |
| CVIT-Single | [SMoLoRA image preparation](https://github.com/Minato-Zackie/SMoLoRA#dataset-images) | [zackie29/SMoLoRA: Instructions_Single](https://huggingface.co/datasets/zackie29/SMoLoRA/tree/main/Instructions_Single) | `SMoLoRA/datasets/` | `SMoLoRA/CVIT_benchmark/Instructions_Single/` |
| CoIN | [CoIN image links and layout](https://github.com/zackschen/CoIN#dataset-preparation) | [HaiyangGuo/UCIT: CoIN_sample](https://huggingface.co/datasets/HaiyangGuo/UCIT/tree/main/CoIN_sample) | `CoIN/` | `CoIN/Instructions/HiDe_CoIN/` |
| TriGap | [TriGap image links and layout](https://huggingface.co/datasets/JuntaoTang/TriGap#-dataset-download) | [JuntaoTang/TriGap](https://huggingface.co/datasets/JuntaoTang/TriGap/tree/main) · [SAME/Prism split configuration](https://github.com/LAMDA-CL/Prism/blob/main/config/benchmarks/TriGap.py) | `TriGap/images/` | `TriGap/instructions/` |

- **UCIT:** Use the JSON release above. Extract `UCIT/VizWiz.zip` for the VizWiz instructions and place the `CLEVR-Math` instruction files in `UCIT/CLEVR/`. Keep `val_coco_type_3000.json` in both the VizWiz and Flickr30k instruction folders. The image directory uses `CLEVR/` as well. Both backbones use the same UCIT data.
- **CVIT-Single:** Use the six tasks listed in [cvit.json](configs/cvit.json). The image sources are [ScienceQA](https://scienceqa.github.io/), [TextVQA](https://textvqa.org/dataset/), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), [ImageNet](https://image-net.org/), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), and [COCO](https://cocodataset.org/#download) for VQAv2. Also place the ScienceQA `problems.json` and `pid_splits.json`, TextVQA `TextVQA_0.5.1_val.json`, and GQA `testdev_balanced_questions.json` at the locations listed in [cvit.json](configs/cvit.json).
- **CoIN:** Place the task folders from `CoIN_sample/` under `CoIN/Instructions/HiDe_CoIN/`, including the ScienceQA, TextVQA, and GQA evaluation labels shipped with those folders. [coin.json](configs/coin.json) lists the exact split filenames. The [full original CoIN instructions](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main/Instructions_Original) are not a substitute for the sampled splits.
- **TriGap:** Place the task folders from the Hugging Face snapshot directly under `TriGap/instructions/`; [trigap.json](configs/trigap.json) uses the original filenames. Download the images separately using the links on the dataset card.

Use the following pretrained checkpoints, listed by [MCITlib](https://github.com/Ghy0501/MCITlib#-models). The directories are relative to `SEARS_MODEL_ROOT`.

| Hugging Face repository | Local directory |
| --- | --- |
| [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main) | `llava-v1.5-7b/` |
| [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) | `clip-vit-large-patch14-336/` |
| [OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B/tree/main) | `Internvl-chat-7b/` |
| [OpenGVLab/InternViT-6B-224px](https://huggingface.co/OpenGVLab/InternViT-6B-224px/tree/main) | `InternViT-6B-224px/` |

Download the chat model files including `mm_projector.bin`. The InternVL profile reads the embedded 336-pixel visual weights from the InternVL-Chat checkpoint; the InternViT directory supplies its configuration and modeling files. To download the models after environment setup:

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir "$SEARS_MODEL_ROOT/llava-v1.5-7b"
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir "$SEARS_MODEL_ROOT/clip-vit-large-patch14-336"
huggingface-cli download OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B --local-dir "$SEARS_MODEL_ROOT/Internvl-chat-7b"
huggingface-cli download OpenGVLab/InternViT-6B-224px --include '*.py' 'config.json' --local-dir "$SEARS_MODEL_ROOT/InternViT-6B-224px"
```

## Environment

Use Python 3.10, PyTorch 2.0.1 with CUDA 11.8, Transformers 4.31.0, FlashAttention 2.6.3, and Java 11 for caption evaluation. Create and activate the SEARS Conda environment on a new server:

```bash
conda create -n SEARS python=3.10 -y
conda activate SEARS
```

If the SEARS environment already exists, run only `conda activate SEARS`. Install the CUDA 11.8 PyTorch build and a matching FlashAttention wheel before installing the remaining packages from `requirements.txt`.

## Training and evaluation

Run IDs: `ucit-llava`, `cvit-llava`, `coin-llava`, `trigap-llava`, `ucit-internvl`. All profiles use one GPU for training and evaluation.

```bash
bash scripts/server.sh ucit-llava check
bash scripts/server.sh ucit-llava train
bash scripts/server.sh ucit-llava eval
bash scripts/server.sh ucit-llava run
```

`check` validates the input files, model shards, and sampled image paths without starting training. `train` learns the tasks in order and saves one checkpoint per stage. `eval` loads each checkpoint, evaluates all tasks learned by that stage, and writes the task scores and continual-learning metrics to `metrics.json`. `run` executes both steps. Outputs go to `SEARS_OUTPUT_ROOT/<run-id>/`, or `runs/<run-id>/` when the output root is unset.

## Citation

```bibtex

```
