---
title: TutorTune LoRA Demo
emoji: "T"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# TutorTune — Project 2: Fine-tune an LLM

A complete source project for adapting **Qwen/Qwen2.5-0.5B-Instruct** with **LoRA** to produce short Python lessons with `Definition:`, `Example:`, and `Practice:` headings.

**Start here:** train the adapter using `Train_TutorTune_Colab.ipynb`, download it, then run the local comparison app. The ZIP includes source and sample data; model weights are downloaded during execution and trained adapter weights are created by your own training run.

## 1. Extract the project

Extract `TutorTune_Project_2.zip`. Open the `tutortune` folder containing `app.py`, `train.py`, and this README. Do not run commands from the parent directory.

## 2. Windows environment

Install Python 3.11, then open **Command Prompt** inside the project folder:

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

For a later session, enter this folder and activate the existing environment again.

Prepare and validate data without installing model libraries:

```bat
python validate_data.py
```

Expected: 24 training, 6 validation, and 6 test rows. They are synthetic examples for a pipeline smoke check, not a benchmark or a sufficient basis for quality claims.

`make_data.py` regenerates those samples. It **overwrites** the three JSONL files, so do not run it over a dataset you have edited without saving a copy.

## 3. Recommended training route: Google Colab GPU

1. Open Google Colab in your own account.
2. Upload/open `Train_TutorTune_Colab.ipynb`.
3. Select a GPU hardware runtime in the runtime settings. A suitable GPU is not guaranteed on a free account.
4. Run the cells in order. The upload cell asks for the **original project ZIP**; upload `TutorTune_Project_2.zip`.
5. The notebook extracts the project and installs `requirements-train.txt`.
6. Verify that CUDA is available. A GPU around 16 GB VRAM is a practical starting target for this small model; actual memory use varies.
7. Validate the included data, run training, and run the base-versus-adapter evaluation.
8. Download the generated artifact ZIP before closing the notebook runtime.

The code intentionally uses a small non-quantized float32 base with LoRA. It is not QLoRA. Training is blocked when CUDA is unavailable; CPU inference is supported later.

If package installation requests a runtime restart, restart and rerun the folder-selection/setup cells before continuing. If the runtime already has a compatible CUDA-enabled Torch build, keep that working build. Do not replace Torch randomly when debugging a driver issue.

## 4. Alternative: local NVIDIA GPU training

From the activated environment:

```bat
python -m pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python validate_data.py
python train.py
python evaluate.py
```

Your CUDA-enabled Torch installation must match the GPU/driver environment. If CUDA prints `False`, use the notebook route or fix the actual GPU installation before training.

## 5. Understand what training creates

- `runs/`: intermediate checkpoints and training logs.
- `adapter/adapter_config.json`: LoRA/base configuration.
- `adapter/adapter_model.safetensors`: trained adapter parameters.
- `adapter/`: tokenizer files and `training_metadata.json` with the resolved base revision.
- `evaluation_results.json`: base and adapter answers from `evaluate.py`.

The adapter is **not** a complete standalone model. Keep the entire adapter folder together. Training uses assistant-only loss: prompt and padding tokens are masked with `-100`. The exact token prefix is checked rather than assumed. Overlong examples are rejected instead of silently truncated.

## 6. Bring the trained artifact back to Windows

The notebook downloads `TutorTune_Trained_Artifacts.zip`. Extract it **inside this project folder**. Confirm this layout:

```text
tutortune/
  app.py
  inference.py
  config.py
  adapter/
    adapter_config.json
    adapter_model.safetensors
    training_metadata.json
    ...tokenizer files...
```

Avoid accidental nesting such as `tutortune/adapter/adapter/`.

## 7. Run the comparison app

In the activated Windows environment:

```bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Compare **Base model** and **LoRA adapter** using the same question. The first request downloads the base model. CPU inference can be slow; use a sufficiently provisioned host for demonstrations.

No OpenAI API key is required for this project. The named public base model normally needs no access token. If you change to private/gated artifacts, configure access through your host's secrets facility.

## 8. Evaluate honestly

```bat
python evaluate.py
```

Both modes use the same base revision, system prompt, token limit and greedy generation. Base mode disables the adapter. The automatic metric checks only the presence/order of headings; it does **not** measure factual correctness or teaching quality.

Manually review answers and safely run relevant code examples separately. Expand the dataset with reviewed, diverse examples; group related examples before splitting; keep test data independent of tuning. A tiny synthetic dataset can leave quality unchanged or make it worse. Report measured results, never an invented improvement percentage.

## 9. Deploy to a Hugging Face Docker Space

1. Complete training and evaluation first.
2. Create a Hugging Face Space and select **Docker**.
3. Upload `app.py`, `inference.py`, `config.py`, `requirements.txt`, `Dockerfile`, this `README.md`, and the **complete adapter folder**.
4. Keep these files at the Space repository root. No training job runs when visitors open the app.
5. Select hardware with adequate RAM. CPU can be slow; choose supported GPU hardware if latency requires it.
6. Wait for the build and inspect logs. The Dockerfile starts Streamlit on `0.0.0.0:7860`, matching the README metadata.
7. Test both base and adapter modes at the actual URL supplied by the host. Add that verified URL to your portfolio only after it works.

Do not upload `.venv`, base-model caches or `runs/`. Adapter files must be present before inference can work. Remote hosting, model downloads and compute may incur costs depending on your accounts and choices. Do not expose private datasets or credentials in a public Space.

## 10. Files

| File | Purpose |
|---|---|
| `Train_TutorTune_Colab.ipynb` | Guided GPU training, evaluation and artifact download |
| `config.py` | Base model, system prompt and sequence length |
| `make_data.py` | Recreates the synthetic sample dataset |
| `validate_data.py` | Offline dataset checks |
| `data/*.jsonl` | Included topic-separated sample data |
| `train.py` | LoRA training and adapter saving |
| `inference.py` | Exact-base adapter loading and generation |
| `evaluate.py` | Base/adapter comparison and format metric |
| `app.py` | Interactive Streamlit comparison |
| `requirements-train.txt` | Training dependencies |
| `requirements.txt` | Inference dependencies |
| `Dockerfile` | Hosted inference container |
| `DATASET_CARD.md` | Dataset origin, scope and limits |
| `VALIDATION.md` | Checks performed and checks still required |

## Troubleshooting

- **Missing module:** activate the right environment and install through `python -m pip`.
- **No adapter:** train first; extract the full artifact to the correct folder.
- **CUDA unavailable:** confirm the GPU runtime and compatible Torch installation.
- **Out of memory:** reduce sequence length, check stale GPU processes, or choose suitable hardware. Reevaluate after changing the setup.
- **Prefix mismatch:** inspect chat-template token boundaries; do not remove the guard.
- **Model download failure:** check network/model access and available disk space.
- **Different behavior after deployment:** verify the same base revision, tokenizer, adapter and package versions.

The APIs are deliberately pinned to Transformers 4.51.3 and PEFT 0.15.2. This is a source baseline, not a claim that every environment combination is validated. After a working run, save `python -m pip freeze` and record hardware and data versions.

## Sources and model license

- Model card and license: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- LoRA documentation: https://huggingface.co/docs/peft/v0.15.0/package_reference/lora
- Chat templates: https://huggingface.co/docs/transformers/v4.51.3/chat_templating
- Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker

The base model card lists Apache 2.0. Preserve applicable model notices and review the current upstream license when redistributing model artifacts.
