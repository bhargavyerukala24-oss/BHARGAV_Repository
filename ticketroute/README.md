# Project 3 — TicketRoute: Transformer Support-Ticket Classifier

A beginner-friendly project to classify a support ticket into one of four categories. It includes an interactive app, tokenization view, a pretrained zero-shot baseline, supervised fine-tuning and a separate evaluation script.

**Start here:** This ZIP contains the complete source, synthetic sample data and a Colab notebook. It does **not** include pretrained or fine-tuned model weights. The app downloads the baseline model on first use; you create the fine-tuned classifier by running `train.py`.

## 1. What the project does

| Category | Example |
| --- | --- |
| billing problem | I was charged twice. |
| account access problem | I cannot receive my login code. |
| technical error | The dashboard crashes when I upload a file. |
| feature request | Please add a dark mode. |

A **manual review** flag appears when the highest score or its gap to the second score is below the selected threshold. This is an illustrative triage rule; it cannot reliably detect every unfamiliar or out-of-scope ticket. Predictions and scores need human review before real-world use.

Two model modes:

1. **Zero-shot baseline** uses `typeform/distilbert-base-uncased-mnli`, already fine-tuned for natural language inference. It compares the ticket with label hypotheses. You do not train this baseline on the included tickets.
2. **Fine-tuned classifier** starts from the pretrained encoder `distilbert/distilbert-base-uncased` and trains a new four-class classification head plus encoder weights on the included training tickets. Run the training step first.

The zero-shot model and supervised model have different starting checkpoints and inference methods. This is a useful practical baseline comparison, not a controlled measurement of fine-tuning alone. A strict experiment would additionally compare appropriate models with matched starting weights and protocols.

There is **no paid OpenAI API key or subscription** for local inference or training. The pretrained model files and Python packages must be downloaded; availability, internet usage, electricity, and any optional cloud compute/hosting costs depend on your circumstances.

## 2. Folder contents

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and tokenizer display |
| `classifier.py` | Shared inference, score mapping and manual-review logic |
| `config.py` | Model IDs, fixed labels and sequence length |
| `data/train.csv` | 32 synthetic labeled training examples |
| `data/validation.csv` | 8 synthetic validation examples |
| `data/test.csv` | 8 separate synthetic test examples |
| `data/out_of_scope.csv` | 8 unsupported examples for review checks |
| `data_utils.py` | CSV loading, label and cross-split isolation checks |
| `create_sample_data.py` | Regenerates the bundled synthetic dataset |
| `train.py` | Fine-tunes DistilBERT and saves model + tokenizer |
| `evaluate.py` | Metrics, error analysis and review coverage |
| `inspect_tokenizer.py` | Small tokenizer demonstration |
| `Train_TicketRoute_Colab.ipynb` | Optional guided notebook for CPU/GPU training |
| `DATASET_CARD.md` | Dataset origin, limitations and recommended improvements |
| `VALIDATION.md` | Checks actually run and checks still required |
| `tests/test_project.py` | Offline checks with simulated model outputs |

## 3. Windows environment

Extract `TicketRoute_Project_3.zip`. Open Command Prompt **inside the extracted `ticketroute` folder**, where `app.py` is located.

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

The first command should print Python 3.11.x. Use the same environment when running the other commands. If the `py` launcher is missing, install Python 3.11 with the Windows launcher and reopen Command Prompt.

The project constrains Transformers to version 4.51.3 and Accelerate to 1.6.0 so the provided Trainer API stays consistent. The Torch range is not a complete lockfile. Use a CUDA-enabled Torch build that matches your NVIDIA driver if you want GPU acceleration; otherwise training runs on CPU and can take longer. On a working setup, record `python -m pip freeze > requirements-lock.txt` and your hardware details.

## 4. Inspect the data and run offline checks

```bat
python data_utils.py
python -m unittest discover -s tests -v
```

The validator checks required labels, empty rows, exact duplicate text and group IDs across splits. It does **not** detect every near-duplicate or prove that labels are correct. The unit tests use simulated outputs and do not download models.

The sample is intentionally tiny: 32 train / 8 validation / 8 test tickets. It is useful to learn the workflow, not to claim production accuracy. Collect and manually review a larger, representative dataset before interpreting model performance. Keep related messages and paraphrases in the same group and split, remove private data, inspect class balance, and keep your final test split untouched while tuning.

`create_sample_data.py` overwrites the bundled CSV files. Save a copy before running it if you edit them.

## 5. Run the pretrained baseline

```bat
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Choose **Zero-shot baseline**, enter:

```text
I was charged twice for my subscription.
```

Click **Classify ticket**. The first model load needs internet and can take time. The output is a suggested category, the four scores, the score gap, and possibly a manual-review notice. Expand the tokenizer section to see token strings and token IDs. Exact predictions and scores vary.

Try an unrelated question like “What is the weather tomorrow?” and inspect whether review is suggested. The classifier is constrained to four labels, so it may still confidently choose one of them for an out-of-scope input.

## 6. Tokenizers, embeddings and inference

```bat
python inspect_tokenizer.py
```

The tokenizer breaks text into model-specific token IDs, adds special tokens and creates an attention mask. IDs identify tokens; the model converts them into learned vector representations, processes them through Transformer layers, then produces classification scores.

The app's token display shows the ticket text alone. The zero-shot classifier internally considers the ticket together with each label hypothesis. The supervised classifier returns a score for each fixed class.

## 7. Fine-tune the model

Run locally in the project environment:

```bat
python train.py
```

What the script does:

1. Validates the data split and fixes a random seed.
2. Loads the pretrained DistilBERT tokenizer and encoder.
3. Adds a new four-class classification head with a stable ID-to-label mapping.
4. Tokenizes the train and validation examples, truncating long inputs at 256 tokens.
5. Uses dynamic batch padding and Hugging Face Trainer for three epochs.
6. Selects the checkpoint with the best validation macro-F1 and saves the model **and tokenizer** to `models/fine_tuned/`.

The new classification head starts untrained. Training all model weights on 32 synthetic examples is only a pipeline demonstration; it can overfit and may perform worse than the zero-shot baseline. CPU training is supported but can be slow; GPU memory and speed vary. No accuracy number is prefilled.

After training, start the Streamlit app and select **Fine-tuned classifier**. If the app is already open, restart it so it loads the new checkpoint. Re-running `train.py` can overwrite the output folder; copy a model you wish to preserve first.

### Optional: Google Colab notebook

1. Open `Train_TicketRoute_Colab.ipynb` in Google Colab.
2. Optionally select a GPU runtime; free GPU availability is not guaranteed.
3. Run cells in order and upload the original `TicketRoute_Project_3.zip` when requested.
4. The notebook installs packages, checks data, trains and evaluates, then downloads `TicketRoute_Trained_Artifacts.zip`.
5. Extract that artifact into the local `ticketroute` folder so `models/fine_tuned/config.json` exists.

If installation requires a runtime restart, rerun folder selection/setup cells. The runtime is temporary; download artifacts before ending the session.

## 8. Evaluate without leaking the test set

First use validation for development:

```bat
python evaluate.py --mode both --split validation
```

Adjust the app's threshold and margin based on your validation requirements. Then evaluate the held-out test set and held-out unsupported examples:

```bat
python evaluate.py --mode both --split test --threshold 0.65 --margin 0.15 --include-oos
```

Replace thresholds with the values you chose on validation. Reports are saved to `reports/`. The test report includes:

- Accuracy and macro-F1 over **all** labeled tickets, before excluding cases marked for review.
- Per-class precision, recall and F1, and confusion matrix in the listed label order.
- Auto-route coverage: fraction not flagged for review.
- Accuracy among auto-routed tickets; `null` if none were accepted.
- Out-of-scope manual-review rate when `--include-oos` is supplied, with each prediction saved for inspection.

These metrics answer different questions. A high review rate can hide poor model performance if you report only accepted-ticket accuracy. Scores are not calibrated probabilities; compare the models using the same data and settings, inspect wrong predictions and report the actual sample sizes. The eight out-of-scope examples are held out for a final check; if you repeatedly use them for tuning, collect a fresh out-of-scope test set.

The default `0.65` top score and `0.15` gap are teaching defaults, not measured optimal thresholds. Both methods can be confidently wrong. The app does not automatically send tickets to a team.

## 9. Deployment

### Baseline demo on Streamlit Community Cloud

1. Create a GitHub repository, for example `ticketroute-transformers`.
2. Upload the contents of `ticketroute/`, including `app.py`, `classifier.py`, `config.py`, `requirements.txt`, data, tests and README. `app.py` must be at the repository root. Do not upload `.venv`, `runs`, or `reports`.
3. Go to https://share.streamlit.io/ and create an app from your repository and branch, with `app.py` as the entry point. Select a compatible Python version in Advanced settings.
4. Deploy, wait for the first model download, then test the actual URL the host gives you.
5. First test **Zero-shot baseline**. The fine-tuned option requires the artifact you created in training.

### Add your fine-tuned model to the hosted app

The fine-tuned checkpoint can be large, so the `.gitignore` excludes `models/`. One option is to upload your own trained model folder to a Hugging Face model repository using a free account, subject to current provider terms and storage limits.

1. After training, check that `models/fine_tuned/` contains `config.json`, weights and tokenizer files. Review the upstream model license and the data you plan to share.
2. Create a public Hugging Face model repository under your account, for example `YOUR_USERNAME/ticketroute-model`.
3. In the activated project environment, install the Hub CLI if needed and sign in:

```bat
python -m pip install -U "huggingface_hub[cli]"
hf auth login
hf upload YOUR_USERNAME/ticketroute-model models/fine_tuned .
```

Do not print or share your Hugging Face token. Replace the placeholder with your own repository ID. Check that the model and tokenizer files really appear in the repo.

4. In Streamlit Community Cloud's Advanced settings/Secrets, add this **non-secret** root-level setting with your actual public repo ID:

```toml
TICKET_MODEL_ID = "YOUR_USERNAME/ticketroute-model"
```

The app reads `TICKET_MODEL_ID` as an environment variable. Root-level Streamlit secrets are exposed as environment variables. Restart/redeploy and select **Fine-tuned classifier**. Confirm the loaded label mapping and run a test ticket. A private/gated model needs authentication and additional configuration not supplied in this public demo.

The host must be able to download the model and have sufficient memory. If it cannot, use a suitable host or run locally; do not claim a live deployment until you test the actual URL. Downloading model files and optional hosting can consume resources, and service free tiers/limits can change.

## 10. Troubleshooting

- `No module named ...`: activate `.venv` and install `requirements-train.txt` with that environment's Python.
- Model download fails: check internet, the model ID, Hugging Face availability and disk space.
- CUDA unavailable: check your GPU, driver and CUDA-enabled Torch build, or train on CPU.
- Out of memory: reduce `per_device_train_batch_size` and/or `MAX_LENGTH` and rerun; re-evaluate after changes.
- Fine-tuned model missing: run training or extract the trained artifact under `models/fine_tuned/`.
- Labels show `LABEL_0`: your saved checkpoint must preserve the mapping in `config.py`.
- Hosted baseline works but fine-tuned fails: upload the model folder, configure the public repo ID and check server logs.
- Low quality: inspect examples and labels, increase representative training data, compare against the baseline and collect a fresh held-out set.

## 11. Interview explanation

“I built a support-ticket classifier using pretrained Transformers. I started with a zero-shot NLI baseline, then fine-tuned a DistilBERT sequence classifier on labeled examples. I display tokenization, compare both models on held-out data, and measure manual-review coverage and out-of-scope behavior. The bundled data is synthetic and too small for a production performance claim.”

## Official references

- DistilBERT NLI model card: https://huggingface.co/typeform/distilbert-base-uncased-mnli
- Base DistilBERT model card: https://huggingface.co/distilbert/distilbert-base-uncased
- Text classification with Trainer: https://huggingface.co/docs/transformers/v4.51.3/en/tasks/sequence_classification
- Hub upload: https://huggingface.co/docs/huggingface_hub/guides/cli
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Review the current model licenses and terms before redistributing weights or using a model commercially.
