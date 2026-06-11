# Mirage Probes: How VLMs Fake Visual Understanding

This is the official code repository for the paper "**Mirage Probes: How Vision Models Fake Visual Understanding**".

## Overview

We study whether or not mirage behavior in VLMs in linearly represented in image-present latent space. We find that:
1. Mirage behavior is **linearly decodable** from image-present activations across benchmarks and models.
2. Our contrastive *Mirage Probes* framework effectively **reduces surface-level confounds**.
3. Mirage behavior seems to be enabled by **two distinct underlying mechanisms**: *textual biases* and *spurious images*.
4. **Text-only mitigations are likely insufficient** for prevent mirage behavior in VLMs.

## Setup Instructions

In order to run the code in this repository:
1. Install Git LFS **before** cloning.
2. Build a conda environment with the provided environment.yml file.
3. Run the dataset downloaders from this repo root to populate the local `raw_data` folder:
```bash
python scripts/data/data_downloaders/download_vqa_rad.py
python scripts/data/data_downloaders/download_microvqa.py
python scripts/data/data_downloaders/download_mmmu_pro.py
python scripts/data/data_downloaders/download_medxpertqa_mm.py
```
4. Build the local parquet datasets:
```bash
python scripts/data/build_datasets.py
```
5. (Optional) Set up a **separate** environment for starting vLLM servers. Start a server with something like:
```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve AIDC-AI/Ovis2.5-2B \
  --served-model-name ovis2_5_2b \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8001 \
  --generation-config vllm \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=8,video=0 \
  --trust-remote-code
```

## Running Scripts

These sections describe the workflows for generating data, extracting activations, and running the probe suite.

### 1) Mutation and Response Generation

*Note: If you only want to use the existing probe training datasets provided in this repository, you can skip this step.*

This workflow is driven by:
- `scripts/data/gen_mutations_get_responses.py`
- `scripts/data/build_contrastive_pairs.py`

The generation script writes a model-scoped directory under `tmp_artifacts/` by default and produces:
- `mutations.json` (generated question rewrites for each source question)
- `responses.json` (w/ and w/o image VLM responses plus mirage annotations for every variant)
- `mirage_flip_questions.json` (grouped question families where at least one variant flips the mirage label)
- per-dataset subdirectories with the same artifacts restricted to a single benchmark dataset

The contrastive-pair builder consumes `mirage_flip_questions.json` and writes a `contrastive_conversation_pairs.json` artifact containing paired mirage and non-mirage conversations for training.

#### Standard Generation Command

```bash
python scripts/data/gen_mutations_get_responses.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash}
```

By default, the script:
- uses all four datasets
- uses `4` mutations per question
- auto-detects and reuses a compatible `mutations.json` if one already exists

#### Reusing an Existing Mutations Artifact Explicitly

If you want to force reuse of a different known mutations file:

```bash
python scripts/data/gen_mutations_get_responses.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash} \
  --reuse_mutations_path {path_to_desired_mutations_json}
```

#### Running Without an Existing Mutations Artifact

If no compatible `mutations.json` is discoverable, the script generates mutations from scratch automatically. To guarantee fresh mutation generation, ensure no compatible `mutations.json` is visible in the searched locations. Fresh mutation generation requires OpenAI access for the mutator model, so ensure that your `OPENAI_API_KEY` environment variable is set.

#### Using Additional Flags

```bash
python scripts/data/gen_mutations_get_responses.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash} \
  --datasets {comma_separated_list_of_desired_dataset_names} \
  --max_questions_per_dataset {desired_max_questions} \
  --num_mutations {desired_num_mutations} \
  --save_path {desired_save_path} \
  --vllm_base_url {desired_vllm_base_url} \
  --vllm_model_override {desired_vllm_model}
```

#### Building Contrastive Pairs

Standard builder command:

```bash
python scripts/data/build_contrastive_pairs.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash}
```

This looks for `mirage_flip_questions.json` under the standard model-scoped artifact locations and writes the standard contrastive-pairs artifact.

Custom builder command:

```bash
python scripts/data/build_contrastive_pairs.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash} \
  --mirage_flip_path {path_to_desired_mirage_flip_questions_json} \
  --output_path {desired_contrastive_pairs_output_path}
```

### 2) Activation Extraction

This workflow is driven by:
- `scripts/data/activation_extraction/extract_qwen_activations.py`
- `scripts/data/activation_extraction/extract_glm_activations.py`

These scripts write pre-extracted activation caches used by the Qwen and GLM probe trainers. Ovis does not require a separate pre-extraction step for the standard orchestrated runs. If desired, GLM activation extraction requires the completion of Step 1.

#### Qwen Extraction Using Provided Final Artifacts

```bash
python scripts/data/activation_extraction/extract_qwen_activations.py
```

By default, this reads:
- `data/final_data/qwen_contrastive.json`
- `data/final_data/qwen_all_responses.json`

and writes caches to:
- `tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_contrastive_features.pt`
- `tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_all_examples_features.pt`

#### Qwen Additional-Target Caches

For attention/MLP additional-target runs:

```bash
python scripts/data/activation_extraction/extract_qwen_activations.py \
  --extract_additional_feature_caches
```

#### Qwen Extraction From a Custom `tmp_artifacts` Target

If you generated new Qwen artifacts and want caches tied to those artifacts instead of `data/final_data/`:

```bash
python scripts/data/activation_extraction/extract_qwen_activations.py \
  --pairs_path {path_to_contrastive_conversation_pairs_json} \
  --responses_path {path_to_responses_json} \
  --contrastive_output_path {contrastive_cache_output_path} \
  --all_examples_output_path {all_examples_cache_output_path} \
  --overwrite
```

### 3) Running the full probe experiment

This workflow is driven by:
- `run_full_probe_experiment.py`

The orchestrator runs the complete probe suite:
- `logreg_contrastive`
- `logreg_all_examples`
- `mlp_contrastive`
- `mlp_all_examples`
- `concat_contrastive`
- `concat_all_examples`
- `diff_contrastive`
- `diff_all_examples`

and then runs:
- `scripts/analysis/analyze_full_probe_results.py`

#### Standard Residual Run

```bash
python run_full_probe_experiment.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash} \
  --execute \
  --gpus {desired_gpus}
```

Remove the "execute" flag for a dry run. Note that Ovis does not require a pre-extracted cache.

#### Standard Additional-Targets Run (Attention/MLP)

```bash
python run_full_probe_experiment.py \
  --vlm {ovis/qwen3_vl_32b_instruct/glm_4_6v_flash} \
  --execute \
  --gpus {desired_gpus} \
  --extra_args "--include_attention_probes --include_mlp_probes --no_include_residual_probes --llm_feature_strategies text_nonspecial_mean"
```

#### Using Canonical Final_Data Inputs

For Ovis and Qwen, the orchestrated runs default to the canonical final artifacts:
- `data/final_data/<vlm>_all_responses.json`
- `data/final_data/<vlm>_contrastive.json`

#### Using Custom Artifacts In `tmp_artifacts`

If you want the full probe run to use a custom all_response or contrastive_pairs target, regenerate any required pre-extracted caches from those same artifacts first, then pass stage-specific overrides via `--stage_extra_args_json`.

Example override file:

```json
{
  "logreg_contrastive": [
    "--pairs_path", "{desired_contrastive_conversation_pairs_json}",
    "--responses_path", "{desired_responses.json}"
  ],
  "mlp_contrastive": [
    "--pairs_path", "{desired_contrastive_conversation_pairs_json}",
    "--responses_path", "{desired_responses.json}"
  ],
  "concat_contrastive": [
    "--pairs_path", "{desired_contrastive_conversation_pairs_json}",
    "--responses_path", "{desired_responses.json}"
  ],
  "diff_contrastive": [
    "--pairs_path", "{desired_contrastive_conversation_pairs_json}",
    "--responses_path", "{desired_responses.json}"
  ],
  "logreg_all_examples": [
    "--responses_path", "{desired_responses.json}",
    "--contrastive_pairs_path", "{desired_contrastive_conversation_pairs_json}"
  ],
  "mlp_all_examples": [
    "--responses_path", "{desired_responses.json}",
    "--contrastive_pairs_path", "{desired_contrastive_conversation_pairs_json}"
  ],
  "concat_all_examples": [
    "--responses_path", "{desired_responses.json}",
    "--contrastive_pairs_path", "{desired_contrastive_conversation_pairs_json}"
  ],
  "diff_all_examples": [
    "--responses_path", "{desired_responses.json}",
    "--contrastive_pairs_path", "{desired_contrastive_conversation_pairs_json}"
  ]
}
```

#### Output Structure

Each orchestrated run writes a root directory:

`results/results_final/<run_name_prefix>_<vlm>_<timestamp>`

Key outputs:
- `plan.json`: full planned command list with one job entry per stage/benchmark run
- `summary.json`: overall run status and per-stage return codes
- `analysis/top_probe_summary.json`: aggregated best-probe summary consumed by downstream tables/plots
- `analysis/plots/`: layer-wise plots produced by the analysis stage
- per-stage directories with logs and trainer outputs such as per-feature accuracies, heldout summaries, and run configs

## Project Structure

```text
mirage_probes/
├── scripts/                         # Scripts for data prep, training, analysis, and model setup
│   ├── analysis/                    # Table generation, confound analysis, PHI analysis, result summarization
│   ├── data/                        # Data-generation pipeline and local helper code
│   │   ├── activation_extraction/   # Qwen/GLM activation pre-extraction scripts
│   │   ├── data_downloaders/        # Dataset downloaders for VQA-RAD, MicroVQA, MMMU-Pro, MedXpertQA
│   │   └── data_helpers/            # Dataset loaders/converters and model client utilities
│   ├── model_downloaders/           # Model download helpers
│   └── training/                    # Probe trainers for log reg, mlp, concat, diff
├── data/                            # Permanent data artifacts
│   └── final_data/                  # Final Ovis/Qwen response, contrastive-pair, and mutation artifacts
├── raw_data/                        # User-downloaded benchmark source data and images
├── tmp_artifacts/                   # Intermediate artifacts, caches, generated responses, and probe outputs
├── results/                         # Final experiment directories and results summaries
│   ├── results_final/               # Saved orchestrated runs and analysis products
│   └── results_summary/             # Plot/table/postprocessing scripts for final results
├── models/                          # Local model checkpoints or cache directories when used
└── run_full_probe_experiment.py     # Main orchestrator for the full probe suite
```