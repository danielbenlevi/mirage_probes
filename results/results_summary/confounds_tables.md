### Naive Bayes Summary

| Model / Setting | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` | `all` |
|---|---:|---:|---:|---:|
| OVIS Contrastive | 61.88% | 57.67% | 66.67% | 64.85% (66.38%, 63.50%, 56.67%) |
| OVIS All Examples | 75.38% | 81.11% | 67.50% | 75.74% (76.22%, 75.29%, 71.78%) |
| QWEN3-VL-32B-Instruct Contrastive | 59.08% | 54.80% | 45.45% | 52.24% (59.43%, 51.89%, 50.00%) |
| QWEN3-VL-32B-Instruct All Examples | 77.08% | 81.50% | 79.64% | 72.80% (60.59%, 76.17%, 65.97%) |

### Cross-Benchmark Transfer (Single-Benchmark Train/Test)

| Model / Setting | `vqa→mmmu` | `vqa→medx` | `mmmu→vqa` | `mmmu→medx` | `medx→vqa` | `medx→mmmu` |
|---|---:|---:|---:|---:|---:|---:|
| OVIS Contrastive | 52.00% | 53.33% | 73.62% | 16.67% | 50.43% | 53.83% |
| OVIS All Examples | 54.66% | 49.47% | 53.10% | 50.00% | 52.48% | 52.05% |
| QWEN3-VL-32B-Instruct Contrastive | 51.26% | 46.87% | 41.84% | 49.48% | 54.29% | 51.36% |
| QWEN3-VL-32B-Instruct All Examples | 46.34% | 47.45% | 48.66% | 51.31% | 49.71% | 56.92% |

### Phrase-Group Class Separators

Cells show `class1_rate - class0_rate`.

#### OVIS Contrastive

| Phrase Group | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` |
|---|---:|---:|---:|
| `reasoning_scaffold` | -8.2% | +0.0% | -11.1% |
| `image_grounding` | -2.0% | +4.2% | +11.1% |
| `uncertainty` | -8.2% | -8.3% | -11.1% |
| `hedging` | -2.0% | -16.7% | +0.0% |
| `answer_boilerplate` | -12.2% | +4.2% | +0.0% |
| `radiology_terms` | +0.0% | +4.2% | -11.1% |

#### OVIS All Examples

| Phrase Group | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` |
|---|---:|---:|---:|
| `reasoning_scaffold` | -15.1% | -32.8% | -11.0% |
| `image_grounding` | +8.7% | +7.1% | +10.6% |
| `uncertainty` | -22.6% | -3.7% | -9.8% |
| `hedging` | -22.4% | -7.6% | -7.1% |
| `answer_boilerplate` | -6.4% | +11.7% | -0.1% |
| `radiology_terms` | -3.9% | +10.2% | -1.0% |

#### QWEN3-VL-32B-Instruct Contrastive

| Phrase Group | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` |
|---|---:|---:|---:|
| `reasoning_scaffold` | +0.0% | -1.9% | +5.9% |
| `image_grounding` | +1.8% | -0.6% | -4.0% |
| `uncertainty` | +0.9% | +0.0% | -1.0% |
| `hedging` | +3.6% | -4.5% | +1.0% |
| `answer_boilerplate` | +0.9% | +0.6% | -5.9% |
| `radiology_terms` | +0.0% | +2.5% | +2.0% |

#### QWEN3-VL-32B-Instruct All Examples

| Phrase Group | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` |
|---|---:|---:|---:|
| `reasoning_scaffold` | -4.0% | +3.6% | +7.8% |
| `image_grounding` | +3.0% | -3.8% | -32.0% |
| `uncertainty` | -10.1% | -0.1% | +0.8% |
| `hedging` | -14.3% | +8.4% | +4.9% |
| `answer_boilerplate` | +0.8% | +14.7% | -8.6% |
| `radiology_terms` | +0.0% | +2.6% | +0.8% |
