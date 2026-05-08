# 𓅄 Hieroglyph Recognition: From Isolated CNN to Context-Aware Transformer

**CS437 / CS5317 / EE414 / EE513 — Deep Learning, Spring 2026**

**Authors:** Maryem Mahmood (27100407) · Zehra Talat (27100432)

---

## Overview

This project builds a progressive pipeline for Egyptian hieroglyph recognition, advancing from isolated image classification to context-aware sequence labeling. Across five phases, we demonstrate that sequential context — whether from statistical n-gram models or learned transformer attention — substantially improves recognition accuracy on the 176-class Glyph2025 dataset.

| Phase | Method | Per-pos Acc | Macro F1 |
|-------|--------|------------|----------|
| Phase 1 | ResNet-18 (isolated) | 67.08% | 62.83% |
| Phase 2 | + Trigram re-ranking (α=1.0) | 89.21% | 79.33% |
| Phase 3 | + Transformer encoder | 73.51% | 49.90% |
| D4 Enhanced | + LLRD fine-tuning | 74.01% | 48.10% |

> **Key finding:** Progressive fine-tuning with layer-wise learning rate decay (LLRD) yields a +1.19% accuracy and +1.53% F1 improvement over naïve fine-tuning, with gains concentrated on common classes (+3.53% F1) and specific class rescues (W18, G4, D2, P1, G36 recovered from F1=0).

---

## Dataset: Glyph2025

A merged dataset we constructed from two public sources:

- **EHT (Fuentes-Ferrer):** 9,703 images, 310 classes — diverse archaeological sources (stone, papyrus, drawings)
- **Franken/OBC306:** 4,210 images, 171 classes — manually annotated from the Pyramid of Unas with reading-order metadata

**Combined:** 13,287 images, 352 classes (176 after filtering ≥7 samples). Stratified 70/15/15 split.

The Franken subset provides sequential structure (plate number + glyph index encoded in filenames) used for Phases 2–3.

---

## Repository Structure

```
hieroglyph-recognition/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
│
├── notebooks/                         # All experiments and deliverables
│   ├── deliverable2_dataset.ipynb     # Dataset exploration, merging, enrichment
│   ├── phase1_cnn_baseline.ipynb      # CNN baseline experiments (ResNet-18, comparisons)
│   ├── phase2_cnn_ngram.ipynb         # N-gram re-ranking pipeline
│   ├── phase3_cnn_transformer.ipynb   # Transformer-based sequence modeling
│   ├── deliverable4_original.ipynb    # Initial fine-tuning attempt
│   └── deliverable4_enhanced.ipynb    # LLRD fine-tuning + diagnostics
│
├── src/                               # Core implementation code
│   ├── models/
│   │   ├── resnet_classifier.py
│   │   └── transformer.py
│   │
│   ├── data/
│   │   ├── glyph_dataset.py
│   │   ├── sequence_dataset.py
│   │   └── live_sequence_dataset.py
│   │
│   ├── ngram/
│   │   └── ngram_model.py
│   │
│   └── utils/
│       ├── transforms.py
│       ├── splits.py
│       └── metrics.py
│
├── data/                              # Dataset files (added in repo)
│   └── ...                            # raw + processed datasets
│
├── results/                           # Task/phase outputs
│   ├── phase1_results.json
│   ├── phase2_results.json
│   ├── phase3_results.json
│   ├── d4_results.json
│   └── d4e_results.json
│
├── plots/                             # Visualization outputs (Task 5 + others)
│   ├── phase1_training_curves.png
│   ├── phase1_confusion_matrix.png
│   ├── phase2_alpha_tuning.png
│   ├── phase3_training_curves.png
│   ├── d4e_training_curves.png
│   ├── d4e_per_class_f1.png
│   ├── d4e_bleu.png
│   └── d4e_summary_dashboard.png
│
├── checkpoints/                       # Model weights (not tracked / .gitkeep)
│   ├── resnet18_best.pth
│   ├── transformer_best.pth
│   └── transformer_finetuned_best_v2.pth
│
├── paper/                             # Research report (LaTeX)
│   ├── SOA survery
│   └── Report/
│
└── .gitignore
```

---

## Phase Descriptions

### Phase 1: CNN Baseline
ResNet-18 pretrained on ImageNet, fine-tuned on Glyph2025 with two-stage progressive unfreezing (head-only → Layer 4). Handles class imbalance via WeightedRandomSampler + weighted CrossEntropyLoss. Also includes Glyphnet (82.66%) and ConvNeXt-Tiny (56.23%) comparisons.

### Phase 2: CNN + N-gram Re-ranking
Frozen ResNet-18 provides top-K=10 candidates per position. Trigram language model (from `nGrams.txt`, 158 Pyramid Texts) re-ranks candidates using P(sign | 2 previous signs). No training — purely inference-time combination of visual and linguistic evidence.

### Phase 3: CNN + Transformer Encoder
Frozen ResNet-18 features → Linear projection (512→256) + learned positional encoding → 3-layer transformer encoder (4 heads) → per-position classifier. Trained on 52 Franken plate sequences with reading order.

### Deliverable 4: Progressive Fine-tuning
Two attempts: (1) Original — naive progressive unfreezing, +0.20% improvement. (2) Enhanced — LLRD with per-layer learning rates, stronger regularisation, +1.19% accuracy and +1.53% F1 with comprehensive diagnostic analysis.

---

## Outputs:

Phase 1: https://www.kaggle.com/datasets/maryemmahmood/phase1outputs

Phase 2: https://www.kaggle.com/datasets/maryemmahmood/phase2outputs

Phase 3: https://www.kaggle.com/datasets/maryemmahmood/phase3outputs

Phase 4: https://www.kaggle.com/datasets/maryemmahmood/phase4outputs

Task 5: https://www.kaggle.com/datasets/maryemmahmood/task-5-outputs


## Setup & Reproduction

### Requirements
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow
```

### Data
1. **Glyph2025:** Download EHT from [GitHub](https://github.com/rfuentesfe/EgyptianHieroglyphicText) and Franken from [HuggingFace](https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs). Merge using `notebooks/deliverable2_dataset.ipynb`.
2. **N-gram data:** Download `nGrams.txt` and `Lexicon.txt` from the [HuggingFace mirror](https://huggingface.co/datasets/HamdiJr/Egyptian_hieroglyphs/tree/main/LanguageModel).

### Training
Run notebooks in order on Kaggle (T4 GPU recommended):
```
deliverable2_dataset.ipynb → phase1_cnn_baseline.ipynb → phase2_cnn_ngram.ipynb → phase3_cnn_transformer.ipynb → deliverable4_enhanced.ipynb
```

Each notebook saves checkpoints and results JSON files consumed by subsequent phases.

---

## References

1. Barucci, A., et al. (2021). "A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification." *IEEE Access*.
2. Franken, M. & van Gemert, J. (2013). "Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts." *ACM Multimedia*.
3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
4. Li, M., et al. (2023). "TrOCR: Transformer-based OCR with Pre-trained Models." *AAAI*.
5. Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.

---

## License

This project is for academic use as part of the CS437/EE414 Deep Learning course at LUMS, Spring 2026.
