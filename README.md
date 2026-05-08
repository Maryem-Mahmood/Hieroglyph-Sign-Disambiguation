# 𓅄 Hieroglyph Recognition: From Isolated CNN to Context-Aware Transformer

**CS437 / CS5317 / EE414 / EE513, Deep Learning, Spring 2026**

**Authors:** Maryem Mahmood (27100407), Zehra Talat (27100432)

---

## Overview

This project builds a progressive pipeline for Egyptian hieroglyph recognition, advancing from isolated image classification to context-aware sequence labeling. Across five phases, we demonstrate that sequential context (whether from statistical n-gram models or learned transformer attention) substantially improves recognition accuracy on the 176-class Glyph2025 dataset.

| Phase | Method | Per-pos Acc | Macro F1 |
|---|---|---|---|
| Phase 1 | ResNet-18 (isolated) | 67.08% | 62.83% |
| Phase 2 | + Trigram re-ranking (α=1.0) | 89.21% | 79.33% |
| Phase 3 | + Transformer encoder | 73.51% | 49.90% |
| D4 Enhanced | + LLRD fine-tuning | 74.01% | 48.10% |

**Key finding:** Progressive fine-tuning with layer-wise learning rate decay (LLRD) yields a +1.19% accuracy and +1.53% F1 improvement over naïve fine-tuning, with gains concentrated on common classes (+3.53% F1) and specific class rescues (W18, G4, D2, P1, G36 recovered from F1=0).

---

## Dataset: Glyph2025

A merged dataset we constructed from two public sources:

- **EHT (Fuentes-Ferrer):** 9,703 images, 310 classes, diverse archaeological sources (stone, papyrus, drawings)
- **Franken/OBC306:** 4,210 images, 171 classes, manually annotated from the Pyramid of Unas with reading-order metadata

Combined: 13,287 images, 352 classes (176 after filtering ≥7 samples). Stratified 70/15/15 split.

The Franken subset provides sequential structure (plate number + glyph index encoded in filenames) used for Phases 2 and 3.

---

<details>
<summary><h3>I) Repository Structure</h3></summary>

```Hieroglyph-Sign-Disambiguation/
│
├── notebooks/                         
│   ├── task_3_phase_1.ipynb           
│   ├── task_3_phase_2.ipynb           
│   ├── task_3_phase_3.ipynb          
│   ├── task_4.ipynb                   
│   └── task_5.ipynb                   
│
├── plots/                            
│   ├── task_3_phase_1/                
│   ├── task_3_phase_2/                
│   ├── task_3_phase_3/               
│   ├── task_4/                        
│   └── task_5/                       
│
├── results/                           
│   ├── task_3_phase_1/                
│   ├── task_3_phase_2/                
│   ├── task_3_phase_3/               
│   ├── task_4/                        
│   └── task_5/                        
│
└── paper/                             
├── 41_27100432_27100407.pdf       
└── .gitkeep

</details>

<details>
<summary><h3>II) Phase Descriptions</h3></summary>

**Phase 1: CNN Baseline.** ResNet-18 pretrained on ImageNet, fine-tuned on Glyph2025 with two-stage progressive unfreezing (head-only then Layer 4). Handles class imbalance via WeightedRandomSampler and weighted CrossEntropyLoss. Also includes Glyphnet (82.66%) and ConvNeXt-Tiny (56.23%) comparisons.

**Phase 2: CNN + N-gram Re-ranking.** Frozen ResNet-18 provides top-K=10 candidates per position. Trigram language model (from nGrams.txt, 158 Pyramid Texts) re-ranks candidates using P(sign | 2 previous signs). No training, purely inference-time combination of visual and linguistic evidence.

**Phase 3: CNN + Transformer Encoder.** Frozen ResNet-18 features pass through a Linear projection (512 to 256) plus learned positional encoding, then a 3-layer transformer encoder (4 heads), then a per-position classifier. Trained on 52 Franken plate sequences with reading order.

**Deliverable 4: Progressive Fine-tuning.** Two attempts: (1) Original, naive progressive unfreezing, +0.20% improvement. (2) Enhanced, LLRD with per-layer learning rates, stronger regularisation, +1.19% accuracy and +1.53% F1 with comprehensive diagnostic analysis.

</details>

<details>
<summary><h3>III) Kaggle Outputs and Model Checkpoints</h3></summary>

Trained model checkpoints and supporting files are hosted as Kaggle datasets (not committed to GitHub due to file size limits). Each dataset is a self-contained bundle to be added as an input to a Kaggle notebook.

| Resource | Kaggle Dataset | Description |
|---|---|---|
| Phase 1 outputs | [phase1outputs](https://www.kaggle.com/datasets/maryemmahmood/phase1outputs) | `resnet18_best.pth`, split CSVs, results JSON, confusion matrix |
| Phase 2 outputs | [phase2outputs](https://www.kaggle.com/datasets/maryemmahmood/phase2outputs) | n-gram re-ranking results, alpha tuning plots |
| Phase 3 outputs | [phase3outputs](https://www.kaggle.com/datasets/maryemmahmood/phase3outputs) | `transformer_best.pth`, label maps, results JSON, training curves |
| Original D4 outputs | [phase4outputs](https://www.kaggle.com/datasets/maryemmahmood/phase4outputs) | `transformer_finetuned_best.pth`, D4 results JSON |
| Enhanced D4 outputs | [task-5-outputs](https://www.kaggle.com/datasets/maryemmahmood/task-5-outputs) | Enhanced D4 transformer weights, comprehensive diagnostic plots, per-class metrics |

**To reproduce on Kaggle:**

1. Create a new Kaggle notebook
2. Click "+ Add Input" and add the relevant datasets above (plus the Glyph2025 image dataset)
3. Upload the desired notebook from this repo's `notebooks/` folder
4. Adjust the path block in Section 1 if your dataset slugs differ
5. Use a GPU runtime (P100 or T4 x2) for reasonable training time

</details>

<details>
<summary><h3>IV) Setup and Reproduction</h3></summary>

**Requirements:**

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow
```

**Data sources:**

1. Glyph2025: Download EHT from GitHub and Franken from HuggingFace. Merge using `notebooks/deliverable2_dataset.ipynb`.
2. N-gram data: Download `nGrams.txt` and `Lexicon.txt` from the HuggingFace mirror.

**Training order (run on Kaggle, T4 GPU recommended):**
deliverable2_dataset.ipynb
↓
phase1_cnn_baseline.ipynb
↓
phase2_cnn_ngram.ipynb
↓
phase3_cnn_transformer.ipynb
↓
deliverable4_enhanced.ipynb

Each notebook saves checkpoints and results JSON files consumed by subsequent phases.

</details>

<details>
<summary><h3>V) Key Findings</h3></summary>

1. **Architectural switch from ResNet to CNN+Transformer is strongly justified.** The transformer cuts ResNet's Glyph Error Rate roughly in half (52% to 26%), wins 14 of 16 test sequences in head-to-head comparison, and produces dramatic gains on multi-glyph BLEU (BLEU-4: 30% vs 6%).

2. **Enhanced D4 fine-tuning recipe gives modest but real gains over the original D4** when evaluated on identical test images: +1.19% per-position accuracy and +1.53% macro F1, consistent across BLEU-1 to BLEU-4.

3. **Strict sequence accuracy is a length artefact.** Reported as 0% across all models on long test sequences. Per-position accuracy stays roughly flat across length bins while strict sequence accuracy collapses geometrically as p^L.

4. **Rare-class story is more nuanced than expected.** Classes appearing fewer than 3 times in training are essentially unlearnable for any model; the transformer's gains are concentrated on common-class disambiguation rather than rare-class rescue.

</details>

---

## References

- Barucci, A., et al. (2021). "A Deep Learning Approach to Ancient Egyptian Hieroglyphs Classification." IEEE Access.
- Franken, M. and van Gemert, J. (2013). "Automatic Egyptian Hieroglyph Recognition by Retrieving Images as Texts." ACM Multimedia.
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
- Li, M., et al. (2023). "TrOCR: Transformer-based OCR with Pre-trained Models." AAAI.
- Howard, J. and Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL.

---

## License

This project is for academic use as part of the CS437/EE414 Deep Learning course at LUMS, Spring 2026.
