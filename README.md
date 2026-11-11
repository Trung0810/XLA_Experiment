# OmniDiff Research Project

## Overview

This research project aims to **reproduce and improve upon** the OmniDiff paper:

**Paper**: OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning  
**Conference**: ICCV 2025  
**Authors**: Yuan Liu, Saihui Hou, Saijie Hou, Jiabao Du, Shibei Meng, Yongzhen Huang  
**Affiliation**: Beijing Normal University, Beijing University of Posts and Telecommunications, WATRIX.AI

## Research Task

**Image Difference Captioning (IDC)**: Generate natural language descriptions of subtle differences between pairs of similar images.

## Project Goals

1. ✅ **Phase 1**: Comprehensive paper analysis (COMPLETED)
2. ✅ **Phase 2**: Environment setup & reproduction planning (COMPLETED)
3. ⏳ **Phase 3**: Baseline implementation & reproduction (READY TO START)
4. ⏳ **Phase 4**: Research & improvement proposals
5. ⏳ **Phase 5**: Implementation & experimentation
6. ⏳ **Phase 6**: Evaluation & scientific analysis
7. ⏳ **Phase 7**: Final research report & deliverables

## Key Contributions of Original Paper

### 1. OmniDiff Dataset

- **15,598 image pairs** from 324 diverse scenarios
- **60 words average caption length** (vs. 8-32 in existing datasets)
- **12 distinct change types**: viewpoint, illumination, addition, disappearance, removal, substitution, size, color, orientation, pose, OCR, counting
- **Balanced data**: 55% real-world + 45% 3D synthetic
- **Train:Val:Test = 80%:10%:10%**

### 2. M³Diff Model

**M³Diff** (Multi Modal large language model enhanced by Multi-scale Differential perception):

- Extends standard MLLM with **Multi-scale Differential Perception (MDP) module**
- **Plug-and-play design** for easy integration
- **Channel-wise subtraction** for difference computation
- **Cross-attention fusion** for coherent representations
- **LoRA-based fine-tuning** for efficiency

### 3. State-of-the-Art Results

Achieves SOTA performance across 5 benchmarks:
- OmniDiff (new)
- Spot-the-Diff
- IEdit
- CLEVR-Change
- CLEVR-DC

## Project Structure

```
XLA_Experiment/
├── papers/                    # Research papers
│   └── 21. Liu_OmniDiff...pdf
├── research/                  # Research documentation
│   ├── Phase1.md             # ✅ Paper analysis (COMPLETED)
│   ├── Phase2.md             # ⏳ Reproduction planning
│   ├── Phase3.md             # ⏳ Baseline implementation
│   ├── Phase4.md             # ⏳ Improvement research
│   ├── Phase5.md             # ⏳ Experimentation
│   ├── Phase6.md             # ⏳ Evaluation
│   ├── Phase7.md             # ⏳ Final report
│   ├── HANDOFF.md            # Cross-model continuity
│   └── WORK_LOG.md           # Activity log
├── src/                      # Source code
│   ├── models/               # Model implementations
│   ├── datasets/             # Dataset loading
│   ├── training/             # Training scripts
│   ├── evaluation/           # Evaluation metrics
│   └── utils/                # Utilities
├── configs/                  # Configuration files
├── experiments/              # Experiment results
│   ├── baseline/
│   ├── experiment_1/
│   └── ablation/
├── checkpoints/              # Model checkpoints
├── results/                  # Results & metrics
├── logs/                     # Training logs
└── README.md                 # This file
```

## Current Status

**Phase 1**: ✅ **COMPLETED** (November 11, 2026)  
**Phase 2**: ✅ **COMPLETED** (November 11, 2026)

### Phase 1: Completed Work

- Comprehensive analysis of OmniDiff paper
- Documented research problem and motivation
- Extracted key contributions and methodology
- Analyzed dataset construction process
- Recorded experimental setup and metrics
- Identified limitations and research directions

See `research/Phase1.md` for full analysis (15 sections, 60+ key points).

### Phase 2: Completed Work

- **Official Resources**: Investigated (not yet released, will implement from scratch)
- **Environment Analysis**:
  - ✅ Python 3.11.9, PyTorch 2.5.1+cu121, CUDA 12.1
  - ✅ GPU: RTX 4050 6GB VRAM
  - ✅ Missing packages identified
- **Base Model Selected**: LLaVA-1.5-7B with 4-bit quantization
- **MDP Module**: Architecture designed (3-scale differential perception + cross-attention)
- **Training Strategy**: LoRA fine-tuning optimized for 6GB VRAM
- **Dataset Strategy**: CLEVR-Change (priority) or synthetic creation
- **Resource Estimation**: Training feasible in 3-4 hours with optimizations

See `research/Phase2.md` for complete plan (15 sections, detailed specifications).

### Next Steps (Phase 3)

**Immediate Priority**:
1. Install required packages (transformers, peft, accelerate, pycocoevalcap, bitsandbytes)
2. Verify environment setup
3. Search for/prepare dataset (CLEVR-Change or synthetic)
4. Implement baseline LLaVA for IDC
5. Implement MDP module
6. Train M³Diff model
7. Evaluate and compare with baseline
8. Document in Phase3.md

**Estimated Duration**: 18-25 days

## Research Problem Addressed

### Limitations of Existing IDC Datasets

**Breadth Issues**:
- Limited to specific scenes (streets, birds, tabletop)
- Few change types covered
- Poor real-world diversity

**Depth Issues**:
- Too short captions (8-32 words)
- Overly simplistic descriptions
- Insufficient fine-grained detail

**OmniDiff Solution**:
- 324 diverse scenarios (indoor + outdoor)
- 12 comprehensive change types
- 60-word detailed captions
- Real-world + synthetic blend

## Technical Highlights

### Multi-scale Differential Perception (MDP) Module

**Key Innovation**:
1. **Channel-wise Subtraction**: Compute multi-scale difference features
2. **Robust to Distractors**: Handle viewpoint/illumination changes
3. **Cross-Attention Fusion**: Integrate differences with original features
4. **Plug-and-Play**: Easy integration with any MLLM

**Benefits**:
- Enhanced fine-grained perception
- Precise difference localization
- Better descriptive representations
- Strong cross-dataset generalization

## Evaluation Metrics

Five standard metrics (Microsoft COCO evaluation server):

1. **BLEU-4**: N-gram precision
2. **METEOR**: Unigram matching with synonyms
3. **ROUGE-L**: Longest common subsequence
4. **CIDEr-D**: Consensus-based evaluation
5. **SPICE**: Semantic propositional content

## Development Environment

- **OS**: Windows PowerShell
- **Python**: 3.11.9 ✅
- **PyTorch**: 2.5.1+cu121 ✅
- **CUDA**: 12.1 ✅
- **GPU**: NVIDIA RTX 4050 Laptop (6GB VRAM) ✅
- **Base Model**: LLaVA-1.5-7B (4-bit quantized)

**Required Installations** (Phase 3):
- transformers, accelerate, peft, datasets
- pycocoevalcap (evaluation metrics)
- bitsandbytes, einops (optimization)

**Configuration**:
- Quantization: 4-bit inference, 8-bit LoRA training
- Batch size: 1 with gradient accumulation (16 steps)
- Mixed precision: BF16
- Gradient checkpointing: Enabled

## Research Timeline

- **November 11, 2026**: 
  - ✅ Project initialized
  - ✅ Phase 1 completed (Paper analysis)
  - ✅ Phase 2 completed (Environment & planning)
- **TBD**: Phase 3 - Baseline implementation (Est. 18-25 days)
- **TBD**: Phase 4-7 - Improvements, experiments, evaluation, final report

## Key References

- [1] GPT-4o (OpenAI)
- [8] Blender 3D engine
- [10] Birds-to-Words dataset
- [15] LoRA: Low-Rank Adaptation
- [17] Spot-the-Diff / Jhamtani et al.
- [20][33][34] CLEVR-series datasets
- [37] IEdit dataset

See `research/Phase1.md` Section 13 for complete reference list.

## Research Directions

### Dataset Improvements
- Multilingual annotations
- More diverse scenarios
- Video extension
- Automatic annotation pipelines

### Model Improvements
- Hierarchical difference modeling
- Enhanced attention mechanisms
- Contrastive learning
- Multi-task joint training
- Few-shot adaptation

### Applications
- Medical imaging
- Satellite monitoring
- Quality control
- Video summarization
- Augmented reality

## Documentation

All research progress is documented in the `research/` directory:

- **Phase{N}.md**: Detailed documentation for each research phase
- **HANDOFF.md**: Current project status and continuity information
- **WORK_LOG.md**: Chronological activity log

These files ensure that any AI model can continue the research from where it was left off.

## License

This is a research reproduction project. Original paper and code license terms apply once official resources are identified.

## Contact & Acknowledgments

**Original Paper Authors**: Liu et al. (Beijing Normal University)  
**Reproduction Project**: Research experiment for understanding and improving IDC methods

---

**Last Updated**: November 11, 2026  
**Project Status**: Phases 1-2 ✅ COMPLETED | Phase 3 ⏳ READY TO START  
**Next Priority**: Install packages and begin baseline implementation
