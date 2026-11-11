# PHASE 2 - ENVIRONMENT & REPRODUCTION PLANNING

**Status**: COMPLETED  
**Date**: November 11, 2026  
**Previous Phase**: Phase 1 ✅ COMPLETED  

---

## 1. OFFICIAL RESOURCES INVESTIGATION

### 1.1 OmniDiff Official Resources

**Project Page**: https://yuan-liu-omnidiff.github.io/  
**ArXiv Paper**: https://arxiv.org/abs/2503.11093  
**Submission History**:
- v1: Friday, March 14, 2025 (original submission)
- v2: Wednesday, February 11, 2026 (revision)

**Conference**: ICCV 2025  
**Authors**: Yuan Liu, Saihui Hou, Saijie Hou, Jiabao Du, Shibei Meng, Yongzhen Huang  
**Affiliation**: Beijing Normal University, BUPT, WATRIX.AI

### 1.2 Code & Dataset Availability Status

**Paper Statement**: "The dataset, code, and models will be made publicly available to support further research."

**Current Status** (as of November 11, 2026):
- ❌ **Official GitHub Repository**: NOT FOUND yet
  - Searched: "OmniDiff GitHub", "M3Diff GitHub", "Yuan Liu Beijing Normal University GitHub"
  - No public repository found on author's GitHub profiles
  
- ❌ **Dataset Download Link**: NOT AVAILABLE yet
  - Checked: Project page, ArXiv supplementary, HuggingFace, Google Drive
  - No download link provided in public resources
  
- ❌ **Pretrained Weights**: NOT AVAILABLE yet
  - No model checkpoints found on HuggingFace or project page

**Conclusion**: This is a **very recent paper** (ICCV 2025, ArXiv submission in March 2025). The code and dataset are likely **in preparation** for public release but **not yet available** as of November 2026.

### 1.3 Implications for Reproduction

Since official resources are not yet available, we have **two options**:

**Option A: Wait for Official Release** (NOT RECOMMENDED for research progress)
- Pros: Exact implementation details, pretrained weights, official dataset
- Cons: Unknown release timeline, blocks research progress

**Option B: Independent Implementation from Paper** (RECOMMENDED)
- Pros: Immediate progress, deep understanding, potential for improvements
- Cons: More work, may differ from official implementation
- This is the **standard research reproduction approach**

**Decision**: We will proceed with **Option B - Independent Implementation** based on paper details, which aligns with the research methodology requirement.

---

## 2. DEVELOPMENT ENVIRONMENT ANALYSIS

### 2.1 System Information

```
Operating System: Windows
Shell: PowerShell
Python Version: 3.11.9 ✅
Git Version: 2.51.0.windows.2 ✅
```

### 2.2 GPU Configuration

```
GPU: NVIDIA GeForce RTX 4050 Laptop GPU ✅
VRAM: 6.00 GB
CUDA Available: True ✅
PyTorch Version: 2.5.1+cu121 ✅
CUDA Version: 12.1 ✅
Device Count: 1
```

**Analysis**:
- ✅ **Good**: Modern RTX 4050 with CUDA support
- ⚠️ **Limitation**: Only **6GB VRAM** - this is tight for MLLM inference/training
- **Implication**: Will need to use:
  - Small base models (7B parameters max)
  - Quantization (4-bit/8-bit)
  - LoRA for efficient fine-tuning
  - Gradient checkpointing
  - Small batch sizes (likely 1-2)

### 2.3 Installed Packages Analysis

**Currently Installed** (relevant to project):

✅ **Core Deep Learning**:
- `torch==2.5.1+cu121` - PyTorch with CUDA
- `torchvision==0.20.1+cu121` - Vision utilities
- `numpy==1.26.4` - Numerical computing
- `scipy==1.15.3` - Scientific computing

✅ **Data & Preprocessing**:
- `Pillow==10.0.1` - Image processing
- `opencv-python==4.8.1.78` - Computer vision
- `pandas==2.3.3` - Data manipulation
- `scikit-learn==1.9.1` - ML utilities

✅ **Evaluation & Metrics**:
- `nltk==3.10.3` - NLP toolkit (for text metrics)
- `tqdm==4.67.3` - Progress bars

✅ **Model Hub**:
- `huggingface_hub==1.29.0` - Access to HF models

✅ **Utilities**:
- `PyYAML==6.0.3` - Config files
- `requests==2.34.2` - HTTP requests

❌ **MISSING Critical Packages**:
- `transformers` - HuggingFace Transformers (REQUIRED for MLLM)
- `accelerate` - Efficient training utilities
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `bitsandbytes` - 8-bit/4-bit quantization
- `sentencepiece` - Tokenization
- `protobuf` (installed but may need upgrade)
- `pycocotools` or `pycocoevalcap` - COCO evaluation metrics

### 2.4 Required Installation List

**Priority 1 - Essential (Must Install)**:
```bash
pip install transformers
pip install accelerate
pip install peft  # For LoRA fine-tuning
pip install datasets  # HuggingFace datasets
pip install sentencepiece  # Tokenization
```

**Priority 2 - Evaluation (Must Install)**:
```bash
pip install pycocoevalcap  # COCO evaluation metrics (BLEU, METEOR, etc.)
# OR if not available:
pip install pycocotools
pip install cider  # CIDEr metric
pip install spice  # SPICE metric (may need Java)
```

**Priority 3 - Optimization (Recommended)**:
```bash
pip install bitsandbytes  # 8-bit optimization for 6GB VRAM
pip install einops  # Tensor operations
pip install timm  # Vision models
```

**Priority 4 - Development (Optional)**:
```bash
pip install tensorboard  # Training visualization
pip install wandb  # Experiment tracking
pip install black  # Code formatting
```

---

## 3. BASE MLLM SELECTION

### 3.1 Candidate Base Models

Given our **6GB VRAM constraint**, we need a **small yet capable** MLLM:

#### Option 1: LLaVA-1.5-7B (RECOMMENDED)
- **Repository**: https://github.com/haotian-liu/LLaVA
- **Size**: 7B parameters
- **VRAM**: ~5-6GB with 4-bit quantization
- **Pros**:
  - Well-established MLLM architecture
  - Excellent documentation and community support
  - Proven performance on vision-language tasks
  - Easy to customize (multimodal_projector)
  - Official transformers integration
- **Cons**:
  - Requires quantization for 6GB VRAM
- **Status**: ✅ **SELECTED as primary base**

#### Option 2: MiniGPT-4 (Alternative)
- Smaller variant possible
- Less documentation
- **Status**: Backup option

#### Option 3: InstructBLIP (Alternative)
- BLIP-2 based
- Good vision-language alignment
- **Status**: Backup option

### 3.2 Selected Configuration

**Base Model**: `LLaVA-1.5-7B`

**Quantization Strategy**:
- Use **4-bit quantization** (BitsAndBytes) for inference
- Use **8-bit LoRA** for fine-tuning
- Enables running on 6GB VRAM

**Model Components**:
```
LLaVA-1.5-7B
├── Vision Encoder: CLIP ViT-L/14 (frozen)
├── Multi-modal Projector: Linear layers (trainable)
├── Language Model: Vicuna-7B (LoRA)
└── [NEW] MDP Module: Multi-scale Differential Perception (our contribution)
```

---

## 4. DATASET STRATEGY

### 4.1 OmniDiff Dataset - Current Status

**Official Dataset**: NOT YET RELEASED

**Specifications** (from paper):
- 15,598 image pairs
- 324 scenes
- 60 words average caption
- 12 change types
- Real-world (8,609) + Synthetic (6,989)

### 4.2 Reproduction Strategy - Phase-wise Approach

Since official dataset not available, we have **two parallel tracks**:

#### Track A: Existing Benchmark Datasets (IMMEDIATE)

Use **publicly available** IDC datasets to develop and test M³Diff:

1. **Spot-the-Diff** [17]
   - Surveillance footage
   - 13,192 pairs
   - Status: Check availability

2. **IEdit** [37]
   - Diverse real-life
   - 3,939 pairs
   - Status: Check availability

3. **CLEVR-Change** [33]
   - Synthetic tabletop
   - 79,606 pairs
   - Status: Likely available (CLEVR is public)

4. **CLEVR-DC** [20]
   - Viewpoint-agnostic
   - Status: Check availability

**Strategy**: 
- Start with **CLEVR-Change** (most likely available)
- Implement and train M³Diff baseline
- Validate MDP module effectiveness
- Extend to other datasets as they become available

#### Track B: Synthetic Dataset Creation (PARALLEL)

**Create mini-OmniDiff** using Blender:

**Phase 2B.1**: Simple Synthetic Scenes (10-20 scenes)
- Use Blender to create basic 3D environments
- Generate image pairs with controlled changes
- Manual annotation or template-based captions
- Validate MDP module on controlled data

**Phase 2B.2**: Complex Synthetic Scenes (50-100 scenes)
- Use ArtStation-like 3D assets
- More realistic environments
- 12 change types coverage
- More detailed annotations

**Phase 2B.3**: Real-world Collection (if needed)
- On-site photography (similar to paper)
- Web scraping (with proper permissions)
- Human annotation

**Timeline**:
- Phase 2B.1: Can start immediately after baseline
- Phase 2B.2-3: Contingent on official release status

### 4.3 Annotation Strategy

**For Synthetic Data**:
- Use **template-based generation** for initial version
- Gradually increase complexity
- Consider GPT-4 Vision for annotation assistance (with human verification)

**For Real-world Data** (if collecting):
- Follow paper's annotation guidelines:
  - Referring part + Change part structure
  - Focus on semantic changes
  - Ignore distractors (viewpoint, illumination)

---

## 5. MDP MODULE IMPLEMENTATION PLAN

### 5.1 MDP Module Architecture

Based on paper analysis, the MDP module consists of:

```
Multi-scale Differential Perception (MDP) Module
│
├── Input: Feature pairs from Vision Encoder
│   ├── Image A features: [B, C, H, W]
│   └── Image B features: [B, C, H, W]
│
├── Multi-scale Feature Extraction
│   ├── Scale 1: Original resolution
│   ├── Scale 2: Downsampled 2x
│   └── Scale 3: Downsampled 4x
│
├── Channel-wise Subtraction
│   └── Diff_features = |Features_A - Features_B|
│
├── Cross-Attention Fusion
│   ├── Query: Original features
│   ├── Key/Value: Difference features
│   └── Output: Fused features
│
└── Output: Enhanced features for LLM
```

### 5.2 Implementation Steps

**Step 1: Basic Differential Perception** (Phase 3)
```python
class DifferentialPerception(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
    
    def forward(self, feat_a, feat_b):
        # Channel-wise subtraction
        diff = torch.abs(feat_a - feat_b)
        # Project difference features
        diff_proj = self.projection(diff)
        return diff_proj
```

**Step 2: Multi-scale Extension** (Phase 3)
```python
class MultiScaleDifferentialPerception(nn.Module):
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.scales = nn.ModuleList([
            DifferentialPerception(dim) for _ in range(num_scales)
        ])
        self.downsample = nn.ModuleList([
            nn.AvgPool2d(2**i) for i in range(num_scales)
        ])
```

**Step 3: Cross-Attention Fusion** (Phase 3)
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, original_feat, diff_feat):
        # Cross-attention: Q from original, K/V from difference
        fused, _ = self.cross_attn(
            query=original_feat,
            key=diff_feat,
            value=diff_feat
        )
        return self.norm(original_feat + fused)
```

**Step 4: Full MDP Module** (Phase 3)
```python
class MDPModule(nn.Module):
    def __init__(self, dim, num_scales=3, num_heads=8):
        super().__init__()
        self.multi_scale_diff = MultiScaleDifferentialPerception(dim, num_scales)
        self.fusion = CrossAttentionFusion(dim, num_heads)
    
    def forward(self, feat_a, feat_b):
        # Multi-scale difference
        diff_feats = self.multi_scale_diff(feat_a, feat_b)
        # Cross-attention fusion
        fused_a = self.fusion(feat_a, diff_feats)
        fused_b = self.fusion(feat_b, diff_feats)
        # Return both enhanced features
        return fused_a, fused_b
```

### 5.3 Integration with LLaVA

**Modification Points**:

1. **Vision Encoder Output**: Extract multi-scale features
2. **Before Projector**: Insert MDP module
3. **Projector Input**: Use fused features instead of raw features
4. **LLM Input**: Concatenate enhanced vision tokens from both images

**Integration Architecture**:
```
Image Pair (A, B)
    ↓
CLIP Vision Encoder (frozen)
    ↓
Vision Features (A, B)
    ↓
[INSERT] MDP Module ← OUR CONTRIBUTION
    ↓
Enhanced Features (A', B')
    ↓
Multi-modal Projector (trainable)
    ↓
Vision Tokens → [IMG_A] [IMG_B]
    ↓
Concatenate: [IMG_A] [IMG_B] [TEXT_PROMPT]
    ↓
LLM (Vicuna-7B with LoRA)
    ↓
Generated Caption
```

---

## 6. TRAINING STRATEGY

### 6.1 Training Paradigm

Following paper's "simple yet effective fine-tuning strategy":

**Phase 3A: Baseline (Without MDP)**
- Establish baseline with standard LLaVA on available IDC dataset
- Verify environment and training pipeline
- Measure baseline performance

**Phase 3B: Add MDP Module**
- Integrate MDP module into architecture
- Train with MDP enabled
- Compare with baseline

**Phase 3C: Full M³Diff**
- Complete M³Diff implementation
- Full training on best available dataset
- Cross-dataset evaluation

### 6.2 Training Configuration

**Given 6GB VRAM Constraints**:

```yaml
# Model Configuration
base_model: "liuhaotian/llava-v1.5-7b"
quantization: "4bit"  # For inference
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training Configuration
batch_size: 1  # Gradient accumulation to simulate larger batch
gradient_accumulation_steps: 16  # Effective batch size = 16
learning_rate: 2e-4
num_epochs: 3
warmup_ratio: 0.03
weight_decay: 0.0

# Optimization
mixed_precision: "bf16"  # Or "fp16"
gradient_checkpointing: true
max_seq_length: 512  # Limit sequence length
max_image_size: 336  # LLaVA default

# MDP Module
mdp_scales: 3
mdp_heads: 8
mdp_dim: 1024  # Match CLIP output dim
```

### 6.3 Training Phases Breakdown

**Phase 3 Training Schedule**:

1. **Pretrain MDP Module** (Optional, 1 epoch)
   - Freeze vision encoder + LLM
   - Train only MDP + projector
   - On large dataset (CLEVR-Change)

2. **Fine-tune Full Model** (3 epochs)
   - Freeze vision encoder
   - Train MDP + projector + LoRA
   - On target dataset

3. **Ablation Studies** (Phase 6)
   - Without MDP (baseline)
   - With MDP (full model)
   - MDP variants (scales, attention)

---

## 7. EVALUATION PIPELINE

### 7.1 Metrics Implementation

**Required Metrics** (from paper):
1. **BLEU-4**: N-gram precision
2. **METEOR**: Unigram matching with synonyms
3. **ROUGE-L**: Longest common subsequence
4. **CIDEr-D**: Consensus-based evaluation
5. **SPICE**: Semantic propositional content

**Implementation Options**:

**Option A**: Use `pycocoevalcap` (RECOMMENDED)
```python
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Unified evaluation
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE")
]
```

**Option B**: Use HuggingFace `evaluate` library
```python
import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
# Note: CIDEr and SPICE may not be available
```

**Decision**: Use **Option A** (pycocoevalcap) for complete COCO-style evaluation.

### 7.2 Evaluation Script Structure

```python
# evaluate.py
def evaluate_model(model, dataloader, metrics):
    predictions = []
    references = []
    
    for batch in dataloader:
        # Generate captions
        pred = model.generate(batch)
        predictions.append(pred)
        references.append(batch['caption'])
    
    # Compute all metrics
    scores = compute_metrics(predictions, references, metrics)
    return scores

def compute_metrics(preds, refs, metrics):
    # Format for COCO eval
    gts = {i: [ref] for i, ref in enumerate(refs)}
    res = {i: [pred] for i, pred in enumerate(preds)}
    
    results = {}
    for scorer, method in metrics:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                results[m] = sc
        else:
            results[method] = score
    
    return results
```

---

## 8. PROJECT STRUCTURE REFINEMENT

### 8.1 Detailed Source Code Organization

```
src/
├── models/
│   ├── __init__.py
│   ├── mdp_module.py          # Multi-scale Differential Perception
│   ├── m3diff.py              # Complete M³Diff model
│   ├── llava_wrapper.py       # LLaVA base model wrapper
│   └── utils.py               # Model utilities
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py        # Base IDC dataset class
│   ├── clevr_change.py        # CLEVR-Change loader
│   ├── spot_diff.py           # Spot-the-Diff loader
│   ├── iedit.py               # IEdit loader
│   ├── omnidiff.py            # OmniDiff loader (when available)
│   ├── synthetic.py           # Our synthetic dataset
│   └── transforms.py          # Image transformations
│
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Main training loop
│   ├── trainer_config.py      # Training configurations
│   └── optimizer.py           # Optimizer and scheduler setup
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # BLEU, METEOR, ROUGE, CIDEr, SPICE
│   ├── evaluator.py           # Evaluation pipeline
│   └── visualize.py           # Result visualization
│
└── utils/
    ├── __init__.py
    ├── config.py              # Configuration management
    ├── logging.py             # Logging utilities
    ├── checkpoint.py          # Checkpoint save/load
    └── data_utils.py          # Data processing utilities
```

### 8.2 Configuration Files Structure

```
configs/
├── base_config.yaml           # Base configuration
├── mdp_config.yaml            # MDP module config
├── training/
│   ├── baseline.yaml          # Baseline training
│   ├── m3diff_full.yaml       # Full M³Diff training
│   └── ablation.yaml          # Ablation studies
├── datasets/
│   ├── clevr_change.yaml      # CLEVR-Change dataset config
│   ├── spot_diff.yaml         # Spot-the-Diff config
│   └── synthetic.yaml         # Synthetic dataset config
└── models/
    ├── llava_7b.yaml          # LLaVA-7B config
    └── llava_7b_quantized.yaml # Quantized version
```

---

## 9. IMPLEMENTATION TIMELINE & MILESTONES

### 9.1 Phase 3 Breakdown (Baseline Implementation)

**Phase 3.1**: Environment Setup (1 day)
- ✅ Already analyzed environment
- [ ] Install missing packages
- [ ] Verify installations
- [ ] Test basic LLaVA loading

**Phase 3.2**: Dataset Preparation (2-3 days)
- [ ] Search and download CLEVR-Change (or alternative)
- [ ] Implement dataset loader
- [ ] Implement preprocessing pipeline
- [ ] Verify data loading

**Phase 3.3**: LLaVA Baseline (3-4 days)
- [ ] Load LLaVA-1.5-7B with quantization
- [ ] Adapt for image pair input
- [ ] Implement baseline training script
- [ ] Run baseline training (1 epoch test)
- [ ] Verify baseline works

**Phase 3.4**: MDP Module Implementation (3-4 days)
- [ ] Implement differential perception
- [ ] Implement multi-scale extraction
- [ ] Implement cross-attention fusion
- [ ] Integrate with LLaVA architecture
- [ ] Test forward pass

**Phase 3.5**: M³Diff Training (5-7 days)
- [ ] Full training setup
- [ ] Train M³Diff model
- [ ] Monitor training progress
- [ ] Save checkpoints

**Phase 3.6**: Evaluation & Comparison (2-3 days)
- [ ] Implement evaluation metrics
- [ ] Evaluate baseline vs M³Diff
- [ ] Generate comparison results
- [ ] Create Phase3.md document

**Total Phase 3 Estimate**: ~18-25 days (with 6GB VRAM constraints)

### 9.2 Potential Blockers & Mitigation

**Blocker 1**: Dataset Not Available
- **Mitigation**: Use CLEVR-Change (most likely public) or create synthetic dataset
- **Fallback**: Simple synthetic scenes with Blender

**Blocker 2**: VRAM Insufficient for Training
- **Mitigation**: More aggressive quantization, smaller batch size, gradient accumulation
- **Fallback**: Use gradient checkpointing, reduce model size, or use CPU offloading

**Blocker 3**: Evaluation Metrics Installation Issues
- **Mitigation**: Manual implementation of metrics if packages unavailable
- **Fallback**: Use HuggingFace evaluate library subset

**Blocker 4**: Training Too Slow
- **Mitigation**: Reduce dataset size, shorter epochs, smaller model
- **Fallback**: Focus on correctness over convergence for proof-of-concept

---

## 10. COMPUTATIONAL RESOURCE ESTIMATION

### 10.1 Memory Requirements

**Inference** (with 4-bit quantization):
- LLaVA-7B quantized: ~3.5 GB
- MDP Module: ~0.5 GB
- Activation memory: ~1-1.5 GB
- **Total**: ~5-5.5 GB ✅ (Fits in 6GB)

**Training** (with optimizations):
- Model: ~4 GB
- Gradients (LoRA only): ~0.5 GB
- Optimizer states: ~1 GB
- Batch data (size=1): ~0.5 GB
- **Total**: ~6 GB ✅ (Tight but feasible)

**Optimizations Needed**:
- Gradient checkpointing: ✅ Essential
- Gradient accumulation: ✅ Essential (simulate larger batch)
- 8-bit Adam optimizer: ✅ Recommended
- Mixed precision (BF16): ✅ Essential

### 10.2 Time Estimates

**Assumptions**:
- Dataset: 10,000 pairs (CLEVR-Change subset)
- Batch size: 1 (with accumulation=16)
- Epochs: 3
- GPU: RTX 4050 (6GB)

**Estimates**:
- Forward pass per image pair: ~2-3 seconds
- Backward pass: ~3-4 seconds
- Total per step: ~6 seconds

**Training Time**:
- Steps per epoch: 10,000 / 16 = 625 steps
- Time per epoch: 625 * 6 = 3,750 seconds ~ **1 hour**
- Total training: 3 epochs ~ **3-4 hours**

**Evaluation Time**:
- Test set: ~1,000 pairs
- Generation time: ~2 seconds/pair
- Total evaluation: **~30-45 minutes**

### 10.3 Disk Space Requirements

- LLaVA-1.5-7B model: ~14 GB (full precision)
- LLaVA-1.5-7B (quantized cache): ~4 GB
- Dataset (CLEVR-Change): ~10-20 GB
- Checkpoints (LoRA only): ~100 MB per checkpoint
- Logs and results: ~500 MB
- **Total**: ~30-40 GB

**Recommendation**: Ensure at least **50 GB free space**.

---

## 11. RISK ASSESSMENT & MITIGATION

### Risk 1: Insufficient VRAM (HIGH PROBABILITY)
**Impact**: Cannot train model, project blocked  
**Mitigation**:
- Use aggressive 4-bit quantization
- Implement gradient checkpointing
- Use CPU offloading for some layers
- Reduce max sequence length
**Contingency**: Use Google Colab Pro or cloud GPU (A100)

### Risk 2: Dataset Unavailable (MEDIUM PROBABILITY)
**Impact**: Cannot reproduce exact results  
**Mitigation**:
- Use publicly available IDC datasets (CLEVR-Change)
- Create synthetic dataset with Blender
**Contingency**: Focus on methodology validation, not exact reproduction

### Risk 3: Training Instability (MEDIUM PROBABILITY)
**Impact**: Model doesn't converge, poor results  
**Mitigation**:
- Start with baseline LLaVA (verify it works)
- Add MDP incrementally
- Monitor loss curves carefully
- Use gradient clipping
**Contingency**: Simplify MDP module, reduce complexity

### Risk 4: Evaluation Metrics Issues (LOW PROBABILITY)
**Impact**: Cannot compare with paper  
**Mitigation**:
- Install pycocoevalcap early
- Test metrics before full training
- Prepare manual implementation
**Contingency**: Use subset of metrics, qualitative evaluation

### Risk 5: Time Overrun (MEDIUM PROBABILITY)
**Impact**: Cannot complete all phases  
**Mitigation**:
- Prioritize Phase 3 (baseline) completion
- Document thoroughly for handoff
- Use smaller datasets for faster iteration
**Contingency**: Complete Phases 3-4, defer 5-7 to next session

---

## 12. NEXT STEPS (Immediate Actions for Phase 3)

### Step 1: Install Required Packages ✅ NEXT
```bash
cd d:\PYTHON\XLA_Experiment

# Essential packages
pip install transformers accelerate peft
pip install datasets sentencepiece protobuf

# Evaluation metrics
pip install pycocoevalcap
# If not available, try:
# pip install git+https://github.com/tylin/coco-caption.git

# Optimization
pip install bitsandbytes einops

# Development
pip install tensorboard wandb
```

### Step 2: Test Basic Setup
```bash
# Create test script
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Transformers:', transformers.__version__)
print('GPU:', torch.cuda.get_device_name(0))
"
```

### Step 3: Search for Dataset
- Search for CLEVR-Change dataset
- Check availability of Spot-the-Diff
- Document findings

### Step 4: Create Basic Project Structure
```bash
mkdir src\models src\datasets src\training src\evaluation src\utils
mkdir configs\training configs\datasets configs\models
```

### Step 5: Begin Baseline Implementation
- Load LLaVA-1.5-7B
- Adapt for image pairs
- Test inference on dummy data

---

## 13. SUCCESS CRITERIA FOR PHASE 2

✅ **Phase 2 is COMPLETE when**:

- [x] Official resources investigated (code/dataset availability checked)
- [x] Development environment fully analyzed (Python, PyTorch, GPU, packages)
- [x] Base MLLM selected (LLaVA-1.5-7B)
- [x] MDP module architecture designed
- [x] Dataset strategy defined (existing + synthetic)
- [x] Training strategy planned (LoRA + quantization)
- [x] Evaluation pipeline designed
- [x] Project structure refined
- [x] Resource requirements estimated
- [x] Risk assessment completed
- [x] Phase2.md document created
- [x] Next steps clearly defined

**Status**: ✅ **ALL CRITERIA MET**

---

## 14. HANDOFF TO PHASE 3

### What Phase 3 Needs to Do:

1. **Install packages** (Step 1 from Next Steps)
2. **Verify environment** (GPU, transformers loading)
3. **Find dataset** (CLEVR-Change or alternative)
4. **Implement baseline** (LLaVA for IDC)
5. **Implement MDP module**
6. **Train M³Diff**
7. **Evaluate results**
8. **Create Phase3.md**

### Files Phase 3 Will Create:

```
src/models/mdp_module.py
src/models/m3diff.py
src/models/llava_wrapper.py
src/datasets/base_dataset.py
src/datasets/clevr_change.py (or other)
src/training/trainer.py
src/evaluation/metrics.py
configs/base_config.yaml
configs/mdp_config.yaml
experiments/baseline/logs/
experiments/baseline/checkpoints/
```

### Expected Outputs from Phase 3:

- Working baseline LLaVA on IDC task
- Implemented MDP module
- Trained M³Diff model
- Evaluation metrics computed
- Comparison: Baseline vs M³Diff
- Phase3.md documentation

---

## 15. SUMMARY

### Key Decisions Made:

1. ✅ **Implementation Approach**: Independent from-scratch (official not available)
2. ✅ **Base Model**: LLaVA-1.5-7B with 4-bit quantization
3. ✅ **Training Method**: LoRA fine-tuning for efficiency
4. ✅ **Dataset Strategy**: Start with CLEVR-Change, create synthetic if needed
5. ✅ **Resource Management**: Aggressive optimization for 6GB VRAM

### Feasibility Assessment:

| Aspect | Status | Confidence |
|--------|--------|------------|
| Environment | ✅ Ready | HIGH |
| Base Model | ✅ Available | HIGH |
| Dataset | ⚠️ TBD | MEDIUM |
| VRAM Sufficiency | ⚠️ Tight | MEDIUM |
| Training Time | ✅ Acceptable | HIGH |
| Implementation | ✅ Planned | HIGH |

**Overall Feasibility**: ✅ **FEASIBLE** with identified optimizations

### Phase 2 Status:

**✅ COMPLETED**

We have:
- Thoroughly investigated official resources
- Analyzed development environment
- Selected and justified base model
- Designed MDP module architecture
- Planned dataset strategy
- Designed training pipeline
- Estimated resources
- Assessed risks
- Defined clear next steps

**Ready to proceed to Phase 3**: ✅ YES

---

**PHASE 2 STATUS: ✅ COMPLETED**  
**Next Phase**: Phase 3 - Baseline Implementation & Reproduction  
**Estimated Duration**: 18-25 days  
**Priority**: Install packages and verify environment
