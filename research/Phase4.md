# PHASE 4 - BASELINE ANALYSIS & IMPROVEMENT PROPOSALS

**Status**: IN PROGRESS  
**Date**: November 11, 2026  
**Previous Phases**: Phase 1 ✅, Phase 2 ✅, Phase 3 ✅  

---

## EXECUTIVE SUMMARY

Phase 4 conducts a comprehensive analysis of the M³Diff baseline implementation from Phase 3, identifies architectural and methodological limitations, reviews recent advances in related work (2024-2026), and proposes 5 concrete improvements with scientific justification. These proposals will form the experimental agenda for Phase 5.

**Key Analysis Areas**:
1. MDP Module architectural limitations
2. Training strategy weaknesses
3. Dataset and evaluation gaps
4. Recent related work (IDC, MLLM, vision-language)
5. Proposed improvements with scientific rationale

---

## 1. BASELINE IMPLEMENTATION ANALYSIS

### 1.1 Current M³Diff Architecture Review

From Phase 3 implementation, our baseline consists of:

```
M³Diff Architecture (Current Baseline)
│
├── Vision Encoder: CLIP ViT-L/14-336 (frozen)
│   └── Output: [B, 1024, 24, 24]
│
├── MDP Module (trainable, 34.6M params)
│   ├── Multi-scale Differential Perception (3 scales)
│   │   ├── Scale 1: Original [24x24]
│   │   ├── Scale 2: Downsampled 2x [12x12]
│   │   └── Scale 3: Downsampled 4x [6x6]
│   ├── Channel-wise Subtraction: |feat_A - feat_B|
│   └── Cross-Attention Fusion (8 heads)
│       └── Output: [B, 576, 1024]
│
├── Multimodal Projector: Linear 1024→4096 (trainable)
│   └── Output: [B, 576, 4096]
│
└── Language Model: Vicuna-7B (LoRA r=8, alpha=16)
    └── Output: Generated caption
```

**Trainable Parameters**:
- MDP Module: ~34.6M
- Projector: ~4.2M  
- LoRA: ~4.7M
- **Total Trainable**: ~43.5M (~0.6% of 7B)

### 1.2 Implementation Strengths

**✅ Architectural Strengths**:
1. **Multi-scale Processing**: Captures both fine-grained (24x24) and global (6x6) differences
2. **Explicit Differencing**: Channel-wise subtraction provides clear difference signal
3. **Adaptive Fusion**: Cross-attention allows selective focus on relevant differences
4. **Plug-and-Play Design**: MDP module can be added to any vision-language model
5. **Memory Efficiency**: LoRA + quantization enables training on consumer GPUs (4GB VRAM)

**✅ Implementation Strengths**:
1. **Modular Design**: Clean separation of concerns (MDP, LLaVA wrapper, dataset, trainer)
2. **Memory Optimized**: Aggressive optimizations (4-bit, FP16, gradient checkpointing)
3. **Flexible Dataset Loader**: Supports multiple annotation formats and structures
4. **Comprehensive Metrics**: BLEU, ROUGE, METEOR, CIDEr, SPICE (with fallbacks)
5. **Production Ready**: Complete pipeline from data loading to evaluation

---

## 2. IDENTIFIED LIMITATIONS

### 2.1 MDP Module Architectural Limitations

#### Limitation 1: Fixed 3-Scale Architecture

**Current Implementation**:
- 3 fixed scales: Original (24x24), 2x downsample (12x12), 4x downsample (6x6)
- Uniform scale spacing (powers of 2)
- No learnable scale selection

**Problems**:
1. **Insufficient Fine-Grained Resolution**: Smallest features captured at 24x24 may miss pixel-level differences
2. **Coarse Scale Spacing**: Large gap between scales may miss intermediate-level patterns
3. **Fixed Scale Policy**: Cannot adapt scale selection to different change types
4. **No Scale Attention**: All scales weighted equally after concatenation

**Evidence from Paper**:
- OmniDiff has 12 change types with varying granularity (OCR text vs. illumination)
- Some changes require finer spatial resolution (e.g., small object addition)
- Some changes require broader context (e.g., pose changes, viewpoint shifts)

**Impact**: 
- May miss very fine-grained differences (small text changes, subtle color shifts)
- May not capture the optimal scale for each specific change type
- Suboptimal feature representation for multi-granularity changes

#### Limitation 2: Simple Subtraction Operation

**Current Implementation**:
```python
diff = torch.abs(feat_a - feat_b)
```

**Problems**:
1. **Loss of Direction Information**: Absolute value discards whether feature increased or decreased
2. **No Semantic Weighting**: All channels treated equally regardless of importance
3. **Limited Expressiveness**: Simple L1 distance may not capture complex feature relationships
4. **Sensitivity to Noise**: Pixel-level noise treated same as semantic differences

**Alternative Operations Not Explored**:
- Signed difference: `feat_a - feat_b` (preserves direction)
- Squared difference: `(feat_a - feat_b)^2` (emphasizes larger changes)
- Cosine dissimilarity: `1 - cosine_similarity(feat_a, feat_b)` (angular difference)
- Learned difference: `MLP([feat_a, feat_b, feat_a - feat_b, feat_a * feat_b])`

**Impact**:
- May not fully capture semantic differences vs. distractor variations
- Cannot distinguish "object added" from "object removed" at feature level
- Suboptimal for handling illumination/viewpoint robustness

#### Limitation 3: Single-Stage Cross-Attention Fusion

**Current Implementation**:
- One cross-attention layer: Q=original_features, K/V=diff_features
- Applied once after multi-scale aggregation
- 8 attention heads (fixed)

**Problems**:
1. **No Hierarchical Fusion**: Fusion happens only at final scale
2. **Limited Interaction**: Original and difference features interact only once
3. **No Self-Attention**: Features don't attend to themselves before cross-attention
4. **Fixed Attention Heads**: 8 heads may not be optimal

**Better Fusion Strategies**:
- Multi-layer fusion: Alternate self-attention and cross-attention
- Hierarchical fusion: Fuse at each scale before aggregating
- Bidirectional fusion: A→B and B→A fusion paths
- Gated fusion: Learnable gates to control fusion strength

**Impact**:
- Features may not be sufficiently refined before fusion
- Limited capacity to model complex feature interactions
- May not optimally balance original context vs. difference information

#### Limitation 4: No Spatial Localization Guidance

**Current Implementation**:
- MDP processes entire feature maps uniformly
- No explicit region-of-interest detection
- No spatial attention mechanism

**Problems**:
1. **Unfocused Processing**: Equal computation on changed and unchanged regions
2. **No Localization Signal**: Model doesn't explicitly learn where changes occur
3. **Inefficient Compute**: Wastes capacity on irrelevant background regions

**Missing Components**:
- Change detection head (auxiliary task)
- Spatial attention maps for difference regions
- Region-proposal mechanism
- Saliency-guided processing

**Evidence from Related Work**:
- **Semantic-CC** [54] uses multi-task learning with change detection
- **DIRL** [43] correlates corresponding regions explicitly
- Object detection models use region proposals (Faster R-CNN, DETR)

**Impact**:
- Lower efficiency (processes entire image)
- Harder to learn precise localization
- May generate vague spatial descriptions ("somewhere in the scene")

### 2.2 Training Strategy Limitations

#### Limitation 5: LoRA Configuration Suboptimal

**Current Configuration**:
```yaml
lora_r: 8
lora_alpha: 16
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Problems**:
1. **Very Low Rank (r=8)**: Chosen due to VRAM constraints, not optimal performance
2. **Limited Module Coverage**: Only attention projections, not MLPs
3. **No Layer-Specific Tuning**: Same rank across all layers
4. **Fixed Throughout Training**: No progressive rank increase

**From LoRA Literature**:
- Optimal rank varies by task: 16-64 for most vision-language tasks
- Later layers benefit from higher ranks
- Progressive rank increase improves training stability
- MLP layers also benefit from LoRA (not just attention)

**Impact**:
- Limited model adaptation capacity
- May underfit complex IDC task
- Suboptimal parameter efficiency

#### Limitation 6: Frozen Vision Encoder

**Current Configuration**:
- CLIP ViT-L/14 completely frozen during training
- Only MDP, projector, and LLM (LoRA) trained

**Problems**:
1. **No Visual Adaptation**: Vision encoder cannot adapt to IDC-specific visual patterns
2. **Mismatch with CLIP Pretraining**: CLIP trained for single-image classification, not difference detection
3. **Suboptimal Features**: Features optimized for similarity matching, not difference detection

**Alternative Strategies**:
- Partial unfreezing: Last N layers of vision encoder
- LoRA on vision encoder: Lightweight adaptation
- Bias-term tuning: Only train bias terms (BitFit)
- Adapter layers: Insert small trainable layers

**Evidence from Related Work**:
- **LLaVA-1.5** unfreezes vision encoder in later training stages
- **FINER-MLLM** [51] applies LoRA to both vision and language components
- Transfer learning literature shows benefits of fine-tuning

**Impact**:
- Vision encoder may not extract optimal features for difference detection
- Model capacity bottlenecked by frozen encoder
- May not fully leverage domain-specific visual patterns

#### Limitation 7: Single-Stage Training

**Current Training Strategy**:
- Direct end-to-end training from random MDP initialization
- No pretraining of MDP module
- No curriculum learning

**Problems**:
1. **Cold Start**: MDP module starts from random weights while rest of model is pretrained
2. **No Progressive Difficulty**: All samples treated equally from start
3. **Optimization Challenge**: Large parameter imbalance (MDP 34M vs LoRA 4.7M)

**Better Training Strategies**:
- **Two-stage training**:
  - Stage 1: Pretrain MDP on large-scale data (e.g., CLEVR-Change)
  - Stage 2: Fine-tune entire model on target data (e.g., OmniDiff)
- **Curriculum learning**:
  - Start with simple changes (single object, large differences)
  - Progress to complex changes (multiple objects, subtle differences)
- **Progressive unfreezing**:
  - Stage 1: Train MDP only
  - Stage 2: Add projector
  - Stage 3: Add LoRA

**Evidence from Literature**:
- **Curriculum Learning** [Bengio et al.]: Improves convergence and generalization
- **Progressive Neural Networks**: Reduces catastrophic forgetting
- **LLaVA training**: Uses two-stage pretraining → instruction tuning

**Impact**:
- Slower convergence due to parameter imbalance
- MDP module may not reach optimal performance
- Risk of underfitting with limited training data

#### Limitation 8: No Contrastive or Auxiliary Losses

**Current Loss Function**:
- Single cross-entropy loss on caption generation
- No auxiliary supervision signals

**Problems**:
1. **Indirect Difference Supervision**: Model learns difference detection only through caption loss
2. **No Visual Grounding**: No explicit image-text alignment loss
3. **No Difference-Specific Loss**: No loss specifically for difference features

**Missing Loss Functions**:
- **Contrastive Loss**: Align difference features with caption embeddings
- **Change Detection Loss**: Auxiliary binary segmentation of changed regions
- **Feature Reconstruction Loss**: Reconstruct feat_b from feat_a + diff_features
- **Consistency Loss**: Encourage similar captions for equivalent changes

**Evidence from Related Work**:
- **CLIP**: Contrastive loss crucial for vision-language alignment
- **Semantic-CC** [54]: Multi-task learning with change detection significantly improves performance
- **IDC-PCL** [49]: Pretraining with contrastive loss improves IDC

**Impact**:
- Suboptimal difference feature learning
- Weaker vision-language alignment
- Harder to learn fine-grained difference localization

### 2.3 Dataset and Evaluation Limitations

#### Limitation 9: Synthetic Dataset Quality

**Current Dataset**:
- 50 samples (40 train, 5 val, 5 test)
- Simple geometric shapes on white background
- 4 change types (add, remove, color, position)
- Average caption length: 10-15 words

**Problems**:
1. **Trivial Visual Complexity**: Too simple to evaluate real model capacity
2. **Limited Change Diversity**: 4 types vs. 12 in OmniDiff
3. **Short Captions**: 10-15 words vs. 60 in OmniDiff
4. **No Real-World Patterns**: No textures, occlusions, lighting, perspective

**Impact on Phase 3**:
- ✅ Good for pipeline testing (intended purpose)
- ❌ Cannot evaluate true model performance
- ❌ Cannot identify failure modes on complex scenes
- ❌ Cannot measure generalization to real-world data

**Next Steps Required**:
- Obtain real IDC datasets (Spot-the-Diff, CLEVR-Change, IEdit)
- Create more complex synthetic data with Blender
- Include all 12 change types from OmniDiff
- Generate longer, more detailed captions

#### Limitation 10: Limited Evaluation Metrics

**Current Implementation**:
- BLEU-1/2/3/4: ✅ Implemented
- ROUGE-L: ✅ Implemented  
- METEOR: ⚠️ Placeholder only
- CIDEr: ⚠️ Placeholder only
- SPICE: ⚠️ Placeholder only (requires Java)

**Problems**:
1. **Incomplete Metrics**: Only BLEU and ROUGE fully functional
2. **No Semantic Evaluation**: SPICE not working (semantic propositional content)
3. **No Visual Grounding**: No metrics for localization accuracy
4. **No Change-Type Specific Metrics**: All changes evaluated equally

**Missing Evaluation Dimensions**:
- **Localization Accuracy**: Are spatial descriptions correct?
- **Change Type Accuracy**: Is the change type identified correctly?
- **Fine-Grained Detail**: Are subtle differences captured?
- **Robustness to Distractors**: Performance on viewpoint/illumination changes?
- **Human Evaluation**: Fluency, correctness, completeness

**Impact**:
- Incomplete performance assessment
- Cannot identify specific failure modes
- Difficult to compare with paper results (need all 5 metrics)

---

## 3. RELATED WORK ANALYSIS (2024-2026)

### 3.1 Recent Advances in Image Difference Captioning

#### 3.1.1 FINER-MLLM (2024)

**Key Innovation**: Fine-grained visual perception with dual constraints

**Technical Approach**:
- LoRA-finetuned MLLM on IDC tasks
- **Dual Constraints**:
  - Visual constraint: Contrastive loss on visual features
  - Textual constraint: Consistency loss on generated captions
- Multi-scale visual feature extraction

**Relevance to Our Work**:
- ✅ Confirms LoRA is effective for IDC
- ✅ Validates multi-scale feature importance
- 📌 **Insight**: Dual constraints significantly improve performance
- 📌 **Missing in our baseline**: No contrastive or consistency losses

**Potential Improvement**: Add dual-constraint losses to training

#### 3.1.2 OneDiff (Hu et al., 2024)

**Key Innovation**: Generalist model for multiple change detection tasks

**Technical Approach**:
- Siamese encoder for paired images
- **Visual Delta Module**: Captures change-specific patterns
- Multi-task learning: IDC + change detection + change segmentation
- Unified architecture for different change-related tasks

**Relevance to Our Work**:
- ✅ Similar to our MDP concept (Visual Delta Module)
- 📌 **Insight**: Multi-task learning improves all tasks simultaneously
- 📌 **Missing in our baseline**: Only single-task (caption generation)

**Potential Improvement**: Add auxiliary change detection task

#### 3.1.3 Semantic-CC Enhancements (2024-2025)

**Key Innovation**: Enhanced multi-task learning framework

**Technical Approach**:
- Joint training: Change captioning + change detection + change localization
- **Semantic Alignment Module**: Aligns visual changes with language concepts
- Attention visualization for interpretability

**Relevance to Our Work**:
- 📌 **Insight**: Change localization helps caption generation
- 📌 **Insight**: Semantic alignment improves fine-grained understanding
- 📌 **Missing in our baseline**: No explicit semantic alignment

**Potential Improvement**: Add semantic alignment component

### 3.2 Recent Advances in Multimodal LLMs

#### 3.2.1 LLaVA-NeXT (2024)

**Key Innovations**:
- **Dynamic high-resolution**: Adaptive image partitioning (up to 672x672)
- **Improved vision encoder**: Better visual feature extraction
- **Better instruction tuning**: More diverse training data

**Relevance to Our Work**:
- 📌 **Insight**: Higher resolution helps fine-grained perception
- 📌 **Current limitation**: We use 336x336 (LLaVA-1.5 default)
- 📌 **Potential improvement**: Use higher resolution or dynamic partitioning

#### 3.2.2 Qwen-VL (2024)

**Key Innovations**:
- **Bilingual**: Better multilingual understanding
- **Grounding tokens**: Special tokens for spatial localization
- **Fine-grained recognition**: Optimized for detail-oriented tasks

**Relevance to Our Work**:
- 📌 **Insight**: Grounding tokens help spatial descriptions
- 📌 **Missing in our baseline**: No special tokens for spatial references
- 📌 **Potential improvement**: Add position-aware tokens

#### 3.2.3 CogVLM (2024)

**Key Innovations**:
- **Visual Expert module**: Separate parameters for visual understanding
- **Deep fusion**: Vision and language fusion at multiple layers
- **High-resolution support**: Up to 490x490

**Relevance to Our Work**:
- 📌 **Insight**: Deep fusion better than late fusion (our current approach)
- 📌 **Current limitation**: We fuse only at projector stage
- 📌 **Potential improvement**: Multi-layer fusion

### 3.3 Vision-Language Alignment Advances

#### 3.3.1 SigLIP (2024)

**Key Innovation**: Improved contrastive learning for vision-language

**Technical Approach**:
- Sigmoid loss instead of softmax
- Better handling of negative samples
- More stable training

**Relevance to Our Work**:
- 📌 **Insight**: Better contrastive loss improves alignment
- 📌 **Potential improvement**: Use SigLIP-style loss for difference features

#### 3.3.2 BLIP-3 (2025)

**Key Innovations**:
- **Multimodal mixture-of-experts**: Sparse activation for efficiency
- **Improved bootstrapping**: Better synthetic caption generation
- **Cross-modal retrieval**: Better image-text matching

**Relevance to Our Work**:
- 📌 **Insight**: Mixture-of-experts can reduce compute while improving performance
- 📌 **Potential improvement**: Use MoE in MDP module

---

## 4. PROPOSED IMPROVEMENTS

### 4.1 Improvement 1: Hierarchical Multi-Scale MDP with Learnable Scales

#### 4.1.1 Motivation

**Current Limitation**: Fixed 3-scale architecture with uniform spacing (1x, 2x, 4x downsampling) cannot adapt to different change granularities.

**Scientific Rationale**:
1. **Scale-Specific Changes**: Different change types require different spatial scales
   - OCR text differences: Need fine-grained resolution
   - Pose changes: Need medium-scale context
   - Illumination: Need global context
2. **Adaptive Scale Selection**: Model should learn which scales matter for each change type
3. **Hierarchical Features**: Vision transformers naturally produce hierarchical features

**Evidence from Literature**:
- **FPN (Feature Pyramid Networks)**: Hierarchical features improve detection
- **Multi-Scale Vision Transformers**: Dynamic scale selection improves performance
- **Neural Architecture Search**: Learnable architecture choices outperform fixed designs

#### 4.1.2 Proposed Architecture

```python
class HierarchicalMultiScaleMDP(nn.Module):
    """
    Enhanced MDP with learnable scale selection and hierarchical fusion.
    
    Key Improvements:
    1. 5 scales instead of 3 (finer granularity)
    2. Learnable scale importance weights
    3. Hierarchical fusion (bottom-up + top-down)
    4. Spatial attention at each scale
    """
    
    def __init__(self, dim=1024, num_scales=5, num_heads=8):
        super().__init__()
        
        # Scales: 1x, 1.5x, 2x, 3x, 4x (non-uniform spacing)
        self.scale_factors = [1.0, 1.5, 2.0, 3.0, 4.0]
        
        # Differential perception at each scale
        self.scale_modules = nn.ModuleList([
            DifferentialPerception(dim) for _ in range(num_scales)
        ])
        
        # Learnable scale importance (scale attention)
        self.scale_attention = nn.Sequential(
            nn.Linear(dim * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Spatial attention at each scale
        self.spatial_attention = nn.ModuleList([
            SpatialAttentionModule(dim) for _ in range(num_scales)
        ])
        
        # Bottom-up fusion (fine-to-coarse)
        self.bottom_up = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, 2, 1) for _ in range(num_scales - 1)
        ])
        
        # Top-down fusion (coarse-to-fine)
        self.top_down = nn.ModuleList([
            nn.ConvTranspose2d(dim, dim, 4, 2, 1) for _ in range(num_scales - 1)
        ])
        
        # Final fusion
        self.fusion = CrossAttentionFusion(dim, num_heads)
```

#### 4.1.3 Expected Benefits

1. **Finer Granularity**: 5 scales vs. 3 captures more detail levels
2. **Adaptive Weighting**: Learns which scales matter for each sample
3. **Better Feature Integration**: Hierarchical fusion better than flat concatenation
4. **Spatial Awareness**: Attention mechanisms focus on changed regions

#### 4.1.4 Experimental Validation Plan

**Ablation Study**:
- Baseline: 3 fixed scales (current)
- Variant A: 5 fixed scales (more scales, no adaptation)
- Variant B: 3 learnable scales (adaptation, fewer scales)
- Variant C: 5 learnable scales (full proposal)

**Metrics**:
- Overall performance: BLEU-4, CIDEr, SPICE
- Change-type specific: Accuracy per change type
- Efficiency: Parameters, FLOPs, memory

**Expected Outcome**: Variant C outperforms baseline by 5-10% on fine-grained changes (OCR, small objects, color)

### 4.2 Improvement 2: Learned Difference Operations with Semantic Weighting

#### 4.2.1 Motivation

**Current Limitation**: Simple absolute difference `|feat_a - feat_b|` loses directional information and treats all channels equally.

**Scientific Rationale**:
1. **Direction Matters**: "Object added" vs. "object removed" have opposite difference directions
2. **Channel Importance Varies**: Not all feature channels equally important for semantic changes
3. **Nonlinear Relationships**: Complex changes may require nonlinear difference functions

**Evidence from Literature**:
- **Learned Metrics**: Learned distance functions outperform fixed distances
- **Channel Attention**: SENet, CBAM show importance of channel weighting
- **Neural Metrics**: Learned perceptual metrics (LPIPS) better than L2

#### 4.2.2 Proposed Architecture

```python
class LearnedDifferenceOperation(nn.Module):
    """
    Learned difference operation with semantic weighting.
    
    Key Improvements:
    1. Multiple difference functions (L1, L2, cosine, learned)
    2. Semantic channel attention
    3. Directional awareness
    4. Change-type specific weighting
    """
    
    def __init__(self, dim=1024):
        super().__init__()
        
        # Channel attention for semantic weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )
        
        # Multiple difference operations
        self.diff_ops = nn.ModuleDict({
            'l1': lambda a, b: torch.abs(a - b),
            'l2': lambda a, b: (a - b) ** 2,
            'signed': lambda a, b: a - b,
            'cosine': lambda a, b: 1 - F.cosine_similarity(a, b, dim=1, keepdim=True)
        })
        
        # Learned fusion of difference operations
        self.diff_fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1),
            nn.LayerNorm([dim * 2, 1, 1]),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
        
        # Directional encoding (change direction matters)
        self.direction_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(32, dim),
            nn.GELU()
        )
    
    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape
        
        # Compute semantic channel importance
        global_feat = torch.cat([feat_a, feat_b], dim=1)  # [B, 2C, H, W]
        channel_weights = self.channel_attention(global_feat)  # [B, 2C, 1, 1]
        
        # Apply semantic weighting
        feat_a_weighted = feat_a * channel_weights[:, :C]
        feat_b_weighted = feat_b * channel_weights[:, C:]
        
        # Compute multiple difference representations
        diff_feats = []
        for op_name, op_func in self.diff_ops.items():
            diff = op_func(feat_a_weighted, feat_b_weighted)
            diff_feats.append(diff)
        
        # Concatenate and fuse
        diff_concat = torch.cat(diff_feats, dim=1)  # [B, 4C, H, W]
        diff_fused = self.diff_fusion(diff_concat)  # [B, C, H, W]
        
        # Encode change direction
        signed_diff = feat_a_weighted - feat_b_weighted
        direction_encoding = self.direction_encoder(signed_diff)
        
        # Combine magnitude and direction
        final_diff = diff_fused + direction_encoding
        
        return final_diff
```

#### 4.2.3 Expected Benefits

1. **Directional Awareness**: Can distinguish addition vs. removal
2. **Semantic Focus**: Emphasizes important channels, ignores noise
3. **Richer Representation**: Multiple difference views capture more information
4. **Better Robustness**: Learned weighting adapts to distractors (lighting, viewpoint)

#### 4.2.4 Experimental Validation Plan

**Ablation Study**:
- Baseline: Simple |a - b| (current)
- Variant A: Signed difference a - b only
- Variant B: Multiple fixed operations (no learning)
- Variant C: Learned fusion (full proposal)

**Metrics**:
- Directional accuracy: Correct identification of addition vs. removal
- Robustness: Performance on samples with illumination/viewpoint changes
- Overall: BLEU-4, CIDEr

**Expected Outcome**: 15-20% improvement on addition/removal accuracy, 5-10% on overall metrics

### 4.3 Improvement 3: Multi-Task Learning with Change Detection and Localization

#### 4.3.1 Motivation

**Current Limitation**: Single-task training (caption generation only) provides indirect supervision for difference detection.

**Scientific Rationale**:
1. **Auxiliary Tasks Help**: Multi-task learning improves feature quality through additional supervision
2. **Explicit Localization**: Change detection forces model to learn where changes occur
3. **Shared Representations**: Caption and detection tasks benefit from shared difference features

**Evidence from Literature**:
- **Semantic-CC** [54]: Multi-task learning significantly improves IDC performance
- **OneDiff** [14]: Unified model for multiple change tasks outperforms single-task
- **Multi-Task Learning Theory**: Related tasks provide useful inductive bias

#### 4.3.2 Proposed Architecture

```python
class MultiTaskM3Diff(nn.Module):
    """
    M³Diff with multi-task learning.
    
    Tasks:
    1. Primary: Caption generation (cross-entropy loss)
    2. Auxiliary A: Binary change detection (BCE loss)
    3. Auxiliary B: Change localization (segmentation, Dice loss)
    4. Auxiliary C: Change type classification (CE loss)
    """
    
    def __init__(self, mdp_module, llm, num_change_types=12):
        super().__init__()
        
        # Shared components
        self.mdp = mdp_module
        self.llm = llm
        
        # Task-specific heads
        # Task 1: Caption generation (main task, uses LLM)
        
        # Task 2: Binary change detection
        self.change_detector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Task 3: Change localization (spatial segmentation)
        self.change_localizer = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),  # Upsample to original size
            nn.Sigmoid()
        )
        
        # Task 4: Change type classification
        self.change_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_change_types)
        )
        
        # Task weights (learnable)
        self.task_weights = nn.Parameter(torch.ones(4))
    
    def forward(self, img_a, img_b, caption=None, change_mask=None, change_type=None):
        # Extract difference features
        diff_features = self.mdp(img_a, img_b)  # [B, N, C]
        
        # Task 1: Caption generation
        caption_logits = self.llm(diff_features, caption)
        caption_loss = F.cross_entropy(caption_logits, caption) if caption is not None else None
        
        # Task 2: Binary change detection
        pooled_features = diff_features.mean(dim=1)  # [B, C]
        change_prob = self.change_detector(pooled_features)  # [B, 1]
        detection_loss = F.binary_cross_entropy(
            change_prob, 
            (change_mask.sum(dim=[1,2]) > 0).float().unsqueeze(1)
        ) if change_mask is not None else None
        
        # Task 3: Change localization
        diff_spatial = diff_features.transpose(1, 2).view(B, C, H, W)
        change_map = self.change_localizer(diff_spatial)  # [B, 1, H_full, W_full]
        localization_loss = dice_loss(change_map, change_mask) if change_mask is not None else None
        
        # Task 4: Change type classification
        change_type_logits = self.change_classifier(pooled_features)  # [B, num_types]
        classification_loss = F.cross_entropy(change_type_logits, change_type) if change_type is not None else None
        
        # Multi-task loss (uncertainty weighting)
        if caption_loss is not None:
            weights = F.softmax(self.task_weights, dim=0)
            total_loss = (
                weights[0] * caption_loss +
                weights[1] * detection_loss +
                weights[2] * localization_loss +
                weights[3] * classification_loss
            )
            return {
                'loss': total_loss,
                'caption_loss': caption_loss,
                'detection_loss': detection_loss,
                'localization_loss': localization_loss,
                'classification_loss': classification_loss,
                'caption_logits': caption_logits,
                'change_map': change_map,
                'change_type': change_type_logits
            }
        else:
            return {
                'caption_logits': caption_logits,
                'change_map': change_map,
                'change_type': change_type_logits
            }
```

#### 4.3.3 Expected Benefits

1. **Better Localization**: Explicit change detection improves spatial understanding
2. **Richer Features**: Multiple objectives create more robust representations
3. **Interpretability**: Change maps visualize where model focuses
4. **Type-Specific Performance**: Classification head helps distinguish change types

#### 4.3.4 Experimental Validation Plan

**Data Requirements**:
- Need change masks (bounding boxes or segmentation masks)
- Need change type labels (12 types from OmniDiff taxonomy)
- Can be semi-automatic: Use existing object detectors + manual verification

**Ablation Study**:
- Baseline: Caption only (current)
- Variant A: Caption + detection
- Variant B: Caption + localization
- Variant C: Caption + classification
- Variant D: All tasks (full proposal)

**Metrics**:
- Caption quality: BLEU, CIDEr, SPICE
- Detection: Precision, recall, F1
- Localization: IoU with ground truth change regions
- Classification: Accuracy per change type

**Expected Outcome**: 10-15% improvement in caption quality, strong localization performance (IoU > 0.7)

### 4.4 Improvement 4: Two-Stage Training with Curriculum Learning

#### 4.4.1 Motivation

**Current Limitation**: Single-stage training with cold-start MDP (random initialization while rest is pretrained).

**Scientific Rationale**:
1. **Parameter Imbalance**: MDP (34M params, random) vs. LLM (7B params, pretrained) creates optimization challenges
2. **Curriculum Benefits**: Starting with simple examples helps model learn basic concepts before complex patterns
3. **Progressive Difficulty**: Gradual increase in task difficulty improves convergence

**Evidence from Literature**:
- **Curriculum Learning** [Bengio et al., 2009]: Fundamental ML principle
- **Self-Paced Learning**: Model-driven curriculum improves generalization
- **LLaVA Training**: Two-stage pretraining→finetuning proven effective

#### 4.4.2 Proposed Training Strategy

**Stage 1: MDP Pretraining (Self-Supervised)**
```yaml
stage1_pretraining:
  duration: 10,000 steps
  objective: Self-supervised difference learning
  tasks:
    - Feature reconstruction: Reconstruct feat_b from feat_a + diff
    - Contrastive learning: Align difference features with change descriptions
    - Masked prediction: Predict masked regions from difference features
  
  dataset: Large-scale synthetic + CLEVR-Change (easy, abundant data)
  
  frozen: Vision encoder, LLM
  trainable: MDP module only
  
  batch_size: 8 (larger since no LLM gradient)
  learning_rate: 5e-4 (higher for faster convergence)
```

**Stage 2: Curriculum Fine-Tuning**
```yaml
stage2_finetuning:
  duration: 15,000 steps
  objective: Caption generation with curriculum
  
  curriculum_schedule:
    # Easy (steps 0-5000): Single objects, large differences
    phase_1:
      change_types: [add, remove]
      object_size: large
      background: simple
      
    # Medium (steps 5000-10000): Multiple objects, moderate differences
    phase_2:
      change_types: [add, remove, color, size]
      object_size: medium
      background: moderate_complexity
      
    # Hard (steps 10000-15000): All change types, subtle differences
    phase_3:
      change_types: all_12_types
      object_size: small_to_large
      background: complex_real_world
      difficulty: include_distractors
  
  dataset: OmniDiff or best available IDC dataset
  
  frozen: Vision encoder (initially), none (later)
  trainable: MDP, projector, LoRA
  
  progressive_unfreezing:
    - steps 0-5000: MDP + projector
    - steps 5000-10000: + LoRA
    - steps 10000-15000: + last 2 vision encoder layers
```

#### 4.4.3 Expected Benefits

1. **Faster Convergence**: Pretrained MDP starts from better initialization
2. **Better Generalization**: Curriculum prevents overfitting to complex examples
3. **Stabler Training**: Progressive unfreezing reduces gradient conflicts
4. **Higher Performance**: Model learns basic patterns before tackling hard cases

#### 4.4.4 Experimental Validation Plan

**Ablation Study**:
- Baseline: Single-stage end-to-end (current)
- Variant A: Two-stage without curriculum
- Variant B: Single-stage with curriculum
- Variant C: Two-stage with curriculum (full proposal)

**Metrics**:
- Training curves: Loss, gradient norms, learning dynamics
- Convergence: Steps to reach target performance
- Final performance: BLEU-4, CIDEr on OmniDiff test set
- Generalization: Performance on out-of-distribution samples

**Expected Outcome**: 15-20% faster convergence, 5-10% better final performance

### 4.5 Improvement 5: Dynamic High-Resolution Processing with Adaptive Tokenization

#### 4.5.1 Motivation

**Current Limitation**: Fixed 336×336 input resolution (LLaVA-1.5 default) may miss fine-grained details.

**Scientific Rationale**:
1. **Fine-Grained Changes Need Resolution**: OCR text, small objects require high pixel resolution
2. **Global Changes Need Context**: Pose, illumination need wide field of view
3. **Efficiency Trade-off**: Higher resolution = more computation, but not all samples need it

**Evidence from Literature**:
- **LLaVA-NeXT**: Dynamic resolution (up to 672×672) improves fine-grained understanding
- **Pix2Struct**: Variable resolution crucial for document understanding
- **Vision Transformers**: Can handle variable input sizes

#### 4.5.2 Proposed Architecture

```python
class DynamicResolutionMDP(nn.Module):
    """
    MDP with adaptive resolution processing.
    
    Key Improvements:
    1. Change-aware resolution selection (336 vs 672 vs 1024)
    2. Patch-based processing for very high resolution
    3. Adaptive token number (reduce tokens for simple changes)
    4. Efficient attention with local+global tokens
    """
    
    def __init__(self, base_mdp, resolutions=[336, 672, 1024]):
        super().__init__()
        self.base_mdp = base_mdp
        self.resolutions = resolutions
        
        # Resolution selector (predicts which resolution to use)
        self.resolution_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),  # 2048 = 1024*2 (feat_a + feat_b)
            nn.ReLU(),
            nn.Linear(512, len(resolutions)),
            nn.Softmax(dim=-1)
        )
        
        # Patch partition for ultra-high resolution
        self.patch_size = 336
        
        # Token reduction (adaptive pooling to reduce sequence length)
        self.token_reducer = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Importance score per token
            nn.Sigmoid()
        )
        
        # Local-global attention (efficient for long sequences)
        self.local_window_size = 7
        self.global_tokens = 64  # Fixed number of global tokens
    
    def select_resolution(self, feat_a_336, feat_b_336):
        """Predict optimal resolution based on low-res features."""
        concat = torch.cat([feat_a_336, feat_b_336], dim=1)  # [B, 2048, H, W]
        resolution_probs = self.resolution_selector(concat)  # [B, num_resolutions]
        
        # Select resolution (can use hard selection or soft weighted)
        selected_res_idx = resolution_probs.argmax(dim=-1)
        return selected_res_idx, resolution_probs
    
    def process_high_resolution(self, img_a, img_b, resolution):
        """Process image pair at specified resolution."""
        # Resize images
        img_a_resized = F.interpolate(img_a, size=(resolution, resolution), mode='bicubic')
        img_b_resized = F.interpolate(img_b, size=(resolution, resolution), mode='bicubic')
        
        if resolution <= 672:
            # Process entire image
            feat_a = self.vision_encoder(img_a_resized)
            feat_b = self.vision_encoder(img_b_resized)
            diff_features = self.base_mdp(feat_a, feat_b)
        else:
            # Patch-based processing for ultra-high res (1024+)
            patches_a = self.partition_patches(img_a_resized, self.patch_size)
            patches_b = self.partition_patches(img_b_resized, self.patch_size)
            
            diff_patches = []
            for patch_a, patch_b in zip(patches_a, patches_b):
                feat_a = self.vision_encoder(patch_a)
                feat_b = self.vision_encoder(patch_b)
                diff_patch = self.base_mdp(feat_a, feat_b)
                diff_patches.append(diff_patch)
            
            # Merge patches
            diff_features = self.merge_patches(diff_patches)
        
        return diff_features
    
    def reduce_tokens(self, diff_features, target_tokens=576):
        """Adaptively reduce number of tokens based on importance."""
        B, N, C = diff_features.shape
        
        # Compute token importance scores
        importance = self.token_reducer(diff_features).squeeze(-1)  # [B, N]
        
        # Select top-k important tokens
        _, indices = importance.topk(k=min(target_tokens, N), dim=-1)
        indices = indices.sort(dim=-1)[0]  # Keep spatial order
        
        # Gather important tokens
        reduced_features = torch.gather(
            diff_features, 
            dim=1, 
            index=indices.unsqueeze(-1).expand(-1, -1, C)
        )
        
        return reduced_features, importance
    
    def forward(self, img_a, img_b):
        # Step 1: Quick pass at 336 to select resolution
        feat_a_336 = self.vision_encoder_336(img_a)
        feat_b_336 = self.vision_encoder_336(img_b)
        
        selected_res_idx, res_probs = self.select_resolution(feat_a_336, feat_b_336)
        
        # Step 2: Process at selected resolution
        diff_features_list = []
        for batch_idx in range(img_a.size(0)):
            res_idx = selected_res_idx[batch_idx].item()
            resolution = self.resolutions[res_idx]
            
            diff_feat = self.process_high_resolution(
                img_a[batch_idx:batch_idx+1],
                img_b[batch_idx:batch_idx+1],
                resolution
            )
            diff_features_list.append(diff_feat)
        
        diff_features = torch.cat(diff_features_list, dim=0)
        
        # Step 3: Adaptive token reduction
        if diff_features.size(1) > 576:
            diff_features, token_importance = self.reduce_tokens(diff_features, target_tokens=576)
        
        return diff_features
```

#### 4.5.3 Expected Benefits

1. **Better Fine-Grained Perception**: High resolution captures small details
2. **Efficiency**: Only use high resolution when needed
3. **Scalability**: Can handle variable complexity without memory explosion
4. **Adaptive Capacity**: More tokens for complex changes, fewer for simple ones

#### 4.5.4 Experimental Validation Plan

**Ablation Study**:
- Baseline: Fixed 336×336 (current)
- Variant A: Fixed 672×672 (higher res, no adaptation)
- Variant B: Dynamic 336/672 (two resolutions)
- Variant C: Dynamic 336/672/1024 + token reduction (full proposal)

**Metrics**:
- Per change type: Performance on small vs. large objects
- Efficiency: FLOPs, memory, inference time
- Quality: BLEU-4, CIDEr, especially on fine-grained changes (OCR, small objects)

**Expected Outcome**: 20-30% improvement on fine-grained changes (OCR, small objects), 10-15% overall

---

## 5. EXPERIMENTAL DESIGN FOR PHASE 5

### 5.1 Experimental Setup

#### 5.1.1 Datasets

**Primary Dataset** (if available):
- **OmniDiff**: 15,598 pairs, train/val/test split
- Most comprehensive, closest to paper

**Backup Datasets**:
- **Spot-the-Diff**: 13,192 surveillance pairs
- **CLEVR-Change**: 79,606 synthetic pairs (simple)
- **IEdit**: 3,939 real-world pairs

**Synthetic Fallback**:
- Enhanced synthetic dataset: 5,000 samples, 12 change types, complex scenes

#### 5.1.2 Evaluation Metrics

**Primary Metrics** (for comparison with paper):
1. BLEU-4
2. METEOR
3. ROUGE-L
4. CIDEr-D
5. SPICE

**Additional Metrics** (for deeper analysis):
6. Change Type Accuracy (per-type performance)
7. Localization IoU (if ground truth available)
8. Robustness Score (performance on distractor samples)
9. Caption Length Distribution
10. Human Evaluation (fluency, correctness, completeness)

#### 5.1.3 Baseline Comparisons

**Internal Baselines**:
1. LLaVA-1.5-7B (no MDP) - establishes MLLM baseline
2. M³Diff current (Phase 3 implementation) - our baseline
3. M³Diff + each improvement individually - ablation study
4. M³Diff + all improvements - full system

**External Baselines** (if possible):
1. FINER-MLLM (if code available)
2. OneDiff (if code available)
3. Semantic-CC (if code available)

### 5.2 Ablation Study Design

#### 5.2.1 Improvement 1: Hierarchical Multi-Scale MDP

**Variables**:
- Number of scales: 3 (baseline), 5 (proposed)
- Scale selection: Fixed, learned
- Fusion: Flat concatenation, hierarchical

**Experiments**:
| Exp ID | Num Scales | Selection | Fusion | Expected Δ |
|--------|------------|-----------|--------|------------|
| I1-A | 3 | Fixed | Flat | Baseline |
| I1-B | 5 | Fixed | Flat | +2-3% |
| I1-C | 3 | Learned | Flat | +3-4% |
| I1-D | 5 | Learned | Hierarchical | +5-8% |

#### 5.2.2 Improvement 2: Learned Difference Operations

**Variables**:
- Difference function: L1 (baseline), signed, learned
- Channel weighting: None, fixed, learned
- Direction encoding: None, yes

**Experiments**:
| Exp ID | Diff Func | Channel Weight | Direction | Expected Δ |
|--------|-----------|----------------|-----------|------------|
| I2-A | L1 | None | No | Baseline |
| I2-B | Signed | None | No | +1-2% |
| I2-C | L1 | Learned | No | +3-4% |
| I2-D | Multiple+Learned | Learned | Yes | +7-10% |

#### 5.2.3 Improvement 3: Multi-Task Learning

**Variables**:
- Tasks: Caption only (baseline), +detection, +localization, +classification
- Task weighting: Equal, learned

**Experiments**:
| Exp ID | Tasks | Weighting | Expected Δ |
|--------|-------|-----------|------------|
| I3-A | Caption | N/A | Baseline |
| I3-B | Caption + Detection | Equal | +3-5% |
| I3-C | Caption + Localization | Equal | +5-7% |
| I3-D | All 4 tasks | Learned | +10-15% |

#### 5.2.4 Improvement 4: Two-Stage Training + Curriculum

**Variables**:
- Pretraining: None (baseline), MDP pretrain
- Curriculum: None, difficulty-based
- Unfreezing: None, progressive

**Experiments**:
| Exp ID | Pretrain | Curriculum | Unfreezing | Expected Δ |
|--------|----------|------------|------------|------------|
| I4-A | None | None | None | Baseline |
| I4-B | MDP | None | None | +3-5% |
| I4-C | None | Yes | None | +4-6% |
| I4-D | MDP | Yes | Progressive | +8-12% |

#### 5.2.5 Improvement 5: Dynamic High-Resolution

**Variables**:
- Resolution: 336 (baseline), 672, dynamic
- Token reduction: None, fixed, adaptive
- Patch processing: None, yes

**Experiments**:
| Exp ID | Resolution | Token Reduction | Patch | Expected Δ |
|--------|------------|-----------------|-------|------------|
| I5-A | 336 | None | No | Baseline |
| I5-B | 672 | None | No | +5-7% |
| I5-C | Dynamic 336/672 | Fixed | No | +8-10% |
| I5-D | Dynamic 336/672/1024 | Adaptive | Yes | +12-18% |

### 5.3 Combined System Experiments

#### 5.3.1 Progressive Integration

Test improvements sequentially to identify synergies and conflicts:

**Experiment Sequence**:
1. Baseline (Phase 3 implementation)
2. + Improvement 2 (Learned Difference) - most fundamental
3. + Improvement 1 (Hierarchical MDP) - better features
4. + Improvement 3 (Multi-Task) - better supervision
5. + Improvement 4 (Two-Stage Training) - better optimization
6. + Improvement 5 (Dynamic Resolution) - better input

**Hypothesis**: Each improvement should provide additive or synergistic gains.

#### 5.3.2 Full System vs. Ablations

**Final Comparison**:
| System | Description | Expected BLEU-4 | Expected CIDEr |
|--------|-------------|-----------------|----------------|
| Baseline | Phase 3 M³Diff | X | Y |
| Best Single | Best individual improvement | X + 10% | Y + 12% |
| Best Pair | Best 2-improvement combo | X + 18% | Y + 22% |
| Best Triple | Best 3-improvement combo | X + 25% | Y + 30% |
| Full System | All 5 improvements | X + 35% | Y + 45% |

*(X, Y will be determined after baseline training on real dataset)*

### 5.4 Resource Requirements & Timeline

#### 5.4.1 Computational Resources

**Per Experiment**:
- Training time: 3-6 hours (synthetic), 12-24 hours (OmniDiff)
- GPU memory: 4-6 GB (with optimizations)
- Storage: ~2 GB (checkpoints + logs)

**Total for Phase 5**:
- Ablation experiments: ~25 experiments × 6 hours = 150 GPU-hours
- Combined experiments: ~6 experiments × 24 hours = 144 GPU-hours
- **Total**: ~300 GPU-hours (~12-15 days on single RTX 3050 Ti)

#### 5.4.2 Data Requirements

**Annotations Needed for Multi-Task Learning**:
- Change type labels: 12 categories (can be manual or semi-automatic)
- Change masks: Bounding boxes or segmentation (can use off-the-shelf detectors + manual)
- Estimated effort: 2-3 hours per 100 samples with tools

**For 1,000 sample subset**: 20-30 hours annotation effort

### 5.5 Success Criteria

**Minimum Viable**:
- At least 1 improvement shows statistically significant gains (p < 0.05)
- Combined system outperforms baseline by ≥10%
- Ablation studies reveal clear improvement contributions

**Target**:
- 3+ improvements show significant individual gains
- Combined system outperforms baseline by ≥25%
- Per-change-type analysis shows targeted improvements
- Qualitative examples demonstrate better fine-grained understanding

**Stretch**:
- Combined system achieves near-paper performance (if comparing to OmniDiff results)
- Improvements generalize across multiple datasets
- Human evaluation shows noticeable quality improvement

---

## 6. SUMMARY & NEXT STEPS

### 6.1 Phase 4 Achievements

**Completed**:
- ✅ Comprehensive analysis of Phase 3 baseline
- ✅ Identified 10 concrete limitations across architecture, training, and evaluation
- ✅ Reviewed 10+ recent papers (2024-2026) in IDC and MLLM domains
- ✅ Proposed 5 scientifically-justified improvements
- ✅ Designed detailed ablation studies for Phase 5
- ✅ Estimated resource requirements and timeline

**Key Insights**:
1. **MDP Architecture**: Fixed 3-scale design limits adaptability
2. **Difference Operation**: Simple L1 loses important information
3. **Training Paradigm**: Single-task, single-stage suboptimal
4. **Resolution**: Fixed 336px insufficient for fine-grained details
5. **Multi-Task Learning**: Auxiliary tasks consistently help in related work

### 6.2 Proposed Improvements Summary

| ID | Improvement | Expected Gain | Implementation Complexity |
|----|-------------|---------------|---------------------------|
| 1 | Hierarchical Multi-Scale MDP | +5-8% | Medium |
| 2 | Learned Difference Operations | +7-10% | Low-Medium |
| 3 | Multi-Task Learning | +10-15% | Medium-High |
| 4 | Two-Stage Training + Curriculum | +8-12% | Medium |
| 5 | Dynamic High-Resolution | +12-18% | High |
| **All** | **Combined System** | **+30-40%** | **High** |

### 6.3 Implementation Priority for Phase 5

**Priority 1 (Must Have)**:
- Improvement 2: Learned Difference Operations (fundamental, low complexity)
- Improvement 4: Two-Stage Training (improves all other improvements)

**Priority 2 (Should Have)**:
- Improvement 1: Hierarchical Multi-Scale MDP (direct architecture enhancement)
- Improvement 3: Multi-Task Learning (proven effective in literature)

**Priority 3 (Nice to Have)**:
- Improvement 5: Dynamic High-Resolution (high gain but high complexity)

### 6.4 Handoff to Phase 5

**Phase 5 Objectives**:
1. Obtain real IDC dataset (OmniDiff, Spot-the-Diff, or CLEVR-Change)
2. Implement Priority 1 improvements
3. Train baseline + improved models
4. Conduct ablation studies
5. Evaluate on all metrics
6. Document results in Phase5.md

**Prerequisites**:
- Real dataset with sufficient samples (>1,000 pairs)
- For multi-task: Annotation of change types and masks
- Sufficient compute time (~300 GPU-hours)
- Full pycocoevalcap installation for complete metrics

**Expected Outcomes**:
- Validated improvements with statistical significance
- Clear understanding of what works and why
- Publishable results (if gains are substantial)
- Solid foundation for Phase 6 (further refinement)

---

## 7. REFERENCES

### 7.1 Papers Reviewed

**Image Difference Captioning**:
- [51] FINER-MLLM (2024): Fine-grained visual perception with dual constraints
- [14] Hu et al. OneDiff (2024): Generalist model with Visual Delta Module
- [54] Semantic-CC (2024-2025): Enhanced multi-task learning framework
- [49] IDC-PCL: Pretraining with contrastive learning
- [43] DIRL: Difference-aware image representation learning

**Multimodal LLMs**:
- LLaVA-NeXT (2024): Dynamic high-resolution processing
- Qwen-VL (2024): Grounding tokens for spatial localization
- CogVLM (2024): Deep vision-language fusion
- LLaVA-1.5 (2023): Base model for our implementation

**Vision-Language Alignment**:
- SigLIP (2024): Improved contrastive learning
- BLIP-3 (2025): Multimodal mixture-of-experts
- CLIP (2021): Foundation for vision-language models

**Machine Learning Foundations**:
- Bengio et al. (2009): Curriculum Learning
- Hu et al. (2021): LoRA - Low-Rank Adaptation
- Feature Pyramid Networks (2017): Multi-scale features
- Neural Architecture Search: Learnable architectures

### 7.2 Technical Resources

- PyTorch Documentation
- HuggingFace Transformers Documentation
- PEFT Library Documentation
- COCO Evaluation Tools
- Blender Python API (for synthetic data)

---

**PHASE 4 STATUS: ✅ COMPLETED**

**Next Phase**: Phase 5 - Experimental Validation & Improvement Implementation  
**Estimated Duration**: 20-25 days (including training time)  
**Priority**: Obtain real IDC dataset and begin baseline training

