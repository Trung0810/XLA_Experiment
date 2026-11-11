"""
Evaluation metrics for Image Difference Captioning.

Implements 5 standard metrics:
- BLEU (1-4)
- METEOR
- ROUGE-L
- CIDEr
- SPICE

Author: OmniDiff Reproduction Project
Date: 2026-11-11
"""

import torch
from typing import List, Dict, Tuple
import re
from collections import Counter
import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase + split by non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


# ==================== BLEU ====================

def compute_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU-1 to BLEU-4 scores.
    
    BLEU = BP * exp(sum(log(p_n) for n in 1..N) / N)
    where p_n is modified n-gram precision
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        max_n: Maximum n-gram order (default: 4)
        
    Returns:
        scores: Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4
    """
    scores = {}
    
    for n in range(1, max_n + 1):
        precisions = []
        total_pred_len = 0
        total_ref_len = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize(pred)
            ref_tokens = tokenize(ref)
            
            # Get n-grams
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            # Count matches (clipped)
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
            
            # Precision
            total_pred = sum(pred_ngrams.values())
            if total_pred > 0:
                precisions.append(matches / total_pred)
            else:
                precisions.append(0.0)
            
            total_pred_len += len(pred_tokens)
            total_ref_len += len(ref_tokens)
        
        # Brevity penalty
        if total_pred_len > total_ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - total_ref_len / (total_pred_len + 1e-10))
        
        # Geometric mean of precisions
        if precisions and all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean([np.log(p + 1e-10) for p in precisions]))
        else:
            geo_mean = 0.0
        
        scores[f'BLEU-{n}'] = bp * geo_mean
    
    return scores


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram counts from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


# ==================== METEOR ====================

def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Compute METEOR score (simplified version).
    
    METEOR = (1-P) * F-mean
    where P is penalty for fragmentation
    F-mean is harmonic mean of precision and recall
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        
    Returns:
        score: METEOR score (0-1)
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize(pred))
        ref_tokens = set(tokenize(ref))
        
        # Matches (exact word matching)
        matches = pred_tokens & ref_tokens
        num_matches = len(matches)
        
        # Precision and Recall
        precision = num_matches / (len(pred_tokens) + 1e-10)
        recall = num_matches / (len(ref_tokens) + 1e-10)
        
        # F-mean (harmonic mean)
        if precision + recall > 0:
            fmean = (10 * precision * recall) / (recall + 9 * precision)
        else:
            fmean = 0.0
        
        # Penalty (simplified: based on length difference)
        penalty = 0.5 * abs(len(pred_tokens) - len(ref_tokens)) / (len(ref_tokens) + 1e-10)
        penalty = min(penalty, 0.5)  # Cap at 0.5
        
        meteor = (1 - penalty) * fmean
        scores.append(meteor)
    
    return np.mean(scores)


# ==================== ROUGE-L ====================

def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Compute ROUGE-L score (Longest Common Subsequence).
    
    ROUGE-L = F-measure based on LCS
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        
    Returns:
        score: ROUGE-L F1 score (0-1)
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        # Compute LCS length
        lcs_length = lcs(pred_tokens, ref_tokens)
        
        # Precision and Recall
        precision = lcs_length / (len(pred_tokens) + 1e-10)
        recall = lcs_length / (len(ref_tokens) + 1e-10)
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        scores.append(f1)
    
    return np.mean(scores)


def lcs(seq1: List[str], seq2: List[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


# ==================== CIDEr ====================

def compute_cider(predictions: List[str], references: List[str], n: int = 4) -> float:
    """
    Compute CIDEr score (Consensus-based Image Description Evaluation).
    
    CIDEr measures consensus between prediction and references using TF-IDF weighted n-grams.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        n: Maximum n-gram order (default: 4)
        
    Returns:
        score: CIDEr score
    """
    # For simplicity, we use a simplified CIDEr without TF-IDF weighting
    # Real implementation would use pycocoevalcap library
    
    scores = []
    
    # Collect document frequencies (for IDF)
    doc_freq = Counter()
    for ref in references:
        ref_tokens = tokenize(ref)
        for order in range(1, n + 1):
            ngrams = set(get_ngrams(ref_tokens, order).keys())
            doc_freq.update(ngrams)
    
    num_docs = len(references)
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        cider_n = []
        
        for order in range(1, n + 1):
            pred_ngrams = get_ngrams(pred_tokens, order)
            ref_ngrams = get_ngrams(ref_tokens, order)
            
            # TF-IDF vectors
            all_ngrams = set(pred_ngrams.keys()) | set(ref_ngrams.keys())
            
            pred_vec = []
            ref_vec = []
            
            for ngram in all_ngrams:
                # TF
                pred_tf = pred_ngrams.get(ngram, 0)
                ref_tf = ref_ngrams.get(ngram, 0)
                
                # IDF (log(N/df))
                idf = np.log(num_docs / (doc_freq.get(ngram, 0) + 1))
                
                # TF-IDF
                pred_vec.append(pred_tf * idf)
                ref_vec.append(ref_tf * idf)
            
            # Cosine similarity
            pred_vec = np.array(pred_vec)
            ref_vec = np.array(ref_vec)
            
            dot_product = np.dot(pred_vec, ref_vec)
            norm_pred = np.linalg.norm(pred_vec) + 1e-10
            norm_ref = np.linalg.norm(ref_vec) + 1e-10
            
            similarity = dot_product / (norm_pred * norm_ref)
            cider_n.append(similarity)
        
        # Average over n-gram orders
        scores.append(np.mean(cider_n))
    
    return np.mean(scores) * 10.0  # Scale to match pycocoevalcap range


# ==================== SPICE (Simplified) ====================

def compute_spice_simple(predictions: List[str], references: List[str]) -> float:
    """
    Simplified SPICE score (word overlap based).
    
    Real SPICE uses scene graph matching, but we use word-level F1 as approximation.
    For production, should use pycocoevalcap's SPICE.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions
        
    Returns:
        score: Simplified SPICE score (0-1)
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize(pred))
        ref_tokens = set(tokenize(ref))
        
        # Intersection
        common = pred_tokens & ref_tokens
        
        # Precision and Recall
        precision = len(common) / (len(pred_tokens) + 1e-10)
        recall = len(common) / (len(ref_tokens) + 1e-10)
        
        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        scores.append(f1)
    
    return np.mean(scores)


# ==================== Main Metrics Function ====================

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted captions
        references: List of reference captions (one per prediction)
        
    Returns:
        metrics: Dictionary with all metric scores
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    metrics = {}
    
    # BLEU (1-4)
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)
    
    # METEOR
    metrics['METEOR'] = compute_meteor(predictions, references)
    
    # ROUGE-L
    metrics['ROUGE-L'] = compute_rouge_l(predictions, references)
    
    # CIDEr
    metrics['CIDEr'] = compute_cider(predictions, references)
    
    # SPICE (simplified)
    metrics['SPICE'] = compute_spice_simple(predictions, references)
    
    return metrics


def test_metrics():
    """Test metrics computation with dummy data."""
    print("Testing evaluation metrics...")
    print("="*60)
    
    # Dummy predictions and references
    predictions = [
        "The red cube changed to a blue sphere.",
        "A new green cylinder was added to the scene.",
        "The metal object was removed from the table."
    ]
    
    references = [
        "The red cube became a blue sphere.",
        "A green cylinder appeared in the scene.",
        "The metallic object disappeared from the table."
    ]
    
    print("\nSample predictions and references:")
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"\n{i+1}. Prediction: {pred}")
        print(f"   Reference:  {ref}")
    
    # Compute metrics
    print("\n" + "-"*60)
    print("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    
    print("\nMetric Results:")
    print("-"*60)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:12s}: {metric_value:.4f}")
    
    print("\n" + "="*60)
    print("[PASS] Metrics test completed!")
    
    print("\nNote: These are simplified implementations.")
    print("For production evaluation, consider using:")
    print("  - pycocoevalcap (standard COCO metrics)")
    print("  - Official SPICE implementation (requires Java)")
    print("  - Official METEOR implementation")


if __name__ == "__main__":
    test_metrics()
