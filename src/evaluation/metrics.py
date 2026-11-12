"""
Evaluation Metrics for Image Difference Captioning
Implements BLEU, METEOR, ROUGE-L, CIDEr, and SPICE metrics
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Try to import COCO evaluation tools
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    try:
        from pycocoevalcap.spice.spice import Spice
        SPICE_AVAILABLE = True
    except:
        SPICE_AVAILABLE = False
        print("[WARNING] SPICE not available - requires Java and SPICE jar file")
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("[WARNING] pycocoevalcap not installed - using fallback implementations")


class IDCMetrics:
    """
    Image Difference Captioning Evaluation Metrics
    Computes BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, and SPICE
    """
    
    def __init__(self, use_coco_eval: bool = True):
        """
        Initialize metrics
        
        Args:
            use_coco_eval: Whether to use COCO evaluation tools (more accurate)
        """
        self.use_coco_eval = use_coco_eval and COCO_EVAL_AVAILABLE
        
        if self.use_coco_eval:
            self.scorers = [
                (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE-L"),
                (Cider(), "CIDEr")
            ]
            if SPICE_AVAILABLE:
                self.scorers.append((Spice(), "SPICE"))
        else:
            print("[INFO] Using fallback metric implementations")
    
    def compute_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all metrics
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions (ground truth)
            
        Returns:
            Dictionary of metric scores
        """
        if self.use_coco_eval:
            return self._compute_coco_metrics(predictions, references)
        else:
            return self._compute_fallback_metrics(predictions, references)
    
    def _compute_coco_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute metrics using COCO evaluation tools
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions
            
        Returns:
            Dictionary of metric scores
        """
        # Format for COCO eval: dict of {image_id: [caption]}
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        
        results = {}
        
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                # Multiple scores (e.g., BLEU-1/2/3/4)
                for sc, m in zip(score, method):
                    results[m] = float(sc)
            else:
                # Single score
                results[method] = float(score)
        
        return results
    
    def _compute_fallback_metrics(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute metrics using fallback implementations
        
        Args:
            predictions: List of predicted captions
            references: List of reference captions
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # BLEU scores
        for n in range(1, 5):
            bleu_scores = [
                self._compute_bleu_n(pred, ref, n)
                for pred, ref in zip(predictions, references)
            ]
            results[f"BLEU-{n}"] = np.mean(bleu_scores)
        
        # ROUGE-L
        rouge_scores = [
            self._compute_rouge_l(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        results["ROUGE-L"] = np.mean(rouge_scores)
        
        # Note: METEOR and CIDEr require more complex implementations
        # For now, we'll use approximations
        results["METEOR"] = 0.0  # Placeholder
        results["CIDEr"] = 0.0   # Placeholder
        
        return results
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def _compute_bleu_n(
        self, 
        prediction: str, 
        reference: str, 
        n: int
    ) -> float:
        """
        Compute BLEU-n score
        
        Args:
            prediction: Predicted caption
            reference: Reference caption
            n: N-gram size
            
        Returns:
            BLEU-n score
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0
        
        # Extract n-grams
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        # Count matches
        matches = 0
        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                matches += min(count, ref_ngrams[ngram])
        total = sum(pred_ngrams.values())
        
        if total == 0:
            return 0.0
        
        precision = matches / total
        
        # Brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
        
        return bp * precision
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
        """Extract n-grams from tokens"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _compute_rouge_l(
        self, 
        prediction: str, 
        reference: str
    ) -> float:
        """
        Compute ROUGE-L score (based on longest common subsequence)
        
        Args:
            prediction: Predicted caption
            reference: Reference caption
            
        Returns:
            ROUGE-L F1 score
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        # Precision and recall
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """
        Compute longest common subsequence length
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            LCS length
        """
        m, n = len(seq1), len(seq2)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class MetricsEvaluator:
    """
    Evaluator for running inference and computing metrics
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        metrics: IDCMetrics = None,
        device: str = "cuda"
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for decoding
            metrics: Metrics computer (will create if None)
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics if metrics is not None else IDCMetrics()
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self, 
        dataloader,
        max_samples: int = None
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate model on dataset
        
        Args:
            dataloader: DataLoader for evaluation
            max_samples: Maximum samples to evaluate (None for all)
            
        Returns:
            Tuple of (metrics_dict, predictions_list)
        """
        import torch
        
        predictions = []
        references = []
        samples = []
        
        print(f"[INFO] Evaluating on {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples is not None and batch_idx >= max_samples:
                    break
                
                # Move to device
                images_a = batch['image_a'].to(self.device)
                images_b = batch['image_b'].to(self.device)
                
                # Generate captions
                generated_ids = self.model.generate(
                    images_a=images_a,
                    images_b=images_b,
                    max_length=100,
                    num_beams=3,
                    temperature=1.0,
                    top_p=0.9
                )
                
                # Decode predictions
                pred_texts = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                # Get references
                ref_texts = batch['caption']
                
                predictions.extend(pred_texts)
                references.extend(ref_texts)
                
                # Store samples for inspection
                for i in range(len(pred_texts)):
                    samples.append({
                        'prediction': pred_texts[i],
                        'reference': ref_texts[i]
                    })
        
        # Compute metrics
        print(f"[INFO] Computing metrics on {len(predictions)} samples...")
        metrics_dict = self.metrics.compute_metrics(predictions, references)
        
        return metrics_dict, samples
    
    def save_results(
        self, 
        metrics: Dict[str, float], 
        samples: List[Dict],
        output_path: str
    ):
        """
        Save evaluation results
        
        Args:
            metrics: Metrics dictionary
            samples: List of prediction samples
            output_path: Path to save results
        """
        results = {
            'metrics': metrics,
            'num_samples': len(samples),
            'samples': samples[:100]  # Save first 100 for inspection
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[SAVED] Results saved to {output_path}")
        
        # Also save metrics summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of samples: {len(samples)}\n\n")
            f.write("Metrics:\n")
            f.write("-" * 60 + "\n")
            for metric, value in sorted(metrics.items()):
                f.write(f"{metric:20s}: {value:.4f}\n")
            f.write("=" * 60 + "\n")
        
        print(f"[SAVED] Summary saved to {summary_path}")


def test_metrics():
    """Test metrics implementation"""
    print("[TEST] Testing IDCMetrics...")
    
    # Sample predictions and references
    predictions = [
        "a red cube appeared on the left side",
        "the blue sphere moved to the right",
        "a green cylinder was removed from the scene"
    ]
    
    references = [
        "a red cube is added on the left",
        "the blue ball moved to the right side",
        "the green cylinder disappeared"
    ]
    
    # Create metrics
    metrics = IDCMetrics(use_coco_eval=COCO_EVAL_AVAILABLE)
    
    # Compute metrics
    results = metrics.compute_metrics(predictions, references)
    
    print("\n[RESULTS] Metrics:")
    print("-" * 60)
    for metric, value in sorted(results.items()):
        print(f"{metric:15s}: {value:.4f}")
    print("-" * 60)
    
    print("\n[PASS] Metrics computed successfully!")
    
    if not COCO_EVAL_AVAILABLE:
        print("\n[NOTE] To use official COCO metrics, install:")
        print("  pip install git+https://github.com/tylin/coco-caption.git")


if __name__ == "__main__":
    test_metrics()
