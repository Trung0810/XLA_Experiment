"""
Create Enhanced Synthetic IDC Dataset

This script generates a more realistic synthetic Image Difference Captioning
dataset to replace the simple 50-sample geometric dataset.

Enhancements over simple synthetic:
- 1,000+ samples (20x larger)
- 12 change types (vs 4): add, remove, color, position, size, orientation, 
  viewpoint, illumination, substitution, pose, OCR, counting
- Complex scenes with multiple objects
- Realistic textures and backgrounds
- Longer captions (30-50 words vs 10-15)

Phase 5 - Experimental Validation
