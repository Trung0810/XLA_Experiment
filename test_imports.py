#!/usr/bin/env python
"""Test imports of essential packages"""

print("Testing package imports...")
print("-" * 50)

try:
    import peft
    print("[OK] peft:", peft.__version__)
except Exception as e:
    print("[FAIL] peft:", e)

try:
    import transformers
    print("[OK] transformers:", transformers.__version__)
except Exception as e:
    print("[FAIL] transformers:", e)

try:
    import accelerate
    print("[OK] accelerate:", accelerate.__version__)
except Exception as e:
    print("[FAIL] accelerate:", e)

try:
    import datasets
    print("[OK] datasets:", datasets.__version__)
except Exception as e:
    print("[FAIL] datasets:", e)

try:
    import sentencepiece
    print("[OK] sentencepiece: available")
except Exception as e:
    print("[FAIL] sentencepiece:", e)

try:
    import torch
    print("[OK] torch:", torch.__version__)
    print("     CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("     GPU:", torch.cuda.get_device_name(0))
        print("     VRAM: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
except Exception as e:
    print("[FAIL] torch:", e)

try:
    import einops
    print("[OK] einops: available")
except Exception as e:
    print("[FAIL] einops:", e)

try:
    import bitsandbytes
    print("[OK] bitsandbytes: available")
except Exception as e:
    print("[FAIL] bitsandbytes:", e)

print("-" * 50)
print("Test complete!")

