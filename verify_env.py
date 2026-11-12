import torch
import transformers
import peft
import accelerate
import datasets

print('[OK] PyTorch:', torch.__version__)
print('[OK] CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[OK] GPU:', torch.cuda.get_device_name(0))
    print('[OK] VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
print('[OK] transformers:', transformers.__version__)
print('[OK] peft:', peft.__version__)
print('[OK] accelerate:', accelerate.__version__)
print('[OK] datasets:', datasets.__version__)
print('\n[SUCCESS] All packages installed correctly!')
