import sys
sys.path.append(r'D:\python_code\projects\thesis\src\synthesize\lesion_ldm')
from preprocess.create_monai_dataset import create_monai_dataset
from monai.networks.nets import autoencoderkl

vae=autoencoderkl()