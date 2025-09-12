from mace.calculators import MACECalculator
import torch
import os

rank = int(os.environ["rank"])
device = torch.cuda.device_count()

calc = MACECalculator(model_paths='./mace_6.model', device=f'cuda:{rank%device}')
