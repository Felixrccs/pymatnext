from mace.calculators import MACECalculator
import torch


calc = []


for i in range(torch.cuda.device_count()):
    calc.append(MACECalculator(model_paths='./mace_6.model', device=f'cuda:{i}'))
