import torch 
from torch_sim.models.mace import MaceModel

device = "cuda"
dtype = torch.float32

mace = torch.load("./mace_6.model", map_location=device)
batched_energy = MaceModel(model=mace, device=device, dtype=dtype, compute_stress=False, compute_forces=False)
batched_forces = MaceModel(model=mace, device=device, dtype=dtype, compute_stress=False, compute_forces=True)

calc = [batched_energy, batched_forces]
