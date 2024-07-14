import torch

memory_bank = torch.load('/scratch/proj/memory_bank_23000.pth')
print(type(memory_bank))

# print(memory_bank.keys())