import numpy as np
import os

proc = os.listdir("data/processed")

print(f"Total processed : {len(proc)}")
print(f"Planet (label1) : {sum(1 for f in proc if 'label1' in f)}")
print(f"No-planet(label0): {sum(1 for f in proc if 'label0' in f)}")

s = np.load(f"data/processed/{proc[0]}")

print(f"Shape  : {s.shape}")
print(f"Min    : {s.min():.4f}  (should be 0.0)")
print(f"Max    : {s.max():.4f}  (should be 1.0)")
print(f"NaNs   : {np.isnan(s).sum()}  (should be 0)")