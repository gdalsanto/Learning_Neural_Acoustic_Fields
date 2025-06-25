import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Any
import soundfile as sf
@dataclass
class NAFDataset:
    orientation: Any  
    num_infer_receivers: int
    infer_receiver_pos: np.ndarray
    gt_brirs: np.ndarray
    infer_brirs: np.ndarray

result_output_dir = "results/inference_out"
apt = "georg-binaural-0.6" 

# Collect all relevant .pkl files
all_files = [f for f in os.listdir(result_output_dir) if f.startswith(f"{apt}_NAF_") and f.endswith(".pkl")]
# If a file ends with 50.pkl rename it so that it ends with 050.pkl
all_files = [f if not f.endswith("_50.pkl") else f[:-7] + "_050.pkl" for f in all_files]
# Sort files for consistency
all_files.sort()
# restore original name 
all_files = [f if not f.endswith("_050.pkl") else f[:-8] + "_50.pkl" for f in all_files]

# Prepare lists to collect data
orientation = []
all_infer_positions =  np.empty((650, 3, 4), dtype=np.float32)  # Assuming 650 receivers, 4 orientations, and 2D position
all_gt_brirs = np.empty((650, 4, 131072, 2), dtype=np.float32)  # Assuming 650 receivers, 4 orientations, and 131072 samples
all_infer_brirs = np.empty((650, 4, 131072, 2), dtype=np.float32) 
total_receivers = 0
all_orientations = ['0', '90', '180', '270']  

for fname in all_files:
    with open(os.path.join(result_output_dir, fname), "rb") as f:
        data = pickle.load(f)
    # Remove orientation field if present
    ori = data["orientation"]
    indx_start = total_receivers % 650
    indx_stop = ( (total_receivers+data["num_infer_receivers"]) - 1) % 650
    orientation.append(all_orientations[ori])
    all_infer_positions[indx_start:indx_stop+1, :, ori] = np.array(data["infer_receiver_pos"])
    all_gt_brirs[indx_start:indx_stop+1, ori, :, :] = np.squeeze(data["gt_brirs"], 1)
    all_infer_brirs[indx_start:indx_stop+1, ori] = np.squeeze(data["infer_brirs"], 1)
    total_receivers += data["num_infer_receivers"]

# Save a random set of 10 files to wav 
for rndm_idx in np.random.choice(606, 10, replace=False):
    ori = np.random.choice([0, 1, 2, 3])
    wav_file_name = os.path.join(result_output_dir, f"{apt}_infer_{all_orientations[ori]}_{rndm_idx:04d}.wav")
    # librosa.output.write_wav(wav_file_name, myout_wavs[idx, ori, :, :], sr=32000)
    sf.write(wav_file_name, all_infer_brirs[rndm_idx, ori, :, :], 32000)
    gt_wav_file_name = os.path.join(result_output_dir, f"{apt}_gt_{all_orientations[ori]}_{rndm_idx:04d}.wav")
    # librosa.output.write_wav(gt_wav_file_name, gt_wavs[idx, ori, :, :], sr=32000)
    sf.write(gt_wav_file_name, all_gt_brirs[rndm_idx, ori, :, :], 32000)

# Create unified NAFDataset
unified_dataset = NAFDataset(
    orientation=None,
    num_infer_receivers=607,
    infer_receiver_pos=np.round(all_infer_positions[:607, ...], 2),
    gt_brirs=all_gt_brirs[:607, ...],
    infer_brirs=all_infer_brirs[:607, ...],
)

print("Unified NAFDataset created!")
print("Total receivers:", unified_dataset.num_infer_receivers)
print("infer_receiver_pos shape:", unified_dataset.infer_receiver_pos.shape)
print("gt_brirs shape:", unified_dataset.gt_brirs.shape)
print("infer_brirs shape:", unified_dataset.infer_brirs.shape)