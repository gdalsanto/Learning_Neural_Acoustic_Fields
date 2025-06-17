import torchaudio
import dataclasses
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import wavfile
from torchaudio.transforms import Spectrogram
import librosa
from skimage.transform import rescale, resize
from scipy.interpolate import interp1d
import os
from numpy.typing import NDArray, ArrayLike
import gc
import pickle


@dataclasses.dataclass
class NAFDataset:
    num_train_receivers: int 
    num_infer_receivers: int
    train_receiver_pos: NDArray
    infer_receiver_pos: NDArray
    train_brirs: NDArray
    infer_brirs: NDArray
    orientation: ArrayLike


def load_audio(wave_data_loaded, resample=True, sr_loaded=32000, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr

    if resample:
        if wave_data_loaded.shape[1]==0:
            print("len 0")
            assert False
        if wave_data_loaded.shape[1]<int(sr_loaded*0.1):
            padded_wav = librosa.util.fix_length(wave_data_loaded, int(sr_loaded*0.1))
            resampled_wave = librosa.resample(padded_wav, orig_sr=sr_loaded, target_sr=resample_rate)
        else:
            resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    else:
        resampled_wave = wave_data_loaded
    return np.clip(resampled_wave, -1.0, 1.0)

def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:,:,0:1], np.diff(unwrapped_angle, n=1)], axis=-1)

class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft=fft_size
        self.hop = self.n_fft//4
        if use_torch:
            assert False
            self.use_torch = True
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None
        
    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(wav_data_prepad, size = wav_data_prepad.shape[-1]+self.n_fft//2)
        if wav_data.shape[1]<4410:
            wav_data = librosa.util.fix_length(wav_data, 4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            
            transformed_data = np.array([librosa.stft(wav_data[0],n_fft=self.n_fft, hop_length=self.hop),
               librosa.stft(wav_data[1],n_fft=self.n_fft, hop_length=self.hop)])[:,:-1]
#         print(np.array([librosa.stft(wav_data[0],n_fft=self.n_fft, hop_length=self.hop),
#                librosa.stft(wav_data[1],n_fft=self.n_fft, hop_length=self.hop)]).shape, "OLD SHAPE")

        real_component = np.abs(transformed_data)
        img_component = np.angle(transformed_data)
        gen_if = if_compute(img_component)/np.pi
        
        # Add checks
        if np.any(np.isnan(real_component)):
            print("Warning: NaN in real_component")
            real_component = np.nan_to_num(real_component, nan=0.0)
        
        log_mag = np.log(real_component+1e-3)
        if np.any(np.isnan(log_mag)):
            print("Warning: NaN in log_mag")
            log_mag = np.nan_to_num(log_mag, nan=0.0)
        
        return log_mag, gen_if, img_component

def get_wave_if(input_stft, input_if):
    # 2 chanel input of shape [2,freq,time]
    # First input is logged mag
    # Second input is if divided by np.pi
    padded_input_stft = np.concatenate((input_stft, input_stft[:,-1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:,-1:]*0.0), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1)*np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    restored = (np.exp(padded_input_stft)-1e-3)*phase_val
    wave1 = librosa.istft(restored[0], hop_length=512//4)
    wave2 = librosa.istft(restored[1], hop_length=512//4)
    return wave1, wave2


def pad(input_arr, max_len_in, constant=np.log(1e-3)):
    return np.pad(input_arr, [[0,0],[0,0],[0,max_len_in-input_arr.shape[2]]], constant_values=constant)
    


if __name__ == "__main__":
    
    # Path to the .pkl file
    pkl_path = os.path.join('data-local', 'naf_dataset_grid_spacing=0.6m.pkl')

    # Load the .pkl file (this may take a while if the file is large)
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)

    mag_path = "data-local/georg-binaural/magnitudes"
    phase_path = "data-local/georg-binaural/phases"

    # create directories if they do not exist
    os.makedirs(mag_path, exist_ok=True)
    os.makedirs(phase_path, exist_ok=True)

    spec_getter = get_spec()
    room_name = "georg-binaural"
    length_tracker = []

    mag_object = os.path.join(mag_path, room_name)
    phase_object = os.path.join(phase_path, room_name)
    f_mag = h5py.File(mag_object+".h5", 'w')
    f_phase = h5py.File(phase_object+".h5", 'w')
    zz = 0
    orientations = ["0", "90", "180", "270"]

    # initialize a dictionary with the orientations as keys
    train_test_split = (
        {
            "0": [],
            "90": [],
            "180": [],
            "270": []
        },
        {
            "0": [],
            "90": [],
            "180": [],
            "270": []
        }
    )
    print("Found {} train files".format(str(raw_data.num_train_receivers)))
    print("Found {} infer files".format(str(raw_data.num_infer_receivers)))

    # concatenate train and infer brir data
    raw_data.train_brirs = np.concatenate((raw_data.train_brirs, raw_data.infer_brirs), axis=0)

    for orientation in range(4):
        print("Processing orientation {}".format(orientations[orientation]))
        for ff in range(raw_data.num_train_receivers + raw_data.num_infer_receivers):
            zz+= 1 
            if zz % 500==0:
                print(zz)
            try:
                cur_file = raw_data.train_brirs[ff, orientation, ...].astype(np.float32)
                # change the shape from [n_samples, channels] to [channels, n_samples]
                cur_file = cur_file.T
                loaded_wav = load_audio(cur_file)
            except Exception as e:
                print("0 length wav", cur_file, e)
                continue
            real_spec, img_spec, raw_phase = spec_getter.transform(loaded_wav)
            length_tracker.append(real_spec.shape[2])
            f_mag.create_dataset('{}_{:04d}'.format(orientations[orientation], ff), data=real_spec.astype(np.half))
            f_phase.create_dataset('{}_{:04d}'.format(orientations[orientation], ff), data=img_spec.astype(np.half))
            if ff < raw_data.num_train_receivers:
                train_test_split[0][orientations[orientation]].append('{:04d}_{:04d}'.format(ff, raw_data.num_train_receivers + raw_data.num_infer_receivers))
            else:
                train_test_split[1][orientations[orientation]].append('{:04d}_{:04d}'.format(ff, raw_data.num_train_receivers + raw_data.num_infer_receivers))
    print("Max length {}".format(room_name), np.max(length_tracker))
    f_mag.close()
    f_phase.close()

    with open(f'./metadata/train_test_split/{room_name}_complete.pkl', 'wb') as f:
        pickle.dump(train_test_split, f)


 
    raw_path = mag_path
    mean_std = "data-local/georg-binaural/magnitude_mean_std"
    os.makedirs(mean_std, exist_ok=True)
    max_len_dict = {room_name: np.max(length_tracker)}

    files = os.listdir(raw_path)
    for f_name_old in sorted(list(max_len_dict.keys())):
        f_name = f_name_old+".h5"
        print("Processing ", f_name)
        f = h5py.File(os.path.join(raw_path, f_name), 'r')
        keys = list(f.keys())
        max_len = max_len_dict[f_name.split(".")[0]]
        all_arrs = []
        for idx in range(len(keys)):  
            all_arrs.append(pad(f[keys[idx]], max_len).astype(np.single))
        print("Computing mean")
        mean_val = np.mean(all_arrs, axis=(0,1))
        print("Computing std")
        std_val = np.std(all_arrs, axis=(0,1))
        std_val = np.clip(std_val, a_min=1e-6, a_max=None)  # Ensure no zeros
        print("Std min/max after clip:", np.min(std_val), np.max(std_val))
        
        plt.imshow(all_arrs[0][0])
        plt.show()
        plt.imshow(mean_val)
        plt.show()
        plt.imshow(std_val)
        plt.show()
        print(mean_val.shape)
        del all_arrs
        f.close()
        gc.collect()
        with open(os.path.join(mean_std, f_name.replace("h5", "pkl")), "wb") as mean_std_file:
            pickle.dump([mean_val, std_val], mean_std_file)
    
    # Create points.txt
    points_file_path = os.path.join('data-local', room_name, 'points.txt')

    with open(points_file_path, 'w') as f:
        for i, pos in enumerate(np.concatenate((raw_data.train_receiver_pos, raw_data.infer_receiver_pos), axis=0)):
            f.write("{:04d}\t{}\t{}\t{}\n".format(i, pos[0], pos[1], pos[2]))
        # write the sound source position
        f.write("{:04d}\t2.0\t2.0\t1.5\n".format(i+1))
    # Create _minmax.pkl
    minmax_file_path = os.path.join('data-local', f'{room_name}_minmax.pkl')
    min_pos = np.min(np.concatenate((raw_data.train_receiver_pos, raw_data.infer_receiver_pos), axis=0), axis=0)
    max_pos = np.max(np.concatenate((raw_data.train_receiver_pos, raw_data.infer_receiver_pos), axis=0), axis=0)

    with open(minmax_file_path, 'wb') as f:
        pickle.dump((min_pos, max_pos), f)
