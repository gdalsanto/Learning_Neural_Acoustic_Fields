import torch
import dataclasses
from numpy.typing import NDArray, ArrayLike
# Remove CUDA-specific benchmark setting
# torch.backends.cudnn.benchmark = True
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import numpy as np
from data_loading.sound_loader import soundsamples
import pickle
import os
from model.networks import kernel_residual_fc_embeds
from model.modules import embedding_module_log
import math
from options import Options
import librosa 
import h5py
import soundfile as sf
import time

def to_wave(input_spec, orig_phase=None):
    renorm_input = np.concatenate((input_spec, input_spec[-1:]*0.0), axis=0)
    if orig_phase is None:
        out_wave = librosa.griffinlim(renorm_input, win_length=512, hop_length=128, n_iter=100, momentum=0.5, random_state=64)
    else:
        orig_phase = np.concatenate((orig_phase, orig_phase[-1:]*0.0), axis=0)
        f = renorm_input * (np.cos(orig_phase) + (1.j * np.sin(orig_phase)))
        out_wave = librosa.istft(f)
    resampled_wave = librosa.resample(out_wave, orig_sr=22050, target_sr=32000)
    return np.clip(resampled_wave, -1, 1)

@dataclasses.dataclass
class NAFDataset:
    orientation: int
    num_infer_receivers: int
    infer_receiver_pos: NDArray
    gt_brirs: NDArray
    infer_brirs: NDArray

def to_torch(input_arr):
    return input_arr[None]

def test_net(rank, other_args):
    pi = math.pi
    # Set device based on availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_device = device
    
    print("creating dataset")
    dataset = soundsamples(other_args)
    
    # Load original ground truth BRIR data from pickle file
    print("Loading original ground truth BRIR data from pickle file")
    original_pkl_path = os.path.join('data-local', 'naf_dataset_grid_spacing=06m.pkl')
    with open(original_pkl_path, 'rb') as f:
        original_data = pickle.load(f)
    
    # Load train/test split to map indices
    train_test_split_path = os.path.join(other_args.split_loc, other_args.apt + "_complete.pkl")
    with open(train_test_split_path, "rb") as train_test_file_obj:
        train_test_split = pickle.load(train_test_file_obj)
    
    test_split = train_test_split[1]
    
    xyz_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=other_args.num_freqs, ch_dim=2).to(output_device)
    auditory_net = kernel_residual_fc_embeds(input_ch=126, intermediate_ch=other_args.features, grid_ch=other_args.grid_features, num_block=other_args.layers, grid_gap=other_args.grid_gap, grid_bandwidth=other_args.bandwith_init, bandwidth_min=other_args.min_bandwidth, bandwidth_max=other_args.max_bandwidth, float_amt=other_args.position_float, min_xy=dataset.min_pos, max_xy=dataset.max_pos).to(output_device)

    loaded_weights = False
    current_files = sorted(os.listdir(other_args.exp_dir))
    if len(current_files)>0:
        latest = current_files[-1]
        print("Identified checkpoint {}".format(latest))
        # Use device-agnostic map_location
        map_location = {'cuda:%d' % rank: device} if device.type == 'cuda' else device
        weight_loc = os.path.join(other_args.exp_dir, latest)
        weights = torch.load(weight_loc, map_location=map_location)
        print("Checkpoint loaded {}".format(weight_loc))
        auditory_net.load_state_dict(weights["network"])
        loaded_weights = True
    if loaded_weights is False:
        print("Weights not found")
        
    auditory_net.eval()
    container = dict()
    
    first_waveform = True
    num_orientations = 4
    
    split_size = 50
    inference_time = 0.0
    with torch.no_grad():
        for ori in [0, 1, 2, 3]:
            num_sample_test = len(dataset.sound_files_test[["0", "90", "180", "270"][ori]])
            ori_offset = 0
            print("Total {} for orientation {}".format(num_sample_test, str(ori)))
            for split_id in range(0, num_sample_test, split_size):
                save_name = os.path.join(other_args.result_output_dir, other_args.apt+f"_NAF_{ori}_{split_id}.pkl")
                for test_id in range(split_id, min(split_id + split_size, num_sample_test)):
                    ori_offset += 1
                    if ori_offset%100 == 0:
                        print("Currently on {}".format(ori_offset))
                    data_stuff = dataset.get_item_test(ori, test_id)
                    gt = to_torch(data_stuff[0])
                    degree = torch.Tensor([data_stuff[1]]).to(output_device, non_blocking=True).long()
                    position = to_torch(data_stuff[2]).to(output_device, non_blocking=True)
                    non_norm_position = to_torch(data_stuff[3]).to(output_device, non_blocking=True)
                    freqs = to_torch(data_stuff[4]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
                    times = to_torch(data_stuff[5]).to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
                    PIXEL_COUNT = gt.shape[-1]
                    position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
                    freq_embed = freq_embedder(freqs)
                    time_embed = time_embedder(times)
                    total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
                    start_time = time.time()
                    output = auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
                    inference_time += time.time() - start_time
                    myout = output.cpu().numpy()
                    myout = myout.reshape(1, 2, dataset.sound_size[1], dataset.sound_size[2])
                    mygt = gt.numpy()
                    mygt = mygt.reshape(1, 2, dataset.sound_size[1], dataset.sound_size[2])
                    mygt = np.exp(mygt * dataset.std.numpy() + dataset.mean.numpy()) - 1e-3
                    myout = np.exp(myout * dataset.std.numpy() + dataset.mean.numpy()) - 1e-3
                    container["{}_{}".format(ori, dataset.sound_name)] = [myout, mygt, dataset.sound_size]
                    # convert to wav file 
                    myout_wav = np.concatenate((np.expand_dims(to_wave(myout[0, 0, ...]), axis=-1), np.expand_dims(to_wave(myout[0, 1, ...]), axis= -1)), -1)
                    
                    # Resize arrays if this is the first waveform
                    if first_waveform:
                        waveform_length = original_data.infer_brirs[0, 0].shape[0]
                        myout_wavs = np.zeros((split_size, 1, waveform_length, 2))
                        gt_wavs = np.zeros((split_size, 1, waveform_length, 2))
                        infer_positions = np.zeros((split_size, 3))
                        first_waveform = False

                    myout_wavs[test_id%split_size, 0, :, :] = myout_wav[:waveform_length, :]

                    test_key = test_split[["0", "90", "180", "270"][ori]][test_id]
                    # Parse the key to get receiver and emitter indices
                    receiver_idx, _ = test_key.split("_")
                    brir_idx = int(receiver_idx) - original_data.num_train_receivers
                    
                    # Get the original BRIR waveform (shape: [n_samples, channels])
                    gt_wav = original_data.infer_brirs[brir_idx, ori, :waveform_length, :].astype(np.float32)   

                    gt_wavs[test_id%split_size, 0, :, :] = gt_wav
                    infer_positions[test_id%split_size, :] = np.concatenate((non_norm_position.squeeze()[:2].cpu().numpy(), np.array([1.5])), axis=0)

                # save the file in the dataclass
                naf_dataset = NAFDataset(
                    orientation=ori,
                    num_infer_receivers=split_size,
                    infer_receiver_pos=np.round(infer_positions, decimals = 2),
                    gt_brirs=gt_wavs,
                    infer_brirs=myout_wavs
                )
                
                # Save the container with pickle
                with open(save_name, "wb") as saver_file_obj:
                    pickle.dump(dataclasses.asdict(naf_dataset), saver_file_obj)
                    print("Results saved to {}".format(save_name))
                first_waveform = True
    # save a text file with the inference time
    with open(os.path.join(other_args.result_output_dir, "inference_time.txt"), "a") as f:
        f.write("Inference time: {:.3f} s\n".format(inference_time))

if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled

    exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir

    result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    cur_args.result_output_dir = result_output_dir
    if not os.path.isdir(result_output_dir):
        os.mkdir(result_output_dir)

    if not os.path.isdir(cur_args.save_loc):
        print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
        exit()
    if not os.path.isdir(cur_args.exp_dir):
        print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
        exit()
    print("Experiment directory is {}".format(exp_dir))
    world_size = cur_args.gpus
    test_ = test_net(0, cur_args)
    ## Uncomment to run all rooms
    # for apt in ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']:
    #     cur_args.apt = apt
    #     exp_name = cur_args.exp_name
    #     exp_name_filled = exp_name.format(cur_args.apt)
    #     cur_args.exp_name = exp_name_filled
    #
    #     exp_dir = os.path.join(cur_args.save_loc, exp_name_filled)
    #     cur_args.exp_dir = exp_dir
    #
    #     result_output_dir = os.path.join(cur_args.save_loc, cur_args.inference_loc)
    #     cur_args.result_output_dir = result_output_dir
    #     if not os.path.isdir(result_output_dir):
    #         os.mkdir(result_output_dir)
    #
    #     if not os.path.isdir(cur_args.save_loc):
    #         print("Save directory {} does not exist, need checkpoint folder...".format(cur_args.save_loc))
    #         exit()
    #     if not os.path.isdir(cur_args.exp_dir):
    #         print("Experiment {} does not exist, need experiment folder...".format(cur_args.exp_name))
    #         exit()
    #     print("Experiment directory is {}".format(exp_dir))
    #     world_size = cur_args.gpus
    #     test_ = test_net(0, cur_args)
