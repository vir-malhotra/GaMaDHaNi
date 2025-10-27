from random import randint
from typing import Dict, Optional
from sklearn.preprocessing import QuantileTransformer
import gin
import numpy as np
import torch

TensorDict = Dict[str, torch.Tensor]

@gin.configurable
def pitch_read_downsample(inputs: TensorDict,
                          seq_len: int, 
                          decoder_key: str, 
                          min_norm_pitch: int,  
                          time_downsample: int=1, 
                          pitch_downsample: int=1,
                          base_tonic: float=440.,
                          transpose_pitch: Optional[int]=0,
                          start: Optional[int]=None,
                          **kwargs):
        data = inputs[decoder_key]["data"]
        if seq_len is not None:
            start = start if start is not None else randint(0, max(0, data.shape[0] - seq_len * time_downsample - 1))
            end = start + seq_len * time_downsample
            
            f0 = np.copy(data[start : end+1 : time_downsample])
        else:
            f0 = np.copy(data)

        # normalizing pitch contour from hertz to cents
        f0[f0 == 0] = np.nan
        norm_f0 = np.copy(f0)
        norm_f0[~np.isnan(norm_f0)] = (1200) * np.log2(norm_f0[~np.isnan(norm_f0)] / base_tonic)
        del f0

        #discretizing and making sure the values are positive
        norm_f0[~np.isnan(norm_f0)] = np.around(norm_f0[~np.isnan(norm_f0)])
        norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] - (min_norm_pitch)

        #pitch downsampling
        norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] // pitch_downsample + 1    # adding 1 to preserve 0 as a silence token 
        
        # data augmentation, if apply_transform is set to True in config
        if transpose_pitch:
            transposed_values = norm_f0[~np.isnan(norm_f0)] + (transpose_pitch//pitch_downsample)
            norm_f0[~np.isnan(norm_f0)] = transposed_values
        
        # add silence token of 0
        norm_f0[np.isnan(norm_f0)] = 0

        input_tokens = np.copy(norm_f0[:-1, None])
        target_tokens = np.copy(norm_f0[1:, None])

        # Include raga information if available
        result = {
            "decoder_inputs": input_tokens,
            "decoder_targets": target_tokens,
            "sampled_sequence": norm_f0
        }

        # Add global conditions (raga, tonic, singer) if present
        if 'global_conditions' in inputs:
            result['raga'] = inputs['global_conditions'].get('raga', -1)
            result['tonic'] = inputs['global_conditions'].get('tonic', base_tonic)
            result['singer'] = inputs['global_conditions'].get('singer', -1)

        return result
@gin.configurable
def invert_pitch_read_downsample(f0,
                          min_norm_pitch: int,  
                          time_downsample: int=1, 
                          pitch_downsample: int=1,
                          base_tonic: float=440.,
                          seq_len: int=None, 
                          decoder_key: str=None,
                          **kwargs):
    f0 = f0.reshape(-1, 1)
    f0[f0 == 0] = np.nan
    f0[~np.isnan(f0)] = ((f0[~np.isnan(f0)] - 1) * pitch_downsample)
    f0[~np.isnan(f0)] = f0[~np.isnan(f0)] + min_norm_pitch
    # Unnormalize the pitch contours
    f0[~np.isnan(f0)] = base_tonic * (2**(f0[~np.isnan(f0)] / 1200))
    
    return f0

@gin.configurable
def pitch_read_downsample_diff(
    inputs: TensorDict, 
    seq_len: int, 
    decoder_key: str, 
    min_norm_pitch: int, 
    transpose_pitch: Optional[int] = None, 
    time_downsample: int = 1, 
    pitch_downsample: int = 1, 
    qt_transform: Optional[QuantileTransformer] = None,
    min_clip: int = 200,
    max_clip: int = 600,
    add_noise_to_silence: bool = False,
     **kwargs
    ):
    
    data = inputs[decoder_key]["data"]
    if seq_len is not None:
        start = randint(0, max(0, data.shape[0] - seq_len*time_downsample - 1))
        end = start + seq_len*time_downsample
        f0 = np.copy(inputs[decoder_key]['data'][start:end:time_downsample])
    else:
        f0 = np.copy(data[::time_downsample])

    # normalize pitch
    f0[f0 == 0] = np.nan
    norm_f0 = f0
    norm_f0[~np.isnan(norm_f0)] = (1200) * np.log2(norm_f0[~np.isnan(norm_f0)] / 440)
    del f0

    # descretize pitch
    norm_f0[~np.isnan(norm_f0)] = np.around(norm_f0[~np.isnan(norm_f0)])
    norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] - (min_norm_pitch)

    norm_f0[~np.isnan(norm_f0)] = norm_f0[~np.isnan(norm_f0)] // pitch_downsample + 1 # reserve 0 for silence
    
    # data augmentation
    if transpose_pitch:
        transpose_amt = randint(-transpose_pitch, transpose_pitch)  # in cents
        transposed_values = norm_f0[~np.isnan(norm_f0)] + (transpose_amt//pitch_downsample)
        norm_f0[~np.isnan(norm_f0)] = transposed_values

    # clip values HACK to change
    norm_f0[~np.isnan(norm_f0)] = np.clip(norm_f0[~np.isnan(norm_f0)], min_clip, max_clip)

    # add silence token of min_clip - 4
    if add_noise_to_silence:
        norm_f0[np.isnan(norm_f0)] = min_clip - 4 + np.clip(np.random.normal(size=norm_f0[np.isnan(norm_f0)].shape), -3, 3) # making sure noise is between -3 and 3 and thus won't spill into pitched values
    else:
        norm_f0[np.isnan(norm_f0)] = min_clip - 4
    
    if qt_transform:
        qt_inp = norm_f0.reshape(-1, 1)
        norm_f0 = qt_transform.transform(qt_inp).reshape(-1)

    return {"sampled_sequence": norm_f0}

@gin.configurable
def invert_pitch_read_downsample_diff(f0,
                  seq_len: int,
                  min_norm_pitch: int,
                  time_downsample: int,
                  pitch_downsample: int,
                  qt_transform: Optional[QuantileTransformer],
                  min_clip: int,
                  max_clip: int,
                  base_tonic: float=440., 
                  decoder_key: str=None,
                   **kwargs):
    try:
        f0 = f0.detach().cpu().numpy()
    except:
        pass
    if qt_transform is not None:
        f0 = qt_transform.inverse_transform(f0.reshape(-1, 1))
        f0.reshape(1, -1)
    f0[f0 < min_clip] = np.nan
    f0[~np.isnan(f0)] = (f0[~np.isnan(f0)] - 1) * pitch_downsample
    f0[~np.isnan(f0)] = f0[~np.isnan(f0)] + min_norm_pitch
    f0[~np.isnan(f0)] = base_tonic * 2**(f0[~np.isnan(f0)] / 1200)
    f0[np.isnan(f0)] = 0

    return f0