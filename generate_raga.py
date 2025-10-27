"""
Raga-conditioned generation script for GaMaDHaNi.
Extends the original generation script with raga conditioning support.
"""
import numpy as np
import pandas as pd
import torch
import os
from functools import partial
import gin
from absl import flags, app

from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
import gamadhani.utils.pitch_to_audio_utils as p2a
from gamadhani.utils.utils import get_device, plot, save_figure, save_csv, save_audio, download_models, download_data
from absl import app
import logging
from typing import Optional, Callable, Tuple, Any
import matplotlib.pyplot as plt
import pdb

FLAGS = flags.FLAGS

# Original flags
flags.DEFINE_string('pitch_model_type', default='transformer', required=True, help='Whether to use diffusion or transformer model for pitch generation (only transformer supports raga conditioning)')
flags.DEFINE_bool('prime', default=False, help='Boolean value indicating whether to perform Primed pitch generation')
flags.DEFINE_string('pitch_run', default=None, help='Path to the pitch generation model folder')
flags.DEFINE_string('audio_run', default=None, help='Path to the pitch to audio generation model folder')
flags.DEFINE_bool('download_model_from_hf', default=False, help='Boolean value indicating whether to download model files from Huggingface')
flags.DEFINE_string('hf_model_repo_id', default="kmaneeshad/GaMaDHaNi", help='model repository on huggingface, to download model files from')
flags.DEFINE_string('hf_data_repo_id', default="kmaneeshad/GaMaDHaNi-db", help='data repository on huggingface, to download data files from')
flags.DEFINE_integer('number_of_samples', default=1, help='number of samples to generate')
flags.DEFINE_string('outfolder', default=os.getcwd(),help='path where the generated pitch contour plots, csv files and audio files are to be saved' )
flags.DEFINE_integer('seq_len', default=1200, help='Used when pitch_model_type==transformer, total length of the sequence to be generated, when running primed generation, seq_len includes the prime too')
flags.DEFINE_float('temperature', default=1.0, help='Used when pitch_model_type==transformer, controls randomness in sampling; lower values (< 1.0) produce more deterministic results, while higher values (> 1.0) increase diversity. ')
flags.DEFINE_integer('num_steps', default=100, help='Used when pitch_model_type==diffusion, the number of diffusion steps for the model; more steps can improve quality but increase computation time.')
flags.DEFINE_list('singers', default=[3], help='Used by the pitch to audio model, singer IDs for the singer conditioning')
flags.DEFINE_string('pitch_config_path', default=None, help='config file path for the pitch generation model')
flags.DEFINE_string('audio_config_path', default=None, help='config file path for the pitch to audio generation model')
flags.DEFINE_string('qt_pitch_path', default=None, help='Path to the QuantileTransform file for pitch generation diffusion model')
flags.DEFINE_string('qt_audio_path', default=None, help='Path to the QuantileTransform file for pitch to audio diffusion model')

# Raga conditioning flags
flags.DEFINE_integer('raga_id', default=None, help='Raga ID for conditioning (integer). If None, generates without raga conditioning')
flags.DEFINE_string('raga_name', default=None, help='Raga name for reference (not used for conditioning, only for labeling output)')
flags.DEFINE_list('raga_ids', default=None, help='List of raga IDs to generate multiple samples with different ragas')


def generate_pitch_raga(pitch_model,
                        pitch_model_type,
                        invert_pitch_fn,
                        num_samples,
                        seq_len=1200,
                        temperature=1.0,
                        num_steps=None,
                        outfolder=None,
                        processed_primes=None,
                        pitch_sample_rate=200,
                        prime=False,
                        raga_ids=None):
    """
    Generate pitch contours with optional raga conditioning.

    Args:
        raga_ids: Tensor of raga IDs, shape (num_samples,). If None, no conditioning.
    """
    logging.log(logging.INFO, 'Generate pitch function with raga conditioning')

    if raga_ids is not None:
        logging.log(logging.INFO, f'Raga conditioning: {raga_ids.cpu().tolist()}')

    if processed_primes is not None:
        processed_primes = torch.tensor(processed_primes).to(pitch_model.device)

    if pitch_model_type == "diffusion":
        if raga_ids is not None:
            logging.warning("Diffusion model does not support raga conditioning, ignoring raga_ids")
        samples = pitch_model.sample_fn(batch_size=num_samples, num_steps=num_steps, prime=processed_primes)
    else:
        # Transformer with raga conditioning
        samples = pitch_model.sample_fn(
            batch_size=num_samples,
            seq_len=seq_len,
            prime=processed_primes,
            temperature=temperature,
            raga_ids=raga_ids  # Pass raga IDs
        )

    inverted_pitches = [invert_pitch_fn(**{"f0": sample}) for sample in samples.detach().cpu().numpy()]

    if outfolder is not None:
        for i, pitch in enumerate(inverted_pitches):
            # Add raga info to filename if available
            raga_suffix = ""
            if raga_ids is not None:
                raga_suffix = f"_raga{raga_ids[i].item()}"

            outfile = f"{outfolder}/output/"
            time_array = np.arange(0, len(pitch) / pitch_sample_rate, 1/pitch_sample_rate)

            save_csv(
                df=pd.DataFrame({"f0": pitch.reshape(-1), "time": time_array}),
                dest_path=os.path.join(outfile, f"pitch/{i}{raga_suffix}.csv")
            )

            fig = plot(f0_array=pitch, time_array=time_array, prime=prime)
            if raga_ids is not None:
                # Add raga info to plot title
                fig.suptitle(f'Sample {i} - Raga ID: {raga_ids[i].item()}', fontsize=12)
            save_figure(fig, dest_path=os.path.join(outfile, f"plots/{i}{raga_suffix}.png"))
            plt.close(fig)

    return samples, torch.Tensor(np.array(inverted_pitches)).to(pitch_model.device)


def generate_audio(audio_model, f0s, invert_audio_fn, outfolder, singers=[3], num_steps=100, raga_ids=None):
    """Generate audio with optional raga labeling in filenames."""
    singer_tensor = torch.tensor(np.repeat(singers, repeats=f0s.shape[0])).to(audio_model.device)
    samples, _, singers = audio_model.sample_cfg(
        f0s.shape[0], f0=f0s, num_steps=num_steps, singer=singer_tensor,
        strength=3, invert_audio_fn=invert_audio_fn
    )
    audio = invert_audio_fn(samples)

    if outfolder is not None:
        os.makedirs(outfolder, exist_ok=True)
        for i, a in enumerate(audio):
            raga_suffix = ""
            if raga_ids is not None:
                raga_suffix = f"_raga{raga_ids[i].item()}"

            outfile = f"{outfolder}/output/audio/{i}{raga_suffix}.wav"
            logging.log(logging.INFO, f"Saving audio {i} to {outfile}")
            save_audio(audio_array=a.clone().detach().unsqueeze(0).cpu(), dest_path=outfile, sample_rate=16000)

    return audio


def generate(audio_model=None,
            pitch_model=None,
            pitch_model_type=None,
            invert_pitch_fn=None,
            invert_audio_fn=None,
            num_samples=2,
            singers=[3],
            num_steps=100,
            seq_len=1200,
            temperature=1.,
            outfolder='temp',
            audio_seq_len=750,
            pitch_qt=None,
            prime=False,
            processed_primes=None,
            device=None,
            raga_ids=None):
    """
    Generate pitch and audio with optional raga conditioning.

    Args:
        raga_ids: Tensor of raga IDs for conditioning, shape (num_samples,)
    """

    logging.log(logging.INFO, 'Generate function with raga conditioning')

    pitch_model = pitch_model.to(device)

    pitch, inverted_pitch = generate_pitch_raga(
        pitch_model=pitch_model,
        pitch_model_type=pitch_model_type,
        invert_pitch_fn=invert_pitch_fn,
        num_samples=num_samples,
        num_steps=num_steps,
        outfolder=outfolder,
        seq_len=seq_len,
        temperature=temperature,
        processed_primes=processed_primes,
        pitch_sample_rate=100,
        prime=prime,
        raga_ids=raga_ids
    )

    if pitch_qt is not None:
        pitch_qt = p2a.GPUQuantileTransformer(pitch_qt, device)
        def undo_qt(x, min_clip=200):
            pitch = pitch_qt.inverse_transform(x)
            pitch = torch.round(pitch)
            pitch[pitch < 200] = np.nan
            return pitch
        pitch = undo_qt(pitch)
    elif pitch_model_type == "transformer":
        pitch = torch.where(pitch < 200, torch.tensor(float('nan')), pitch)
        pitch = torch.clip(pitch, min=200, max=600)

    interpolated_pitch = p2a.interpolate_pitch(pitch=pitch, audio_seq_len=audio_seq_len)
    interpolated_pitch = torch.nan_to_num(interpolated_pitch, nan=196)
    interpolated_pitch = interpolated_pitch.squeeze(1)
    interpolated_pitch = interpolated_pitch.float()

    audio_model = audio_model.to(device)
    audio = generate_audio(
        audio_model, interpolated_pitch, invert_audio_fn,
        singers=singers, num_steps=100, outfolder=outfolder, raga_ids=raga_ids
    )

    return pitch, audio


def main(argv):
    pitch_path = FLAGS.pitch_run
    audio_path = FLAGS.audio_run
    qt_pitch_path = FLAGS.qt_pitch_path
    qt_audio_path = FLAGS.qt_audio_path
    download_model_from_hf = FLAGS.download_model_from_hf
    hf_model_repo_id = FLAGS.hf_model_repo_id
    hf_data_repo_id = FLAGS.hf_data_repo_id
    prime = FLAGS.prime
    prime_file = None
    pitch_model_type = FLAGS.pitch_model_type
    number_of_samples = FLAGS.number_of_samples
    outfolder = FLAGS.outfolder
    seq_len = FLAGS.seq_len
    temperature = FLAGS.temperature
    num_steps = FLAGS.num_steps
    singers = FLAGS.singers
    device = get_device()
    pitch_config_path = FLAGS.pitch_config_path if FLAGS.pitch_config_path else os.path.join(os.getcwd(), f"configs/{pitch_model_type}_pitch_config.gin")
    audio_config_path = FLAGS.audio_config_path if FLAGS.audio_config_path else os.path.join(os.getcwd(), f"configs/pitch_to_audio_config.gin")

    # Process raga conditioning
    raga_ids_tensor = None
    if FLAGS.raga_ids is not None:
        # Multiple ragas specified
        raga_ids_list = [int(r) for r in FLAGS.raga_ids]
        if len(raga_ids_list) != number_of_samples:
            raise ValueError(f"Number of raga_ids ({len(raga_ids_list)}) must match number_of_samples ({number_of_samples})")
        raga_ids_tensor = torch.tensor(raga_ids_list)
        logging.info(f"Generating with raga IDs: {raga_ids_list}")
    elif FLAGS.raga_id is not None:
        # Single raga for all samples
        raga_ids_tensor = torch.tensor([FLAGS.raga_id] * number_of_samples)
        logging.info(f"Generating all samples with raga ID: {FLAGS.raga_id}")
        if FLAGS.raga_name:
            logging.info(f"  (Raga name: {FLAGS.raga_name})")
    else:
        logging.info("Generating without raga conditioning (unconditioned generation)")

    if pitch_model_type != "transformer" and raga_ids_tensor is not None:
        logging.warning("Raga conditioning only supported for transformer model type")

    # Validation
    if not isinstance(prime, bool):
        raise ValueError("Invalid prime flag")

    if not pitch_model_type:
        raise ValueError("Missing pitch_model_type flag")

    if prime:
        if hf_data_repo_id:
            prime_file = download_data(hf_data_repo_id)
        else:
            raise ValueError("Need to provide hf_data_repo_id when prime is True")
        assert number_of_samples <= 16, "Number of samples should be less than or equal to 16 when prime is True"

    if not isinstance(number_of_samples, int) or number_of_samples <= 0:
        raise ValueError("Invalid number_of_samples flag")

    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("Invalid seq_len flag")

    if not isinstance(temperature, (int, float)):
        raise ValueError("Invalid temperature flag")

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("Invalid num_steps flag")

    if not isinstance(singers, list) or not singers:
        raise ValueError("Invalid singers flag")

    if download_model_from_hf:
        pitch_path, qt_pitch_path, audio_path, qt_audio_path = download_models(hf_model_repo_id, pitch_model_type)

    if seq_len:
        if prime:
            seq_len_cache = int((1/3) * seq_len)
            seq_len_gen = int((2/3) * seq_len)
        else:
            seq_len_gen = seq_len

    # Load models
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, primes = load_pitch_fns(
        pitch_path=pitch_path,
        model_type=pitch_model_type,
        prime=prime,
        prime_file=prime_file,
        qt_path=qt_pitch_path,
        number_of_samples=number_of_samples,
        config_path=pitch_config_path,
        device=device
    )

    audio_model, audio_qt, audio_seq_len, invert_audio_fn = load_audio_fns(
        audio_path=audio_path,
        qt_path=qt_audio_path,
        config_path=audio_config_path,
        device=device
    )

    # Move raga_ids to device if specified
    if raga_ids_tensor is not None:
        raga_ids_tensor = raga_ids_tensor.to(device)

    # Generate
    pitch, audio = generate(
        audio_model=audio_model,
        pitch_model=pitch_model,
        pitch_model_type=pitch_model_type,
        invert_pitch_fn=invert_pitch_fn,
        invert_audio_fn=invert_audio_fn,
        num_samples=number_of_samples,
        singers=singers,
        num_steps=num_steps,
        outfolder=outfolder,
        seq_len=seq_len_gen,
        temperature=temperature,
        prime=prime,
        processed_primes=primes,
        device=device,
        pitch_qt=pitch_qt,
        raga_ids=raga_ids_tensor
    )

    logging.info("Generation complete!")
    if FLAGS.raga_name:
        logging.info(f"Generated samples for raga: {FLAGS.raga_name}")


if __name__ == '__main__':
    app.run(main)
