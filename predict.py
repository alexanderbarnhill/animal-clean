import sys

import argparse
import os
from datetime import datetime
import logging as log
import scipy
import json
from glob import glob
from math import ceil
import data.transforms as T

import torch.cuda
from torch.utils.data import DataLoader

from data import signal
from data.audiodataset import SingleAudioFolder, StridedAudioDataset
from models.model import AnimalClean
from utilities.configuration import get_configuration
import shutil

from utilities.viewing import convert_tensor_to_PIL

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    help="Configuration file for predictions.",
                    default=os.path.join(os.getcwd(), "predict.config.yaml"))
parser.add_argument("--log_output", default=os.path.join(os.getcwd(), "logs"))

args = parser.parse_args()


def get_data_loader(conf, **kwargs):
    if os.path.isdir(conf.input):
        log.info(f"Looking for audio files in {conf.input}")
        if "file_ext" in conf and conf.file_ext is not None:
            log.info(f"Looking for audio files with extension {conf.file_ext}")
            audio_files = glob(conf.input + f"**{os.sep}**{os.sep}*{conf.file_ext}", recursive=True)
        else:
            log.info(f"No extension specified. Looking for WAV files")
            audio_files = glob(conf.input + f"**{os.sep}**{os.sep}*.wav", recursive=True)
        dataset = SingleAudioFolder(
            file_names=audio_files,
            working_dir=conf.input,
            sr=kwargs["sr"],
            n_fft=kwargs["n_fft"],
            hop_length=kwargs["hop_length"],
            n_freq_bins=kwargs["n_freq_bins"],
            freq_compression=kwargs["freq_compression"],
            f_min=kwargs["fmin"],
            f_max=kwargs["fmax"],
            min_max_normalize=False
        )
        concat = False
    elif os.path.isfile(conf.input):
        dataset = StridedAudioDataset(
            conf.input.strip(),
            sequence_len=kwargs["sequence_len"],
            hop=kwargs["hop"],
            sr=kwargs["sr"],
            fft_size=kwargs["n_fft"],
            fft_hop=kwargs["hop_length"],
            n_freq_bins=kwargs["n_freq_bins"],
            f_min=kwargs["fmin"],
            f_max=kwargs["fmax"],
        )
        concat = True
    else:
        raise ValueError(f"Input Path not a valid directory or file: {conf.input}")

    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        shuffle=False
    )

    return loader, concat


def plot_spectrograms():
    pass


def write_audio():
    pass

def clean_signal():
    pass

def get_file_out_name(f:str, conf) -> str:
    maint_struct = False
    if conf.maintain_directory_structure:
        maint_struct = True
    output_dir = conf.output
    if isinstance(f, tuple):
        f = f[0]

    if maint_struct:
        source_dir = os.path.dirname(f)
        dir_diff = source_dir.replace(conf.input, "")
        if dir_diff[0] == os.sep:
            dir_diff = dir_diff[1:]

        output_dir = os.path.join(output_dir, dir_diff)

    audio_out_name = os.path.join(output_dir, os.path.basename(f))
    os.makedirs(os.path.dirname(audio_out_name), exist_ok=True)
    return audio_out_name

def get_img_outputs(conf):
    vis_opts = conf.visualization
    make_new_img_folder = False
    if vis_opts.separate_img:
        make_new_img_folder = True

    if vis_opts.img_out is None and make_new_img_folder:
        base_out = os.path.join(conf.output, "img")
    elif vis_opts.img_out is not None:
        base_out = vis_opts.img_out
    else:
        base_out = conf.output

    input_out, output_out = base_out, base_out
    if vis_opts.split_in_out:
        input_out = os.path.join(base_out, "in")
        output_out = os.path.join(base_out, "out")

    os.makedirs(input_out, exist_ok=True)
    os.makedirs(output_out, exist_ok=True)

    if input_out == output_out:
        log.info(f"Writing image output to {input_out}")
        return input_out, None
    log.info(f"Writing spectral input to {input_out}")
    log.info(f"Writing denoised output image to {output_out}")
    return input_out, output_out


if __name__ == '__main__':
    log_format = log.Formatter(fmt="%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = log.getLogger()
    rootLogger.setLevel(log.INFO)

    consoleHandler = log.StreamHandler()
    consoleHandler.setFormatter(log_format)
    consoleHandler.setLevel(log.INFO)
    consoleHandler.setStream(sys.stdout)

    os.makedirs(args.log_output, exist_ok=True)
    fileHandler = log.FileHandler(
        filename=os.path.join(args.log_output, f"predict_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}.log"),
        mode="w")
    fileHandler.setFormatter(log_format)
    fileHandler.setLevel(log.INFO)

    rootLogger.addHandler(consoleHandler)
    rootLogger.addHandler(fileHandler)

    log.info(f"Set File and Console Logging Handlers")

    configuration = get_configuration(args.config)

    output = configuration.output
    os.makedirs(output, exist_ok=True)

    log.info(f"Writing output to {output}")
    log.info(f"Copying configuration to {output}")
    shutil.copy(args.config, output)
    if configuration.maintain_directory_structure:
        log.info(f"Attempting to maintain source directory structure")
    io, oo = output, output
    if configuration.visualization.active:
        io, oo = get_img_outputs(configuration)
        if oo is None:
            oo = io

    device = torch.device("cuda") if torch.cuda.is_available() and configuration.cuda else torch.device("cpu")
    # log.info(json.dumps(dict(configuration), indent=4))

    run_configuration = torch.load(configuration.model_path, map_location=device)
    species_options = run_configuration["hyper_parameters"]["opts"]


    log.info(f"Restoring model from {configuration.model_path}")

    model = AnimalClean.load_from_checkpoint(
        checkpoint_path=configuration.model_path,
        map_location=device
    )
    model.eval()

    log.info(model)

    if configuration.visualize:
        sp = signal.signal_proc()
    else:
        sp = None

    sr = species_options.dataset.sample_rate
    fmin = species_options.dataset.frequency.frequency_min
    fmax = species_options.dataset.frequency.frequency_max
    compression = species_options.dataset.frequency.compression
    n_fft = species_options.dataset.fft.n_fft
    hop_length = species_options.dataset.fft.hop
    n_freq_bins = species_options.dataset.feature_size.n_freq_bins
    min_max_normalize = species_options.dataset.normalization.method == "min_max"

    sequence_len = int(ceil(configuration.sequence_length * sr))

    hop = sequence_len

    t_decompr_f = T.Decompress(f_min=fmin, f_max=fmax, n_fft=n_fft, sr=sr, device=device)

    data_loader, concatenate = get_data_loader(configuration,
                                               sr=sr,
                                               fmin=fmin,
                                               fmax=fmax,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               n_freq_bins=n_freq_bins,
                                               sequence_len=sequence_len,
                                               hop=hop,
                                               min_max_normalize=min_max_normalize,
                                               freq_compression=compression)
    log.info(f"Data Loader Size: {len(data_loader)}")

    with torch.no_grad():
        for i, b in enumerate(data_loader):

            sample_spec_orig, batch, spec_cmplx, filename = b
            log.info(f"[{i}] :: {filename[0]}")
            batch = batch.to(device)
            try:
                denoised_output = model(batch)
            except RuntimeError:
                continue
            decompressed_net_out = t_decompr_f(denoised_output)
            spec_cmplx = spec_cmplx.squeeze(dim=0).to(device)
            decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)
            audio_spec = decompressed_net_out * spec_cmplx
            window = torch.hann_window(n_fft).to(device)
            audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)
            audio_spec = torch.view_as_complex(audio_spec)
            detected_spec_cmplx = spec_cmplx.squeeze(dim=0).transpose(0, 1)

            if sp is not None:
                input_name = os.path.join(io, f"net_in_spec_{filename[0].split('/')[-1].split('.')[0]}.png")
                output_name = os.path.join(oo, f"net_out_spec_{filename[0].split('/')[-1].split('.')[0]}.png")
                input_img = convert_tensor_to_PIL(batch.squeeze(), transpose=True)
                output_img = convert_tensor_to_PIL(denoised_output.squeeze(), transpose=True)
                input_img.save(input_name)
                output_img.save(output_name)
                # sp.plot_spectrogram(spectrogram=batch.squeeze(dim=0),
                #                     title="",
                #                     output_filepath=input_name,
                #                     sr=sr,
                #                     hop_length=hop_length,
                #                     fmin=fmin,
                #                     fmax=fmax,
                #                     show=False,
                #                     ax_title="spectrogram")
                #
                # sp.plot_spectrogram(spectrogram=denoised_output.squeeze(dim=0),
                #                     title="",
                #                     output_filepath=output_name,
                #                     sr=sr,
                #                     hop_length=hop_length,
                #                     fmin=fmin,
                #                     fmax=fmax,
                #                     show=False,
                #                     ax_title="spectrogram")

            if concatenate:
                audio_out_denoised = torch.istft(audio_spec,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 onesided=True,
                                                 center=True,
                                                 window=window)
                if total_audio is None:
                    total_audio = audio_out_denoised
                else:
                    total_audio = torch.cat((total_audio, audio_out_denoised), 0)
            else:
                total_audio = torch.istft(audio_spec,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          onesided=True,
                                          center=True,
                                          window=window)

                scipy.io.wavfile.write(
                    filename=get_file_out_name(filename, configuration),
                    rate=sr,
                    data=total_audio.detach().cpu().numpy().T)

        scipy.io.wavfile.write(
            filename=get_file_out_name(filename, configuration),
            rate=sr,
            data=total_audio.detach().cpu().numpy().T)
