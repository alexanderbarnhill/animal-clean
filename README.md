# ANIMAL-CLEAN
An animal-independent deep denoising, segmentation, and classification toolkit



# Setup
## Location
Due to potential differences in directory configurations when training on a computing cluster or locally and the desire to keep those settings different, there are two `location` parameters used here.
- `local`
- `cluster`

**ANIMAL--CLEAN automatically looks for the `SLURM_JOB_ID` environment variable to determine if the script is being run in slurm. In this case, the `cluster` settings are used.**
## Species Configuration

The species [configurations](configuration/species) are available for
- [Pygmy pipistrelle](configuration/species/bat.yaml)
- [Chimpanzee](configuration/species/chimp.yaml)
- [Atlantic Cod](configuration/species/cod.yaml)
- [Killer Whale](configuration/species/orca.yaml)
- [Monk Parakeet](configuration/species/parakeet.yaml)
- [Primate](configuration/species/primate.yaml)
- [Warbler](configuration/species/warbler.yaml)


### Data Setup
Found in the [defaults](configuration/defaults.yaml) (overridden by the species if selected)
```    
data_directory: <path to directory with target signals>
noise_sources:
  train: <path to directory with noise files for training>
  val: <path to directory with noise files for validation>
  test: <path to directory with noise files for testing>
human_speech:
  clean: <path to clean human speech files>
  noisy: <path to noisy human speech files>
  batch_percentage: <float between 0 and 1 indicating the percentage of the batch which should be filled with human speech>
working_directory: <(optional) working directory for target signals if paths not absolute> Default: None
cache_directory: <(optional) directory to cache spectrograms created during training> Default: None
```

## Training
Data settings are controlled with the `species` settings (see above).

## Prediction


# Citation
```
@inproceedings{BarnhillACLEAN:2024,
	author={Alexander Barnhill and Elmar Nöth and Andreas Maier and Christian Bergler},
	title={{ANIMAL-CLEAN -- A Deep Denoising Toolkit for Animal-Independent Signal Enhancement}}},
	year=2024,
	booktitle={Proc. Interspeech 2024},
}
```