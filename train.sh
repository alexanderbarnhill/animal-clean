#!/bin/bash

#SBATCH --job-name=ACLEAN
#SBATCH --ntasks=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /cluster/%u/logs/%j.out
#SBATCH -e /cluster/%u/logs/%j.out
#SBATCH --signal=SIGUSR1@90


while getopts s: flag
do
    case "${flag}" in
        s) species=${OPTARG};;
    esac
done


config_file=${species}.yaml

echo "Taking configuration from ${config_file}"
source_dir="/cluster/barnhill/code/animal-clean"
configuration_directory=$source_dir/configuration
species_configuration_directory=$configuration_directory/species
species_configuration_file=$species_configuration_directory/$config_file


export CUDA_VISIBLE_DEVICES=0,1
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"


cluster=/cluster/barnhill
env_dir=$cluster/env
mkdir -p $env_dir
pytorch_version=1.7.1
torchvision_version=0.8.2
pytorch_env=$(hostname)_pytorch-$pytorch_version

virtualenv --system-site-packages -p python3 $env_dir/$pytorch_env
source $env_dir/$pytorch_env/bin/activate

cd $source_dir || exit

echo "Installing requirements"
pip3 install lightning wandb torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 torchmetrics==1.3.0


wandb_key=bf8463a6e7024aac12d6fb224bb1fe5d155cc679
wandb login $wandb_key

python3 -W ignore::UserWarning $source_dir/train.py \
--defaults $configuration_directory \
--species_configuration $species_configuration_file
