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

venv_directory=/cluster/barnhill/code/animal-vae/venv
export CUDA_VISIBLE_DEVICES=0,1
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"


if [ ! -d "$venv_directory" ]
then
  echo "Virtual environment does not exist. Setting it up."
  python3 -m venv $venv_directory

fi

source $venv_directory/bin/activate
cd $source_dir || exit

echo "Installing requirements"
pip3 install -r requirements.txt


wandb_key=bf8463a6e7024aac12d6fb224bb1fe5d155cc679
wandb login $wandb_key

python3 -W ignore::UserWarning $source_dir/train.py \
--defaults $configuration_directory \
--species_configuration $species_configuration_file
