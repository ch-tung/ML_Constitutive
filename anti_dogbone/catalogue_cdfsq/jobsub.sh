#!/bin/bash

# Slurm sbatch options
#SBATCH -o LOG_catalogue_cdf.log-%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=1G

# Loading the required module
source /etc/profile
module load anaconda/2020b
# source /home/gridsan/alegoupil/installdolfin/share/dolfin/dolfin.conf
source ~/installdolfin/share/dolfin/dolfin.conf

# Run the script
python -i ./Gap_score_batch.py

# Clean output_files
# bash remove_old_files.sh