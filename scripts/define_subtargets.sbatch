#!/bin/bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --mem=0
#SBATCH --partition=prod
#SBATCH --account=proj83
#SBATCH --output="topochar-%x-%j.out"
#SBATCH --error="topochar-%x-%j.err"

#module list
#module load archive/2020-08
#module load py-bluepy/2.0.0

source /gpfs/bbp.cscs.ch/project/proj83/home/sood/venv.sh

echo "define_subtargets.py " "$@"
python define_subtargets.py "$@"
