Tips for HPC users
==================

Since SWSPy is parallelised, it may be beneficial to process data on High Performance Computing (HPC) infrastructure for large datasets. Here is a little information on the arcitecture of SWSPy and a brief example on how to structure a HPC job.

SWSPy is parallelised in an embarissingly parallel fashion for performing the phi-dt grid search. Each process is therefore independent during the grid search. However, the processes come together at the end of each process, and so the code should be treated as a shared memory model rather than a distributed memory model. Crucially, this means that for any job, one should only submit to a maximum of one node. SWSPy does not support parallelisation across mulitple nodes for one job.

A simple example of a possible submission script is shown below. This script is written for a SLURM submission management system with the Anaconda package manager installed.

.. code-block:: bash

   # Setup SBATCH params (example only):
   SBATCH --nodes=1 # NOTE: swspy will only run on a single node
   SBATCH --ntasks-per-node=1
   SBATCH --cpus-per-task=48  # NOTE: Make sure number of NUMBA threads specified in swspy!
   SBATCH --mem=64000
   SBATCH --time=12:00:00
   SBATCH --job-name=swspy_run

   # Load python environment:
   module load Anaconda3
   source activate $DATA/swspy_env #Path to anaconda environment with swspy and all dependencies installed

   # Run SWSPy:
   # NOTE: Very important that the number of processors specified above (--cpus-per-task parameter)
   python swspy_run_script.py # Python script detailing specific commands for running swspy (see examples in tutorials).


