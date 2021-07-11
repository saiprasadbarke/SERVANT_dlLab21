#!/bin/sh
#cd dl_lab_project
source project-venv/bin/activate
mkdir -p jobs

first=1500
last=2000
step=25

for arg in $(seq $first $step $last)
do
 echo "python generate_mlp_network_eqaution_dataset.py $arg" > "jobs/job_"$arg"_$(expr "$arg" + "25").sh"
 msub -N "$arg"_$(expr "$arg" + "25") -l nodes=3:ppn=5,walltime=36:00:00,pmem=6gb "jobs/job_"$arg"_$(expr "$arg" + "25").sh"
done
