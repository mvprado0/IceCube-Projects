#!/usr/bin/env bash

export KERAS_BACKEND="tensorflow"
export SINGULARITY_TMPDIR=/data/user/mvprado/tmp
export SINGULARITY_CACHEDIR=/data/user/mvprado/cache

singularity exec --nv -B /home/mvprado/:/home/mvprado/ -B /mnt/lfs7/user/:/data/user/ -B /mnt/lfs6/ana/:/data/ana/ -B /mnt/lfs6/sim/:/data/sim/ /data/user/mvprado/DeepCore_DNN/GPU_container/icetray_combo-stable-tensorflow.1.15.0-ubuntu18.04.sif /usr/local/icetray/env-shell.sh python /data/user/mvprado/DeepCore_DNN/2D_training.py -b $1 -p $2 -c $3 -c2 $4 -d $5 -d2 $6 -j $7  

