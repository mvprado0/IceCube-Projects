#!/usr/bin/env bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`

source /data/user/mvprado/DeepCore_DNN/ve_py3/bin/activate

bash /data/user/mvprado/icetray/python3.6/build/env-shell.sh << SCRIPT

python /data/user/mvprado/DeepCore_DNN/final_code/2D_channels_arrays.py -i /data/ana/LE/oscNext/pass2/genie/level7_v02.00/${1}/oscNext_genie_level7_v02.00_pass2.${1}.${2}.i3.zst -p SRTInIcePulses -a $3

SCRIPT


