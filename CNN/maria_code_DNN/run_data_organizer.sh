#!/usr/bin/env bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`

source /data/user/mvprado/DeepCore_DNN/ve_py3/bin/activate

bash /data/user/mvprado/icetray/python3.6/build/env-shell.sh << SCRIPT

python /data/user/mvprado/DeepCore_DNN/final_code/data_organizer.py -e /data/user/mvprado/DeepCore_DNN/HDF5/2D_files_cut10_inicepulses/NuE_coord/dc_arrays_oscNext* -m /data/user/mvprado/DeepCore_DNN/HDF5/2D_files_cut10_inicepulses/NuMu_coord/dc_arrays_oscNext* 

SCRIPT
