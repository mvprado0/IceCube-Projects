"""

This script takes all the fit files for all theta23 values and organizes them into a single JSON file
for plotting and for keeping better track of all the results.

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""

import os, datetime, collections, copy
import json
import re
import argparse

def extract_values(inputfiles, metric_file, metric_alt_file, octant_file):

    for f in inputfiles:
        
        with open(f, "r") as fv:
            data = json.load(fv)
        
        # Load results of the fit
        metric_val = data["metric_fit"]
        metric_val_alt = data["metric_alternate_fit"]
        theta23_bf = data["theta23_bf"]

        # Determine type of fit
        if "0.json" in f:
            fit_type = 0 
        elif "1.json" in f:
            fit_type = 1 
        elif "2.json" in f:
            fit_type = 2 
        elif "3.json" in f:
            fit_type = 3

        # Determine the mass ordering
        if "nh" in f:
            mass_order = "nh"
        elif "ih" in f:
            mass_order = "ih"

        # Determine the metric
        if "llh" in f:
            metric = "llh"
        elif "conv_llh" in f:
            metric = "conv_llh"
        elif "chi2" in f:
            metric = "chi2"
        elif "mod_chi2" in f:
            metric = "mod_chi2"

        if (metric=="llh" or metric=="conv_llh"):
            metric_val = -1*metric_val
            metric_val_alt = -1*metric_val_alt
   
        # Determine the truth value of theta23
        result = re.search(metric + "_(.*)degree", f)
        theta23 = result.group(1)
    
        store_metric_val(metric_file, mass_order, theta23, metric_val, fit_type)
        store_metric_val(metric_alt_file, mass_order, theta23, metric_val_alt, fit_type)

        if (fit_type==1 or fit_type==2):
            store_octant(octant_file, mass_order, theta23, theta23_bf, fit_type)

        # Deletes single result files after storing the result in the output file
        #os.remove(f)

def store_metric_val(mfile, mass_order, theta23, val, fit_type):
    
    # Sort metric values by category (fit type and mass ordering) and store them in a JSON file
    fit_names_NO = ['NO_NO', 'NO_IO_bf', 'IO_bf_NO', 'IO_bf_IO_bf']
    fit_names_IO = ['IO_IO', 'IO_NO_bf', 'NO_bf_IO', 'NO_bf_NO_bf']
    
    if (mass_order=='nh'):
        fit = fit_names_NO[fit_type]
    elif (mass_order=='ih'):
        fit = fit_names_IO[fit_type]

    names_total = fit_names_NO + fit_names_IO 

    # Pair up metric value with corresponding theta23 truth value
    try:
        with open(mfile) as f:
            data = json.load(f)
        
        data[fit].update( { str(theta23) : val } )

    except FileNotFoundError:
        # First fit of this kind
        data = {}
        for label in names_total:
            if (label==fit):
                data.update( { fit : { str(theta23) : val } } )
            else:
                data.update( { label : { } } )
    
    with open(mfile,'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def store_octant(ofile, mass_order, theta23_true, theta23_fit, fit_type):
    
    # Sort fitted octant by category (fit type and mass ordering) and store them in a JSON file
    fit_names_NO = ['NO_IO_bf', 'IO_bf_NO']
    fit_names_IO = ['IO_NO_bf', 'NO_bf_IO']
    
    if (mass_order=='nh'):
        fit = fit_names_NO[fit_type-1]
    elif (mass_order=='ih'):
        fit = fit_names_IO[fit_type-1]

    names_total = fit_names_NO + fit_names_IO

    # Pair up fitted octant with corresponding theta23 truth value
    try:
        with open(ofile) as f:
            data = json.load(f)

        data[fit] = octant_info(theta23_fit, theta23_true, data[fit])
    
    except FileNotFoundError:
        # First fit of this kind
        data = {}
        oct_dict = {}
    
        oct_dict = octant_info(theta23_fit, theta23_true, oct_dict)

        for label in names_total:
            if (label==fit):
                data.update( { fit : oct_dict } )
            else:
                data.update( { label : { } } )
    
    with open(ofile,'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def octant_info(theta23bf, theta23, oct_dict):

    if(theta23bf <= 45.0):
        if (float(theta23) <= 45.0):
            # Fit picked the same original octant
            oct_dict.update( { str(theta23) : "bf_correct_oct" } )
        elif (float(theta23) > 45.0):
            oct_dict.update( { str(theta23) : "bf_wrong_oct" } )
    elif(theta23bf > 45.0):
        if (float(theta23) <= 45.0):
            # Fit picked the opposite octant to the original octant
            oct_dict.update( { str(theta23) : "bf_wrong_oct" } )
        elif (float(theta23) > 45.0):
            oct_dict.update( { str(theta23) : "bf_correct_oct" } )

    return oct_dict

# Main

parser = argparse.ArgumentParser(description='Import and sort NMO fit result files into one file.')
parser.add_argument('-i', dest='inputfiles', nargs='+', help='JSON files that hold the fitted test statistic results')
parser.add_argument('-fd', dest='fit_directory', type=str, help='Directory to store output files') 

args = parser.parse_args()

metric_file = str(args.fit_directory) + "/metric_file.json"
metric_alt_file = str(args.fit_directory) + "/alternate_metric_file.json"
octant_file = str(args.fit_directory) + "/octant_file.json"

# Sort and store all metric best fit result values in one output file.
# Sort and store all metric alternate result values in a separate output file.
# Sort and store all fitted octant choices in a separate output file.
extract_values(args.inputfiles, metric_file, metric_alt_file, octant_file)
