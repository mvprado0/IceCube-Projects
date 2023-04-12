"""

This script takes all the Pseudotrial fit files for all theta23 values and indeces and organizes them into two JSON files
for plotting and for keeping better track of all the results. 

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""

import os, datetime, collections, copy
import json
import re
import argparse
import ntpath

def search_val(input_str, f):

    # Determine quantity from input file string
    result = re.search(input_str, f)
    quantity = result.group(1)

    return quantity

def extract_values(inputfiles, metric_file, dmetric_file):

    # Input files are in the form: /directory_where_stored/fit_results_(massorder)_(metric)_(theta23)degree_(fittype)_index(#).json
    # Want to take the difference between metric values with different fittypes: 
    
    for f in inputfiles:
        
        # Determine type of fit
        fit_type = int(search_val("degree_(.*)_index", f))
        
        # Determine the mass ordering
        if "nh" in f:
            mass_order = "nh"
        elif "ih" in f:
            mass_order = "ih"

        # Determine index
        index = int(search_val("index(.*).json", f))

        # Determine the metric
        if "llh" in f:
            metric = "llh"
        elif "conv_llh" in f:
            metric = "conv_llh"
        elif "chi2" in f:
            metric = "chi2"
        elif "mod_chi2" in f:
            metric = "mod_chi2"

        # Determine the truth value of theta23
        theta23 = float(search_val(metric + "_(.*)degree", f))
        
        with open(f, "r") as fv:
            data = json.load(fv)
            
        # Load results of the fit
        metric_val = data["metric_fit"]
        
        if (fit_type == 0) or (fit_type == 2): 
        
            # File for fit with fittype that we need to get the dmetric
            head, tail = ntpath.split(f)
            g = head + "/fit_results_" + mass_order + "_" + metric + "_" + str(theta23) + "degree_" + str(fit_type + 1) + "_index" + str(index) + ".json"
      
            try: 
                with open(g, "r") as gv:
                    data2 = json.load(gv)
           
            except FileNotFoundError:
                store_values(metric_file, mass_order, theta23, metric_val, fit_type, index)

            else:
                # Load results of the fit
                metric_val2 = data2["metric_fit"]

                # Convention adopted follows NO-IO for the dmetric
                if (mass_order == "nh"): 
                    dmetric_val = metric_val - metric_val2
                elif (mass_order == "ih"):  
                    dmetric_val = metric_val2 - metric_val
               
                # Store both the metric and the dmetric
                store_values(metric_file, mass_order, theta23, metric_val, fit_type, index, dmfile=dmetric_file, dmval=dmetric_val)
        else:
            # Store only the metric to not double count the dmetric being stored with fit_types 0 and 2
            store_values(metric_file, mass_order, theta23, metric_val, fit_type, index)

def dump_file_content(vfile, val, index, fit, theta23, names_total):        
    
    ind_dict = { str(index) : val }

    # Pair up metric value with corresponding fit, theta23 truth, and index values
    try:
        with open(vfile) as f:
            data = json.load(f)
    
    except FileNotFoundError:
    
        # If this is the first fit of this kind
        data = {}
        for label in names_total:
            if (label==fit):
                data.update( { fit : { str(theta23) : ind_dict } } )
            else:
                data.update( { label : { } } )
    else:
        try:
            # If there is an entry already with that fit and theta23 value
            data[fit][str(theta23)].update( ind_dict )
        
        except KeyError:
        
            # If there is no entry with that fit and theta23 value, then try just the fit value
            data[fit].update( { str(theta23) : ind_dict } )
       
    with open(vfile,'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def store_values(mfile, mass_order, theta23, mval, fit_type, index, dmfile=None, dmval=None):

    fit_names_NO_metric = ['NO_NO', 'NO_IO_bf', 'IO_bf_NO', 'IO_bf_IO_bf']
    fit_names_IO_metric = ['IO_IO', 'IO_NO_bf', 'NO_bf_IO', 'NO_bf_NO_bf']
    
    # Sort metric values by category (fit type and mass ordering) and store them in a JSON file
    fit_names_NO_dmetric = ['NO', 'IO_bf']
    fit_names_IO_dmetric = ['IO', 'NO_bf']
    
    if (fit_type == 1) or (fit_type == 3):
        if (mass_order=='nh'):
            fit_m = fit_names_NO_metric[fit_type]
        elif (mass_order=='ih'):
            fit_m = fit_names_IO_metric[fit_type]
    
        names_total_m = fit_names_NO_metric + fit_names_IO_metric
        dump_file_content(mfile, mval, index, fit_m, theta23, names_total_m)
    else:
        # Fittype can only be 0 or 2 but for the dmetric we need 0 to correspond to index 0 and 2 to correspond to index 1, so we divide fittype by 2 to achieve this
        if (mass_order=='nh'):
            fit_m = fit_names_NO_metric[fit_type]
            fit_dm = fit_names_NO_dmetric[int(fit_type/2)]
        elif (mass_order=='ih'):
            fit_m = fit_names_IO_metric[fit_type]
            fit_dm = fit_names_IO_dmetric[int(fit_type/2)]

        names_total_m = fit_names_NO_metric + fit_names_IO_metric
        names_total_dm = fit_names_NO_dmetric + fit_names_IO_dmetric

        dump_file_content(mfile, mval, index, fit_m, theta23, names_total_m)

        if (dmfile != None) and (dmval != None):
            dump_file_content(dmfile, dmval, index, fit_dm, theta23, names_total_dm)

# Main

parser = argparse.ArgumentParser(description='Import and sort NMO fit result files into one file.')
parser.add_argument('-i', dest='inputfiles', nargs='+', help='Full path to JSON files that hold the fitted test statistic results')
parser.add_argument('-fd', dest='fit_directory', type=str, help='Directory to store output files') 

args = parser.parse_args()

metric_file = str(args.fit_directory) + "/pt_metric_file.json"
dmetric_file = str(args.fit_directory) + "/pt_dmetric_file.json"

# Sort and store all metric and dmetric best fit result values in two different output files
extract_values(args.inputfiles, metric_file, dmetric_file)
