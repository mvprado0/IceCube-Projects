"""

This script performs four different types of fits as part of the 
NMO Asimov analysis (without adding statistical fluctuations). Each
of these fit results are stored separately in a JSON file for later sorting and organizing.

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""

import os, datetime, collections, copy
import numpy as np
import json
import argparse
from pisa import ureg
from pisa.utils.fileio import mkdir, from_file, to_file
from pisa.core.param import ParamSet
from pisa.core.distribution_maker import DistributionMaker
from pisa.analysis import analysis_alternate_fit_1crs
from pisa.analysis.analysis_alternate_fit_1crs import update_param_values
from importlib import reload
from pisa.utils.log import logging, set_verbosity

def global_fit(template, hypo, metric, minimizer_settings):

    # Minimization class
    reload(analysis_alternate_fit_1crs)
    ana = analysis_alternate_fit_1crs.BasicAnalysis()
    set_verbosity(1)

    best_fit_info, alternate_fits = ana.fit_recursively(
        template,
        hypo,
        metric,
        external_priors_penalty=None,
        **minimizer_settings
    )

    return best_fit_info, alternate_fits

def fit_and_store(truth, hypo, metric, minimizer_settings, datafile, param_file=None, index=None):

    results_dict = {}

    if (index == None):
        print("Entering Asimov fit")
    else:
        print("Entering PT fit")

    # Begin mininizer fit
    start_time = datetime.datetime.now()
    
    # Call minimizer function
    fit_results, alternate = global_fit(truth, hypo, [metric], minimizer_settings)
    
    end_time = datetime.datetime.now()
    print ("\nFit finished! Took %s" % (end_time-start_time))
    
    # Minimized test statistic result
    ts = fit_results["metric_val"]
    ts_alt = alternate[0]["metric_val"]

    if (metric=='llh' or metric=='conv_llh'):
        ts = -1*ts
        ts_alt = -1*ts_alt

    # Best fit value of the theta23 parameter after minimization
    theta23_bf = fit_results["params"]["theta23"].value.magnitude
    deltam31_bf = fit_results["params"]["deltam31"].value.magnitude
    
    results_dict.update( { "metric_fit" : ts } )
    results_dict.update( { "metric_alternate_fit" : ts_alt } )
    results_dict.update( { "theta23_bf" : theta23_bf } )
    results_dict.update( { "deltam31_bf" : deltam31_bf } )

    if (index != None):
        results_dict.update( { "index" : index } )

    with open(datafile,'w') as f:
        json.dump(results_dict, f, indent=2, sort_keys=True)
   
    if (param_file != None):
        # Store best fit parameters 
        fit_results["params"].to_json(param_file)

    print("Fit done")

def no_systematics(dist_maker):
  
    # Fixes all systematic parameters
    free_params = ['theta23','deltam31']

    for pipeline in dist_maker:
        for i in range(len(pipeline.params)):
            if (pipeline.params[i].name in free_params):
                pipeline.params[i].is_fixed = False 
            else:
                pipeline.params[i].is_fixed = True

    return dist_maker

def reset_data(template_maker, template_maker_bf, param_file):

    # Reset both template makers to their nominal values at the start of a fit
    template_maker.reset_free()
    template_maker_bf.reset_free()

    # For extra caution 
    update_param_values(template_maker_bf, ParamSet.from_json(param_file), update_nominal_values=True, update_range=True)

    return template_maker, template_maker_bf 

def set_theta23(dist_maker, theta23):
    
    # Set truth value of theta23
    params = ['theta23']
    
    for pipeline in dist_maker:
        for i in range(len(pipeline.params)):
            if (pipeline.params[i].name in params):
                pipeline.params[i].value = theta23 * ureg.degree
    
    print(dist_maker.params["theta23"].value)
    return dist_maker
   
def get_data(pipeline_nu, pipeline_nu_other, pipeline_mu, theta23, param_file, minimizer_settings=None, metric=None, datafile=None, fit_type=None, index=None, no_systs=False):

    template_maker = DistributionMaker([pipeline_nu, pipeline_mu])
    template_maker_bf = DistributionMaker([pipeline_nu_other, pipeline_mu])
  
    # For testing/checking purposes, we can fix all the systematic parameters
    if(no_systs==True):
        template_maker = no_systematics(template_maker)
        template_maker_bf = no_systematics(template_maker_bf)

    # Truth event distribution
    template_maker = set_theta23(template_maker, theta23)   
    asimov = template_maker.get_outputs(return_sum=True)
  
    if (index==None):
        if (fit_type==1):
            # Fit opposite ordering to truth and store test statistic values as well as the best fit parameters
            fit_and_store(asimov, template_maker_bf, metric, minimizer_settings, datafile, param_file=param_file)
        else:
            # Opposite ordering "truth" best fit event distributions
            update_param_values(template_maker_bf, ParamSet.from_json(param_file), update_nominal_values=True, update_range=True)
            asimov_bf = template_maker_bf.get_outputs(return_sum=True)

            return template_maker, template_maker_bf, asimov, asimov_bf
    else:
        # Opposite ordering "truth" best fit event distributions
        update_param_values(template_maker_bf, ParamSet.from_json(param_file), update_nominal_values=True, update_range=True)
        asimov_bf = template_maker_bf.get_outputs(return_sum=True)
        
        # Add fluctuations to the maps in the set and return as a new MapSet
        pseudo_true = asimov.fluctuate(method='poisson', random_state=0, jumpahead=index)
        pseudo_bf = asimov_bf.fluctuate(method='poisson', random_state=0, jumpahead=index)

        return template_maker, template_maker_bf, pseudo_true, pseudo_bf

# Main

parser = argparse.ArgumentParser(description='Run NMO Asimov fits in parallel.')
parser.add_argument('-th', dest='theta23_truth', type=float, help='Truth value of theta23 parameter')
parser.add_argument('-o', dest='mass_order', type=str, help='True mass ordering')
parser.add_argument('-m', dest='metric', type=str, help='Test statistic for minimization')
parser.add_argument('-f', dest='fit_type', type=int, help='Choice of hypo-to-template fit') 
parser.add_argument('-z', dest='index', type=int, required=False, default=None, help='Index of the statistical fluctuation of the data template')
parser.add_argument('-ns', dest='no_systs', action='store_true', required=False, help='Choice to fix systematics')  
parser.add_argument('-no', dest='no_pipeline', type=str, help='Normal Ordering pipeline')
parser.add_argument('-io', dest='io_pipeline', type=str, help='Inverted Ordering pipeline')
parser.add_argument('-mp', dest='muon_pipeline', type=str, help='Muon pipeline')
parser.add_argument('-bfd', dest='bf_directory', type=str, help='Directory where the best fit parameter file is found') 

args = parser.parse_args()

# When adding statistical fluctuations, need to create subdirectory named "pt_files"
datafile = str(args.bf_directory) + "/fit_results_" + str(args.mass_order) + "_" + str(args.metric) + "_" + str(args.theta23_truth) + "degree_" + str(args.fit_type) + ".json"
datafile_pt = str(args.bf_directory) + "/pt_files/fit_results_" + str(args.mass_order) + "_" + str(args.metric) + "_" + str(args.theta23_truth) + "degree_" + str(args.fit_type) + "_index" + str(args.index) + ".json"
syst_param_file = str(args.bf_directory) + "/bestfit_params_fittype_" + str(args.fit_type) + "_true_" + str(args.mass_order)+ "_trueth23_" + str(args.theta23_truth) + "degree_"+ str(args.metric) + ".json"
bf_param_file = str(args.bf_directory) + "/bestfit_params_fittype_1_true_" + str(args.mass_order)+ "_trueth23_" + str(args.theta23_truth) + "degree_"+ str(args.metric) + ".json"

minimizer_settings = "/data/user/mvprado/upgrade_nmo/fridge/analysis/oscNext_NMO_2/settings/minimizer/minimizer_crs2_minuit.json"
minimizer_settings = from_file(minimizer_settings)

if (args.mass_order=='nh'):
    pipeline_nu = args.no_pipeline
    pipeline_nu_other = args.io_pipeline
elif (args.mass_order=='ih'):
    pipeline_nu = args.io_pipeline
    pipeline_nu_other = args.no_pipeline

#Asimov fit
if (args.index == None):
    if (args.fit_type==1):
        # Fit opposite ordering to truth ordering and save fitted parameters to use in fits 2 and 3 
        get_data(pipeline_nu, pipeline_nu_other, args.muon_pipeline, args.theta23_truth, bf_param_file, minimizer_settings=minimizer_settings, metric=args.metric, datafile=datafile, fit_type=args.fit_type, no_systs=args.no_systs)    

    else:
        # Obtain truth and hypo templates
        template_maker_true, template_maker_bf, asimov, asimov_bf = get_data(pipeline_nu, pipeline_nu_other, args.muon_pipeline, args.theta23_truth, bf_param_file, no_systs=args.no_systs)    

        # Reset free parameters to their nominal values
        template_maker_true, template_maker_bf = reset_data(template_maker_true, template_maker_bf, bf_param_file)

        if (args.fit_type==0):
            fit_and_store(asimov, template_maker_true, args.metric, minimizer_settings, datafile, param_file=syst_param_file)
        elif (args.fit_type==2):
            fit_and_store(asimov_bf, template_maker_true, args.metric, minimizer_settings, datafile, param_file=syst_param_file)
        elif (args.fit_type==3):
            fit_and_store(asimov_bf, template_maker_bf, args.metric, minimizer_settings, datafile, param_file=syst_param_file)

else:
    # Obtain truth and hypo templates
    template_maker_true, template_maker_bf, pseudo_true, pseudo_bf = get_data(pipeline_nu, pipeline_nu_other, args.muon_pipeline, args.theta23_truth, bf_param_file, index=args.index, no_systs=args.no_systs)    

    # Reset free parameters to their nominal values
    template_maker_true, template_maker_bf = reset_data(template_maker_true, template_maker_bf, bf_param_file)

    if (args.fit_type==0):
        fit_and_store(pseudo_true, template_maker_true, args.metric, minimizer_settings, datafile_pt, index=args.index)
    elif (args.fit_type==1):
        fit_and_store(pseudo_true, template_maker_bf, args.metric, minimizer_settings, datafile_pt, index=args.index)
    elif (args.fit_type==2):
        fit_and_store(pseudo_bf, template_maker_true, args.metric, minimizer_settings, datafile_pt, index=args.index)
    elif (args.fit_type==3):
        fit_and_store(pseudo_bf, template_maker_bf, args.metric, minimizer_settings, datafile_pt, index=args.index)
