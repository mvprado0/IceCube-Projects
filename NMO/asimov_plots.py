"""

Plotting script for NMO Asimov sensitivity performance and other components of the sensitivity. 
May optionally superimppose GRECO NMO sensitivity in the same plot. Stores all plots in one PDF file.

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""

import os, datetime, collections, copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import json
import argparse
from utils.plotting.standard_modules import *
from pisa import ureg

def greco_plots():

    # GRECO sensitivities for comparison
    x_gno = []
    y_gno = []
    x_gio = []
    y_gio = []

    j = open('greco_NO_sensitivity.txt','r')
    k = open('greco_IO_sensitivity.txt','r')

    lines_no = j.readlines()
    lines_io = k.readlines()

    for line in lines_no:
        x_gno.append(float(line.split(',')[0]))
        y_gno.append(float(line.split(',')[1]))
    for line in lines_io:
        x_gio.append(float(line.split(',')[0]))
        y_gio.append(float(line.split(',')[1]))

    return x_gno, y_gno, x_gio, y_gio

def asimov_vals(theta23list, metric_asimov_file, metric):

    with open(metric_asimov_file) as f:
        data = json.load(f)

    # Order of metric lists stored in metric_all: NO_NO [position 0], IO_bf_NO [position 1], IO_bf_NO [position 2]...etc
    fit_names = ["NO_NO", "NO_IO_bf", "IO_bf_NO", "IO_bf_IO_bf", "IO_NO_bf", "IO_IO", "NO_bf_NO_bf", "NO_bf_IO"]

    metric_all = []
    dmetric_all = []
    xvals_asimov = []

    # Create list of lists with all metric values in specific order for plotting 
    for i, fit in enumerate(fit_names):
        metric_fit = []
        for t, theta in enumerate(theta23list):

            metric_fit.append(data[fit][str(theta)])

            if (i==0):
                th_degree = theta * ureg.degree
                xval = np.sin(th_degree.m_as('radian'))**2
                xvals_asimov.append(xval)

        metric_all.append(metric_fit)

    # Order of dmetric lists stored in dmetric_all: NO [position 0], IO_bf [position 1], IO [position 2], NO_bf [position 3]
    for i in range(int(len(fit_names)/2)): 
        dmetric_asimov = [a - b for a, b in zip(metric_all[2*i], metric_all[2*i+1])]
        dmetric_all.append(dmetric_asimov)

    # Sensitivities
    sens_NO, NA_, sd_NO, _ = sensitivity(metric, obs_dmetric=dmetric_all[0], dmetric_NO=dmetric_all[0], dmetric_IO=dmetric_all[1])
    NA_, sens_IO, _, sd_IO = sensitivity(metric, obs_dmetric=dmetric_all[2], dmetric_NO=dmetric_all[3], dmetric_IO=dmetric_all[2])
    
    simple_sens_NO = simple_sensitivity(metric, obs_dmetric=dmetric_all[0])
    simple_sens_IO = simple_sensitivity(metric, obs_dmetric=dmetric_all[2])

    return xvals_asimov, metric_all, dmetric_all, sens_NO, sens_IO, simple_sens_NO, simple_sens_IO

def sensitivity(metric, obs_dmetric, dmetric_NO, dmetric_IO):

    if (type(obs_dmetric)==list): 
        # Standard deviation
        if (metric=='llh' or metric=='conv_llh'):
            sd_NO = [math.sqrt(2*abs(a)) for a in dmetric_IO]
            sd_IO = [math.sqrt(2*abs(a)) for a in dmetric_NO]
        elif(metric=='chi2' or metric=='mod_chi2'):
            sd_NO = [2*math.sqrt(abs(a)) for a in dmetric_IO]
            sd_IO = [2*math.sqrt(abs(a)) for a in dmetric_NO]

        # Sensitivity calculation
        sensitivity_NO = [(b - a)/c for a, b, c in zip(obs_dmetric, dmetric_IO, sd_NO)]
        sensitivity_IO = [(a - b)/c for a, b, c in zip(obs_dmetric, dmetric_NO, sd_IO)]
    else:
        if (metric=='llh' or metric=='conv_llh'):
            sd_NO = math.sqrt(2*abs(dmetric_IO))
            sd_IO = math.sqrt(2*abs(dmetric_NO))
        elif(metric=='chi2' or metric=='mod_chi2'):
            sd_NO = 2*math.sqrt(abs(dmetric_IO))
            sd_IO = 2*math.sqrt(abs(dmetric_NO))
        
        sensitivity_NO = (dmetric_IO - obs_dmetric)/sd_NO
        sensitivity_IO = (obs_dmetric - dmetric_NO)/sd_IO

    return sensitivity_NO, sensitivity_IO, sd_NO, sd_IO

def simple_sensitivity(metric, obs_dmetric):

    # Sensitivity only using one of the dmetric distributions
    if (type(obs_dmetric)==list): 
        if (metric=='llh' or metric=='conv_llh'):
            sensitivity = [math.sqrt(2*abs(a)) for a in obs_dmetric]
        elif(metric=='chi2' or metric=='mod_chi2'):
            sensitivity = [math.sqrt(abs(a)) for a in obs_dmetric]
    else:
        if (metric=='llh' or metric=='conv_llh'):
            sensitivity = math.sqrt(2*abs(obs_dmetric))
        elif(metric=='chi2' or metric=='mod_chi2'):
            sensitivity = math.sqrt(abs(obs_dmetric))

    return sensitivity

def octant_dependency(alternate_file, octant_info_file, theta23list, dmetric_all, metric, type_label=None, means_list=None, meansx_list=None):
    
    with open(alternate_file) as f:
        alt_data = json.load(f)
    with open(octant_info_file) as f:
        oct_data = json.load(f)

    oct_names = ["NO_IO_bf", "IO_bf_NO", "IO_NO_bf", "NO_bf_IO"]
    oct_all = []
    c_oct_all = []
    w_oct_all = []

    # Order of alt_dmetric_all is ["NO", "IO_bf", "IO", "NO_bf"]
    xval, alt_metric_all, alt_dmetric_all, _, _, _, _ = asimov_vals(theta23list, alternate_file, metric)

    # Make list of lists of octant choice for all minimizer picks for all fits
    for fit in oct_names:
        oct_fit = []
        for t, theta in enumerate(theta23list):
            oct_fit.append(oct_data[fit][str(theta)])

        oct_all.append(oct_fit)
      
    # Assemble lists of correct and wrong octant values for all corresponding metric fits 
    for i in range(len(oct_names)):
        c_oct, w_oct = octant_curves(oct_all[i], metric_all[int(((i**2)+i+2)/2)], alt_metric_all[int(((i**2)+i+2)/2)])
        c_oct_all.append(c_oct)
        w_oct_all.append(w_oct)
  
    # For true NO and true NO bf fits, multiply each value by -1 to obtain correct sign for dmetric values
    # This means using zero for NO_NO, IO_IO, NO_bf_NO_bf, and IO_bf_IO_bf best fits, which is a very good approximation
    for i in [0, 3]:
        c_oct_all[i] = [-1*x for x in c_oct_all[i]]
        w_oct_all[i] = [-1*x for x in w_oct_all[i]]

    if (type_label==None):
        make_dmetric_octant_plot(xval, dmetric_all[0], c_oct_all[0], w_oct_all[0], dmetric_all[2], c_oct_all[2], w_oct_all[2], metric)
    else:
        make_dmetric_octant_plot(xval, dmetric_all[0], c_oct_all[0], w_oct_all[0], dmetric_all[2], c_oct_all[2], w_oct_all[2], metric, type_label, means_list, meansx_list)

def make_sensitivity_plot(xval, sens_NO, sens_IO, simple_sens_NO, simple_sens_IO, simple_sens=False, greco=False, two_sensitivities=False, theta23list=None, asimov_file2=None, metric=None):

    # Make sensitivity plot
    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid()

    if (simple_sens==False):
        ax.plot(xval, sens_NO, label='9.3 years, true NO', lw=4.5, color='red')
        ax.plot(xval, sens_IO, label='9.3 years, true IO', lw=4.5, color='blue')
    else:
        ax.plot(xval, simple_sens_NO, label='9.3 years, true NO', lw=4.5, color='red')
        ax.plot(xval, simple_sens_IO, label='9.3 years, true IO', lw=4.5, color='blue')

    if (two_sensitivities==True):
        xval2, metric_all2, dmetric_all2, sens_NO2, sens_IO2, simple_sens_NO2, simple_sens_IO2 = asimov_vals(theta23list, asimov_file2, metric)
        if (simple_sens==False):
            ax.plot(xval2, sens_NO2, label='Test, true NO', lw=4.5, color='red', linestyle='dashed')
            ax.plot(xval2, sens_IO2, label='Test, true IO', lw=4.5, color='blue', linestyle='dashed')
        else:
            ax.plot(xval2, simple_sens_NO2, label='Test, true NO', lw=4.5, color='red')
            ax.plot(xval2, simple_sens_IO2, label='Test, true IO', lw=4.5, color='blue')

    if (greco==True):
        x_gno, y_gno, x_gio, y_gio = greco_plots()
        ax.plot(x_gno, y_gno, label='3 years, true NO', zorder=3, lw=3, color='red', linestyle='dashed')
        ax.plot(x_gio, y_gio, label='3 years, true IO', zorder=3, lw=3, color='blue', linestyle='dashed')

    ax.set_xlabel(r'$\sin^2 \theta_{23}$', fontsize='x-large')
    ax.set_ylabel(r'$\eta_{\sigma}$', fontsize='x-large')
    plt.title("DeepCore NMO Asimov Sensitivity", fontsize=25)

    ax.legend(loc='upper left', fontsize=16, ncol=1)

def make_dmetric_asimov_plot(xval, dmetric_all, metric, means_list=None, meansx_list=None, err_list=None):

    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid()

    # Order of dmetric_all is ["NO", "IO_bf", "IO", "NO_bf"]
    ax.plot(xval, dmetric_all[0], label='Asimov, true NO', zorder=3, lw=3, color='red')
    ax.plot(xval, dmetric_all[2], label='Asimov, true IO', zorder=3, lw=3, color='blue')
    ax.plot(xval, dmetric_all[1], label='Asimov, best fit IO', zorder=3, lw=3, color='green')
    ax.plot(xval, dmetric_all[3], label='Asimov, best fit NO', zorder=3, lw=3, color='orange')

    # Mean dmetric values with error bars from the statistically fluctuated distributions for comparison against the Asimov dmetric values
    if (means_list != None):
        pt_style = ['red', 'green', 'blue', 'orange']
        num_theta = int(len(meansx_list)/4)

        for g in range(num_theta):
            n = g * 4
            for i in range(4):
                if (i+n==0):
                    ax.errorbar([meansx_list[i+n]], [means_list[i+n]], yerr=[err_list[i+n]], color=pt_style[i], marker='*', markersize=8, linestyle='None', label='PT method mean')
                else:
                    ax.errorbar([meansx_list[i+n]], [means_list[i+n]], yerr=[err_list[i+n]], color=pt_style[i], marker='*', markersize=8, label='_')

    if (metric=='llh'):
        ax.set_ylabel(r'$\Delta LLH_{NO-IO}$', fontsize=30)
    elif (metric=='conv_llh'):
        ax.set_ylabel(r'Convoluted $\Delta LLH_{NO-IO}$', fontsize=30)
    elif (metric=='chi2'):
        ax.set_ylabel(r'$\Delta\chi^{2}_{NO-IO}$', fontsize=30)
    elif (metric=='mod_chi2'):
        ax.set_ylabel(r'Modified $\Delta\chi^{2}_{NO-IO}$', fontsize=30)
    
    ax.set_xlabel(r'$\sin^2 \theta_{23}$', fontsize=30)
    ax.legend(loc='best', fontsize='large', ncol=1)

def make_dmetric_octant_plot(xval, dp_no, dc_no, dw_no, dp_io, dc_io, dw_io, metric, type_label=None, means_list=None, meansx_list=None):

    fig, ax = plt.subplots(figsize=(12,10))
    plt.grid()

    # Dmetric plots for the minimizer's choice of minimum fit value, the fit value 
    # in the same octant as the truth, and the fit value in the opposite octant as 
    # the truth for both true normal ordering and true inverted ordering.
    ax.plot(xval, dp_no, label='true NO, fit choice', zorder=3, lw=3, color='red')
    ax.plot(xval, dc_no, label='true NO, same octant', zorder=3, lw=3, color='red', linestyle='dashed')
    ax.plot(xval, dw_no, label='true NO, opposite octant', zorder=3, lw=3, color='red', linestyle='dotted')
    ax.plot(xval, dp_io, label='true IO, fit choice', zorder=3, lw=3, color='blue')
    ax.plot(xval, dc_io, label='true IO, same octant', zorder=3, lw=3, color='blue', linestyle='dashed')
    ax.plot(xval, dw_io, label='true IO, opposite octant', zorder=3, lw=3, color='blue', linestyle='dotted')

    # Mean dmetric values from the statistically fluctuated distributions for comparison against the Asimov dmetric values
    if (means_list != None):
        
        pt_style = ['red', 'green', 'blue', 'orange']
        num_theta = int(len(meansx_list)/4)
        mean_type = { "NO" : 0, "IO_bf" : 1, "IO" : 2, "NO_bf" : 3}
        t = mean_type[type_label]
        
        for g in range(num_theta):
            n = g * 4
            if (t+n < 4):
               ax.plot([meansx_list[t+n]], [means_list[t+n]], color=pt_style[t], marker='*', label='PT method mean')
            else:
               ax.plot([meansx_list[t+n]], [means_list[t+n]], color=pt_style[t], marker='*', label='_')

    if (metric=='llh'):
        ax.set_ylabel(r'$\Delta LLH_{NO-IO}$', fontsize=30)
    elif (metric=='conv_llh'):
        ax.set_ylabel(r'Convoluted $\Delta LLH_{NO-IO}$', fontsize=30)
    elif (metric=='chi2'):
        ax.set_ylabel(r'$\Delta\chi^{2}_{NO-IO}$', fontsize=30)
    elif (metric=='mod_chi2'):
        ax.set_ylabel(r'Modified $\Delta\chi^{2}_{NO-IO}$', fontsize=30)
    
    ax.set_xlabel(r'$\sin^2 \theta_{23}$', fontsize=30)
    plt.title("9.3 Year NMO Analysis: Octant Dependency", fontsize=25)

    ax.legend(loc='best',  fontsize='large', ncol=1)

def plot_metric_asimov(xval, metric_NO, metric_IO, truth, label1, label2, metric):

    # Plot metric for normal and inverted ordering
    fig, ax = plt.subplots(figsize=(10,10))
    plt.grid()

    ax.plot(xval, metric_NO, label=label1, zorder=3, lw=3, color='red')
    ax.plot(xval, metric_IO, label=label2, zorder=3, lw=3, color='blue')

    if (metric=='llh'):
        ax.set_ylabel(r'LLH', fontsize='x-large')
    elif (metric=='conv_llh'):
        ax.set_ylabel(r'Convoluted LLH', fontsize='x-large')
    elif (metric=='chi2'):
        ax.set_ylabel(r'$\chi^2$', fontsize='x-large')
    elif (metric=='mod_chi2'):
        ax.set_ylabel(r'Modified $\chi^2$', fontsize='x-large')
    
    ax.set_xlabel(r'$\sin^2 \theta_{23}$', fontsize='x-large')
    plt.title("Asimov, Truth: " + truth, fontsize=25)

    ax.legend(loc='best', fontsize='large', ncol=1)

def octant_curves(oct_dict, val_bf, val_alt):

    # Store correct and wrong octant fit values
    c = 'bf_correct_oct'
    w = 'bf_wrong_oct'
    c_oct = []
    w_oct = []

    if (type(oct_dict)==list):
        for n, octant in enumerate(oct_dict):
            if octant in c:
                c_oct.append(val_bf[n])
                w_oct.append(val_alt[n])
            elif octant in w:
                c_oct.append(val_alt[n])
                w_oct.append(val_bf[n])
    
    return c_oct, w_oct

# Main

parser = argparse.ArgumentParser(description='Plot sensitivity and fluctuated pseudotrial distributions.')
parser.add_argument('-m', dest='metric', type=str, help='Metric used for fits')
parser.add_argument('-f', dest='asimov_file', type=str, help='Metric file with main values to plot')
parser.add_argument('-f2', dest='asimov_file2', type=str, required=False, default=None, help='Second metric file for purposes of testing/comparing sensitivities')
parser.add_argument('-af', dest='alternate_file', type=str, required=False, default=None, help='File with the alternate fit metric values')
parser.add_argument('-of', dest='octant_info_file', type=str, required=False, default=None, help='File with octant labels for all fits')
parser.add_argument('-g', dest='greco', action='store_true', required=False, help='Include GRECO sensitivities in plot')  
parser.add_argument('-s', dest='simple_sens', action='store_true', required=False, help='Plot simple sensitivities using only two out of the four fits')  

args = parser.parse_args()

theta23 = np.round(np.linspace(39.0, 51.0, 20),2)
theta23list = theta23.tolist()

assert args.metric is not None, ('Need to provide the metric that is being used!')
assert args.asimov_file is not None, ('Need to provide an Asimov metric file!')

dfit_names = ["NO", "IO_bf", "IO", "NO_bf"]
bf = ['', '_bf', '', '_bf', '_bf', '', '_bf', '']

# Assemble lists of all calculations with asimov values
xval, metric_all, dmetric_all, sens_NO, sens_IO, simple_sens_NO, simple_sens_IO = asimov_vals(theta23list, args.asimov_file, args.metric)

# Plot metric values for each fit in the analysis
for i, name in enumerate(dfit_names):
    plot_metric_asimov(xval, metric_all[2*i], metric_all[2*i+1], name, 'NO'+bf[2*i], 'IO'+bf[2*i+1], args.metric)
   
# Plot dmetric values for each dfit in the analysis
make_dmetric_asimov_plot(xval, dmetric_all, args.metric)

# Plot minimizer octant fit choices and final choice 
if (args.alternate_file != None and args.octant_info_file != None):
   octant_dependency(args.alternate_file, args.octant_info_file, theta23list, dmetric_all, args.metric) 

# Plot sensitivity
make_sensitivity_plot(xval, sens_NO, sens_IO, simple_sens_NO, simple_sens_IO, simple_sens=args.simple_sens, greco=args.greco)

# Compare main sensitivity to a second sensitivity for testing purposes
if (args.asimov_file2 != None):
    make_sensitivity_plot(xval, sens_NO, sens_IO, simple_sens_NO, simple_sens_IO, simple_sens=args.simple_sens, greco=args.greco, two_sensitivities=True, theta23list=theta23list, asimov_file2=args.asimov_file2, metric=args.metric)

dump_figures_to_pdf("asimov_plots_file.pdf")
