
# Neutrino Mass Ordering using IceCube DeepCore

Statistical analysis for the Neutrino Mass Ordering (NMO) using IceCube DeepCore. The goal of this analysis is to measure whether the mass ordering is normal or inverted using neutrino oscillations from atmospheric neutrinos. An approximation method, referred to as the Asimov method, is used here to determine the NMO median sensitivity of the more rigorous Pseudotrial method, which follows a frequentist approach. The Asimov method proves to be a close approximation with the added benefit of being significantly less computationally expensive. In addition to the Asimov method, one can also perform the Pseudotrial method, which incorporates possible statistical fluctuations in the real data due to the limited livetime of data collection.  

## Running the scripts


This is the main script in the analysis. It performs each fit required to obtain a sensitivity to distinguish between the normal ordering hypothesis and the inverted ordering hypothesis using the Asimov method (without adding statistical fluctuations) or the Pseudotrial method (with statistical fluctuations). It outputs the fit values in separate JSON files to be gathered and organized in the second scripts.

*OPTIONAL: Add `-ns` to fit the oscillation physics parameters only and fix all systematic parameters to their truth value.*

```
python nmo_fits.py -th <theta23 truth> -o <mass ordering (nh or ih)> -m <metric> -f <type of fit> -no <NO pipeline path> -io <IO pipeline path> -mp <muon pipeline path> -bfd <opposite ordering best fit file directory path> -z <option to add an index for statistical fluctuations>
```


This script takes all the fit value files for all theta23 values and organizes them into a single JSON file for plotting and for keeping better track of all the results.

For Asimov method:

```
python merge_json_files_asimov.py -i <input JSON files> -fd <output directory path>
```

For Pseudotrial method:

```
python merge_json_files_pt.py -i <input JSON files> -fd <output directory path>
```


This script takes in any Asimov output files from the previous script and produces several relevant Asimov plots (for example, sensitivity plots, metric value plots, octant fit plots, etc.)

OPTIONAL: *(`-af` and `-of` go together; if one is provided, the other must be provided too)*
* `-f2` : Provide a second metric file for comparison/testing purposes
* `-af` : Provide the alternate metric file, a.k.a. the minimized metric values for the theta23 octant fit that the minimizer did not accept as the global minimum
* `-of` : Provide the octant file, a.k.a. the file that stores whether the best fit theta23 value was in the same octant as the truth or the opposite octant
* `-g` : If called, plots GRECO published sensitivities for comparison
* `-s` : Calculate the simple sensitivities instead, which use only two of the four fits
* `-q` : Plot simple sensitivities using only one out of the four fits; if using this option *must* also provide the simple sensitivity `-s` flag

```
python asimov_plots.py -m <metric> -f <input metric JSON file>
```

TODO: Finish cleaning up Pseudotrial plotting script and include.
