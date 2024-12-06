# TransitionDipoleExplorer: A Tool for Analyzing Quantum Espresso's Optical Data

### Motivation: 

Quantum Espresso is a powerful software package for electronic structure calculations. It provides insights into the properties of materials. However, analyzing the optical data generated by Quantum Espresso, in special Dielectric Functions from `epsilon.x` package, can be complex and time-consuming. **TransitionDipoleExplorer** aims to streamline this process by offering a Python tool for visualizing and interpreting optical data via transition dipole information projected over atomic orbitals.

### Key Features:

#### Data Import and Visualization:
1. Imports transition dipole data from Quantum Espresso simulations (Quantum Espresso's `epsilon.x` code should be modified - modified version is supplied with this package).
2. Parses and Visualize data in formats including plots, and Dataframes.
3. Analyses the distribution of transition dipole moments and their dependence on energy, and atomic orbitals.
   
#### Orbital Analysis:
1. Analyze the contributions of individual atomic orbitals to the transition dipole moments.
2. Allows for identification of the dominant orbitals involved in transitions.
3. Allows for visualization the spatial distribution over orbitals to gain a deeper understanding of their role in electronic transitions.
   

### Benefits:

**Enhanced Understanding**: Gain deeper insights into the electronic properties of materials by analyzing transition dipole data projected over the atomic orbitals.
**Efficient Analysis**: Streamline the process of data visualization and interpretation, saving time and effort.
**Data-Driven Decision Making**: Make informed decisions based on quantitative analysis of transition dipole data.

TransitionDipoleExplorer is a valuable tool for researchers working with Quantum Espresso data. It empowers users to extract meaningful information from their simulations and gain a deeper understanding of the electronic properties of materials.

---
### Package Structure

Transition Dipole Explorer relies on two main Python codes:

1. projWFC_parser
2. projectionAnalyzer

In what follows we have the detailided description of the two codes.

---
## projWFC_parser: A Python code to parse PROJWFC output files

This python code parses output files generated by the `projwfc.x`, a program for post proscessing of Quantum Espresso, used in electronic structure calculations. The present code can identify atomic state information and orbital projections from the `projwfc.x`'s output file.

### How to use the code:

1. Save the code as a python file.

2. Specify the filename: The code looks for a file named `${prefix}.projwfc.out` by default (`prefix`should be supplied). You can modify the script to point to a different file by changing the filename variable in the `__main__` block. Here's an example:

```python
prefix = 'SiC'
filename = 'SiC_test.projwfc.out'
```
3. Run the script: Navigate to the directory containing the script and the PROJWFC output file and run:
   
```bash
python projWFC_parser.py
```

### Output:

The code will print the following information to the console:

* Atomic state information: This includes details like state number, atom number, atom type, angular momentum quantum numbers (l and m), and the corresponding orbital name.
* Indexes of duplicate atomic orbitals: The script identifies and groups atomic orbitals with the same name (e.g., 'Px', 'Py', 'Pz') regardless of the principal quantum number.
* Sample of parsed data: The code displays a sample of the data saved to an output file.
* Confirmation message: The script will confirm successful parsing and that the results are saved to a directory.


### Important Information:

The running of Quantum Espresso should follow the standard process:
* Perform structural relaxation
* Do self-consistent-field running (SCF - using converged properties like cutoff energies and k-point grid density).
   - Ensure to be using norm-conserving pseudo-potentials, since the next step is only built over such kind of pseudo-potential.
   - Turn off auto-symmetrization by setting `NOSYM` true.
   - Finally use a uniform k-point grid with all weights equal to each other
      - We wrote a simple Python tool based on ASE library to generate the uniform k-point grid - it is in `Tools` directory [Uniform_Kgrid](https://github.com/anibalbezerra/TransitionDipoleExplorer/blob/master/Tools/UNIFORMKGRID_for_epsilonx.py).
      - You can use tools like [seek-path](https://www.materialscloud.org/work/tools/seekpath) to get the reciprocal vectors and generate the uniformly spaced grid).
* Do non-self-consistent-field running (since optical properties require a dense grid, check for convergence with both energy threshould and k-point grid density).
* Do projected density of states calculation using `projwfc.x` code (use the argument fillpdos to write the projections of the Khom-Sham states over the atomic states). The calculation will generate the output file `${prefix}.projwfc.out` that will be used latter.
* Do the `epsilon.x` calculation using the modified version of the code. It will produce a series of files with the Transition Dipole Moment as a function of the k-points and initial and final Khom-Sham states.


### Functionality:

The code includes several functions:

* `__init__`: This function initializes the class and parses the atomic orbital information from a predefined dictionary.
* `timming`: This is a decorator function that measures the execution time of the wrapped function and prints the time taken.
* `parse_atomic_projections`: This function parses the PROJWFC output file to identify atomic state information and assign orbital names based on the angular momentum quantum numbers.
* `print_proj_info`: This function consolidates the atomic orbital information and prints it to the console.
* `parse_projwfc_output`: This function is the core of the code. It parses the entire PROJWFC output file line by line to extract information about k-points, KS state energies, and orbital projections.
* `save_ks_states_individual`: This function saves the parsed data for each KS state (electronic state) to separate files. The files include k-point information, KS state energy, orbital projections, and the total squared wavefunction value.
* `sum_by_orbital`: This function sums the contribution of each orbital type (e.g., 'Px', 'Py', 'Pz') across all principal quantum numbers, resulting in a single value for each orbital name in the final output file.
* `create_atomic_orbital_header`: This function creates the header row for the output files, including labels for k-point information, orbital projections, and the total squared wavefunction value.
* `find_duplicate_indices`: This function identifies atomic orbitals with the same name (regardless of principal quantum number) and returns a dictionary containing the name and corresponding indices in the atomic_orbitals list.
* `parse`: This function serves as the main entry point. It calls other functions to parse the file, save the results, and print informative messages.

### Dependencies:

The code relies on the following python libraries:

* `re`: Regular expressions library
* `os`: Operating system interaction library
* `math`: Mathematical functions library
* `numpy`: Numerical computing library (optional, used for calculating the magnitude of the k-point vector)
* `pandas`: Data manipulation and analysis library

If you don't have these libraries installed, you can install them using the pip command: 
```bash
pip install re os math numpy pandas
```

---
## projectionAnalyzer

### Functionality
The analyzer class provides various functionalities for analyzing KS state data, including:

* Reading data from parsed projection and dipole moment files.
* Identifying the number of KS states based on the number of projection files.
* Extracting specific data columns (e.g., energy, wavefunctions, k-points).
* Creating difference dataframes based on energy values.
* Ordering dataframes based on specific columns.
* Consolidating data from projections, energy differences, and dipole moments into a single dataframe.
* Calculating weighted dipole moments based on orbital projections.
* Applying Gaussian smoothing to data for noise reduction.
* Calculating energy contributions for response columns representing weighted dipole moments.
* Generating response columns for weighted dipole moments.
* Plotting various figures to visualize the results (optional).

### Usage

The code relies on two group of files:
* The projections files written by the `projWFC_parser` code.
* The Dipole Matrix files written by the modified `epsilon.x` code.

Within the main block, update the path that points to the folder having those files. Update variables `parsedProjectionFile_dir`, and `dipoleMatrixResults_dir`.

The running is done over all Khom-Sham states accounting for each k-point in the reciprocal cell as simulated with `pw.x`. It runs separately for directions `x`, `y`, and `z` setup  by the list `direction`. 

It also uses the variables `initial_state` and `final_state` to determine wich states will be considered for the Transmission Dipole Moment. Set `final_state=None` to run over all states from `initial_state` (change those variables only for debugging).

#### Additional Notes

* The class utilizes the pandas and matplotlib libraries for data manipulation and plotting.
* The coloredlogs library is used for colored logging output (required).
* The multiprocessing library can be used for parallel processing (not explicitly shown in the example).
* The verbosity level can be set to "debug" or "info" to control the amount of information printed during execution.
* The path to save figures can be customized by setting the path2save_Figs argument.

#### Disclaimer

1. As a tool that runs over all Khom Sham states, even its execution being paralellized, the Analyzer could take a long time to be executed. It all depends on the k-points grid density, the number of atoms within the structure, the number of bands evaluated using `pw.x` and the number of atomic orbitals defined by the pseudo-potentials.
2. This is only a tool to get the projections of the optical response over the atomic orbitals. The physic's aspect of the analysis should be done by the researcher.
3. Even `epsilon.x` evaluating both inter and intraband transitions (the last only for metals), the present code was implemented only for the interband transitions. In a next version the intraband contribution should be added.
4.  For more information about the group behind the scenes, plese visit [LDFT](https://www.unifal-mg.edu.br/ldft).

#### HAVE FUN!  


