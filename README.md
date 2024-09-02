# TransitionDipoleExplorer: A Tool for Analyzing Quantum Espresso's Optical Data

## Motivation: 

Quantum Espresso is a powerful software package for electronic structure calculations. It provides insights into the properties of materials. However, analyzing the optical data generated by Quantum Espresso, in special Dielectric Functions from `epsilon.x` package, can be complex and time-consuming. **TransitionDipoleExplorer** aims to streamline this process by offering a Python tool for visualizing and interpreting optical data via transition dipole information projected over atomic orbitals.

## Key Features:

### Data Import and Visualization:
1. Imports transition dipole data from Quantum Espresso simulations (Quantum Espresso's `epsilon.x` code should be modified - modified version is supplied with this package).
2. Parses and Visualize data in formats including plots, and Dataframes.
3. Analyses the distribution of transition dipole moments and their dependence on energy, and atomic orbitals.
   
### Orbital Analysis:
1. Analyze the contributions of individual atomic orbitals to the transition dipole moments.
2. Allows for identification of the dominant orbitals involved in transitions.
3. Allows for visualization the spatial distribution over orbitals to gain a deeper understanding of their role in electronic transitions.
   

## Benefits:

**Enhanced Understanding**: Gain deeper insights into the electronic properties of materials by analyzing transition dipole data projected over the atomic orbitals.
**Efficient Analysis**: Streamline the process of data visualization and interpretation, saving time and effort.
**Data-Driven Decision Making**: Make informed decisions based on quantitative analysis of transition dipole data.

### Target Audience:

Materials scientists
Physicists
Computational chemists
Researchers studying electronic properties of materials

TransitionDipoleExplorer is a valuable tool for researchers working with Quantum Espresso data. It empowers users to extract meaningful information from their simulations and gain a deeper understanding of the electronic properties of materials.

---
## Package Structure

Transition Dipole Explorer relies on two main Python codes:

1. projWGC_parser
2. projectionAnalyzer

In what follows we have the detailided description of the two codes.

---
# projWFC_parser: A Python code to parse PROJWFC output files

This python code parses output files generated by the PROJWFC program, a program used in electronic structure calculations. The code can identify atomic state information and orbital projections from the file.

## How to use the code:

1. Save the code as a python file. For this example, save the code as projWFC_parser.py

2. Specify the filename: The code looks for a file named `${prefix}.projwfc.out` by default. You can modify the script to point to a different file by changing the filename variable in the __main__ block. Here's an example:

```python
prefix = 'SiC'
filename = 'SiC_test.projwfc.out'
```
3. Run the script: Navigate to the directory containing the script and the PROJWFC output file and run:
   
```bash
python projWFC_parser.py
```

## Output:

The code will print the following information to the console:

* Atomic state information: This includes details like state number, atom number, atom type, angular momentum quantum numbers (l and m), and the corresponding orbital name.
* Indexes of duplicate atomic orbitals: The script identifies and groups atomic orbitals with the same name (e.g., 'Px', 'Py', 'Pz') regardless of the principal quantum number.
* Sample of parsed data: The code displays a sample of the data saved to an output file.
* Confirmation message: The script will confirm successful parsing and that the results are saved to a directory.

## Functionality:

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

## Dependencies:

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
