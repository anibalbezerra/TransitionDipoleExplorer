import re
import os
import math
import numpy as np
import pandas as pd
import time
from coloredLogger import color_log 

class projWFC_parser:
    def __init__(self, filename, log_level) -> None:
        self.logger = color_log(log_level).logger # defining logger to print code info during running 
        self.filename = filename

        if filename:
            self.projections = self.parse_atomic_projections()
            self.atomic_orbitals = self.print_proj_info(self.projections)

            duplicatedAtomicOrbitalsDict = self.find_duplicate_indices() # dictionary of the duplicated atomic states columns' indexes
            self.logger.warning('INFO :: Indexes of duplicated Orbitals  - grouped by kind(desconsidering different quantum numbers):')
            print(duplicatedAtomicOrbitalsDict)


    def timming(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()  # Start the timer
            result = func(*args, **kwargs)    # Execute the function
            end_time = time.perf_counter()    # End the timer
            execution_time = end_time - start_time
            print(f"  Function '{func.__name__}' executed in {execution_time:.4f} seconds. \n\n")
            return result
        return wrapper 

    def parse_atomic_projections(self):
        self.logger.info('Parsing Atomic Orbitals projection information')
        orbital_map = {
            0: ['S'],
            1: ['Px', 'Py', 'Pz'],
            2: ['Dxy', 'Dyz', 'Dz2', 'Dxz', 'Dx2-y2'],
            3: ['F_x(5x2-3r2)', 'F_y(5y2-3r2)', 'F_z(5z2-3r2)', 'F_xyz', 'F_z(x2-y2)', 'F_xz2-yz2', 'F_x2y']
        }
        
        projections = {}
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        inside_block = False
        for line in lines:
            if 'Atomic states used for projection' in line:
                inside_block = True

            if inside_block and 'state #' in line:
                match = re.search(r'state #\s*(\d+): atom\s+(\d+)\s+\((.*?)\), wfc\s+(\d+)\s+\(l=(\d+)\s+m=\s*(\d+)\)', line)
                if match:
                    state_number = int(match.group(1))
                    atom_number = int(match.group(2))
                    atom_type = match.group(3).strip()
                    wfc_number = int(match.group(4))
                    l = int(match.group(5))
                    m = int(match.group(6))

                    # Determine the orbital name based on l and m
                    if l in orbital_map and (m-1) < len(orbital_map[l]):
                        orbital = orbital_map[l][m-1]
                    else:
                        orbital = f"unknown(l={l}, m={m})"

                    # Store the result
                    projections[state_number] = {
                        'atom_number': atom_number,
                        'atom_type': atom_type,
                        'wfc_number': wfc_number,
                        'l': l,
                        'm': m,
                        'orbital': orbital
                    }

        return projections 

    def print_proj_info(self, projections):

        atomic_orbitals = []
        self.logger.info('Consolidated atomic orbitals gathered')
        for state, info in projections.items():
            print(f"     State #{state}: Atom {info['atom_number']} ({info['atom_type']}), "
                f"WFC {info['wfc_number']} = {info['orbital']}")
            #atomic_orbitals.append(f"{info['atom_type']}{info['atom_number']}{info['orbital']}")
            atomic_orbitals.append(f"{info['atom_type']}{info['orbital']}")
        
        self.logger.info('List of found atomic_orbitals:')
        print(atomic_orbitals, '\n')
        return atomic_orbitals

    def numerology_sanity_test(self, nbnd, nkstot, e_lineNumber, k_lineNumber, psi2_000_lineNumber, psi2_lineNumber, psi_lineNumber):
        if k_lineNumber == nkstot:
            self.logger.warning(f'Number of k points properly parsed') 
            if e_lineNumber == nkstot*nbnd:
                self.logger.warning(f'Number of energy projections properly parsed')
                if psi2_000_lineNumber == psi2_lineNumber - psi_lineNumber:
                    self.logger.warning(f'Number of psi = 0.00 projections properly parsed')
                    passed = True
        else:
            passed = False
        return passed
    

    def parse_projwfc_output(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        psi2_lineNumber = []
        psi2_000_lineNumber = []
        k_lineNumber = []
        e_lineNumber = []
        psi_lineNumber = []
        for ln, line in enumerate(lines, start=1):
            if line.strip().startswith('|psi|^2 ='):
                psi2_lineNumber.append(ln)
                psi_squared = float(line.split('=')[1].strip())
                if psi_squared == 0.0:
                    psi2_000_lineNumber.append(ln)            
            elif line.startswith(' k ='):
                k_lineNumber.append(ln)
            elif '==== e(' in line:
                e_lineNumber.append(ln)
            elif line.strip().startswith('psi ='): 
                psi_lineNumber.append(ln)
            elif line.strip().startswith('natomwfc'):
                natomwfc = int(line.split('=')[-1].strip())
                self.logger.info(f'natomwfc = {natomwfc}')
            elif line.strip().startswith('nbnd'):
                nbnd = int(line.split('=')[-1].strip())
                self.logger.info(f'nbnd = {nbnd}')
            elif line.strip().startswith('nkstot'):
                nkstot = int(line.split('=')[-1].strip())
                self.logger.info(f'nkstot = {nkstot}')


        self.logger.info(f'line numbers:\n \tk {len(k_lineNumber)} \n \te {len(e_lineNumber)}\n \tpsi2 {len(psi2_lineNumber)} \n \tpsi {len(psi_lineNumber)} \n \tpsi2_00 {len(psi2_000_lineNumber)}') 

        psi_blocks = [''] * len(e_lineNumber)

        passed = self.numerology_sanity_test(nbnd, nkstot, len(e_lineNumber), len(k_lineNumber), len(psi2_000_lineNumber), len(psi2_lineNumber), len(psi_lineNumber))
        if passed: self.logger.info('Numerology sanity test passed - parsing is working properly!')
        
        data = []
        projection = np.zeros((len(e_lineNumber), natomwfc)) #get the number of bands automatically
        ks_state_number = np.zeros(len(e_lineNumber), dtype=int)
        ks_energy = np.zeros(len(e_lineNumber), dtype=float)
        kpoint = np.zeros((len(e_lineNumber),4), dtype=float)
        psi_squared = np.zeros(len(e_lineNumber), dtype=float)

        ik = 0
        for num in range(len(e_lineNumber)):
            start_block = e_lineNumber[num]
            end_block = psi2_lineNumber[num]
            
            string = ''
            for i in range(start_block, end_block + 1):
                if lines[i].strip().startswith('|psi|^2 ='):
                    break
                string += lines[i].strip().split('=')[-1]

            psi_blocks[num] = string
            if psi_blocks[num]:
                psi_components = psi_blocks[num].split('+')
                for component in psi_components:
                    weight, index = component.split('*[#')
                    weight = float(weight.strip())
                    state_index = int(index.strip(']')) - 1  # Convert to 0-based index
                    projection[num, state_index] = weight
        
            
            ks_state_number[num] = int(lines[e_lineNumber[num]-1].split('(')[1].split(')')[0].strip())
            ks_energy[num] = float(lines[e_lineNumber[num]-1].split(' ')[-4])

            if num % nbnd == 0.0:
                #self.logger.critical(f'ik = {ik}, num = {num}, nbnd = {nbnd}, line = {lines[k_lineNumber[ik]-1]}')
                kpoint_values_str = lines[k_lineNumber[ik]-1].split('=')[1].strip().split()
                kpoint_values = list(map(float,kpoint_values_str ))
                
                current_kpoint_magnitude = math.sqrt(sum(k**2 for k in kpoint_values))
                ik += 1

            kpoint[num,:-1] = np.array(kpoint_values) #kpoint (kx,ky,kz)
            kpoint[num,-1] = current_kpoint_magnitude #kpoint magnitude abs(k)

            psi_squared[num] = float(lines[psi2_lineNumber[num]-1].split('=')[1].strip())

            data.append((ks_state_number[num], ks_energy[num], tuple(k for k in kpoint[num,:-1]), kpoint[num,-1], \
                         list(p for p in projection[num,:]), psi_squared[num]))       

        return data
    

    def save_ks_states_individual(self, proj_data):
        self.logger.info('Creating directory to save projection files ')
        KS_proj_file_dir = "./projectionFiles"

        # Create the directory if it doesn't exist
        if not os.path.exists(KS_proj_file_dir):
            os.makedirs(KS_proj_file_dir)
            self.logger.warning(f"     Directory '{KS_proj_file_dir}' created.")
        else:
            self.logger.warning(f"     Directory '{KS_proj_file_dir}' already exists.")

        ks_states = {}
        all_dataframes = []

        for state_info in proj_data:
            ks_state_number, ks_energy, kpoint, kpoint_magnitude, projections, psi_squared = state_info
            if ks_state_number not in ks_states:
                ks_states[ks_state_number] = []
            ks_states[ks_state_number].append((ks_energy, kpoint, kpoint_magnitude, projections, psi_squared))

        projection_headers = self.create_atomic_orbital_header().strip().split()
        self.logger.info('Creating projection_headers')
        print(projection_headers)

        for ks_state_number, state_data in ks_states.items():
            output_file = f"./projectionFiles/KS{ks_state_number}_projections.dat"
            output_file_summed = f"./projectionFiles/KS{ks_state_number}_summed-projections.dat"
            
            with open(output_file, 'w') as f, open(output_file_summed, 'w') as f2:
                f.write(f"*** Projections from Khom-Sham state # {ks_state_number}\n")
                f.write(self.create_atomic_orbital_header())

                data = []
                for ind, (ks_energy, kpoint, kpoint_magnitude, projections, psi_squared) in enumerate(state_data):
                    projections_str = "\t".join(f"{proj:.3f}" for proj in projections)
                    projections_list = [float(proj) for proj in projections]

                    data.append([ind+1, ks_energy, kpoint[0], kpoint[1], kpoint[2], kpoint_magnitude] + projections_list + [psi_squared])

                    f.write(f"{ind+1} \t {ks_energy:.6f} \t {kpoint[0]:.6f} \t {kpoint[1]:.6f} \t {kpoint[2]:.6f} \t "
                            f"{kpoint_magnitude:.3f} \t {projections_str}  \t {psi_squared:.3f}\n")

                # Create DataFrame for the current state
                df = pd.DataFrame(data, columns=projection_headers)
                df = self.sum_by_orbital(df) #sum columns with the same atomic orbital projection (regardless of the principal quantum number)
                f2.write(df.to_string(index=False, header=True))

                all_dataframes.append(df)

        #saving dataframes to a single hdf5 file
        file_name = KS_proj_file_dir + "/structured_projection_dataframes.h5"

        with pd.HDFStore(file_name, mode='w') as store:
            # Save each DataFrame with a unique key
            for i, df in enumerate(all_dataframes):
                store.put(f'df_KS{i}', df, format='table')
            
            # Store the number of DataFrames as metadata
            store.get_storer(f'df_KS0').attrs.nbnd = len(all_dataframes)  # Store `nbnd` as an attribute
        

        return ks_states, all_dataframes
    
    def _read_structured_projection_dataframe(self, filename):
        df_list = []

        with pd.HDFStore(filename, mode='r') as store:
            # Retrieve the stored `nbnd` from the metadata of the first dataframe
            nbnd = store.get_storer('df_KS0').attrs.nbnd
            self.logger.info(f'Reading hdf5 file with structured projection summed in orbitals dataframes returned nbnd = {nbnd}')
            
            # Load the DataFrames back into a list
            for i in range(nbnd):
                df_list.append(store[f'df_KS{i}'])

                #self.logger.critical(f"df shape {df_list[i].shape} \n {df_list[i].tail(2)}")
        return df_list

    
    def sum_by_orbital(self, df):
        grouped_df = df.T.groupby(level=0).sum()
        df = grouped_df.T

        desired_order = ["ik", "e(ev)", "kx", "ky", "kz", "abs(k)"] + \
            [col for col in df.columns if col not in ["ik", "e(ev)", "kx", "ky", "kz", "abs(k)", "|psi|^2"]] + ["|psi|^2"]

        df = df[desired_order]
        return df
    
    def create_atomic_orbital_header(self):
        header = "ik \t e(ev) \t kx \t ky \t kz \t abs(k)"
        # Adding the orbitals dynamically
        orbitals_header = "\t" + "\t".join(self.atomic_orbitals)
        # Final header string
        header += orbitals_header + "\t|psi|^2\n"
        return header
    
    def find_duplicate_indices(self):
        """Finds the indices of duplicate names in a list of atomic states names. """
        name_to_indices = {}
        for i, name in enumerate(self.atomic_orbitals):
            if name in name_to_indices:
                name_to_indices[name].append(i)
            else:
                name_to_indices[name] = [i]

        return {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}

    @timming
    def parse(self):
        projections_data = parser.parse_projwfc_output()
        # returns a dictionary with split information for each KS state - keys are the nbnd KS states
        ks_states, listDataframe = parser.save_ks_states_individual(projections_data) 
        self.logger.info('INSPECTING :: Sample of data saved to file:')
        print(listDataframe[0])
        self.logger.info('Parsing finished with sucess, files saved to directory \n\n.')
        return True
    
if __name__=='__main__':

    path = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/projWfc_files/'
    prefix = 'al'
    filename = path+f'{prefix}.projwfc.out'

    parser = projWFC_parser(filename=filename, log_level='trace')
    parsing = parser.parse()

    parser._read_structured_projection_dataframe('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/projectionFiles/structured_projection_dataframes.h5')
