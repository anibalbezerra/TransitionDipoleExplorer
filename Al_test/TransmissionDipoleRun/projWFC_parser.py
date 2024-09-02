import re
import os
import math
import numpy as np
import pandas as pd
import time


class projWFC_parser:

    def __init__(self, filename) -> None:
        self.filename = filename

        self.projections = self.parse_atomic_projections()
        self.atomic_orbitals = self.print_proj_info(self.projections)

        duplicatedAtomicOrbitalsDict = self.find_duplicate_indices() # dictionary of the duplicated atomic states columns' indexes
        print('INFO :: Indexes of duplicated Orbitals  - grouped by kind(desconsidering different quantum numbers) \n', duplicatedAtomicOrbitalsDict)

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
        print('\n INFO :: Parsing Atomic Orbitals projection information \n')
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
                print('\n PROCESSING INFO :: Atomic states used for projection \n')

            if inside_block and 'state #' in line:
                print(line)
                match = re.search(r'state #\s+(\d+): atom\s+(\d+)\s+\((.*?)\), wfc\s+(\d+)\s+\(l=(\d+)\s+m=\s*(\d+)\)', line)
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
        print('\n PROCESSING INFO :: Consolidating atomic orbitals gathering')
        for state, info in projections.items():
            print(f"     State #{state}: Atom {info['atom_number']} ({info['atom_type']}), "
                f"WFC {info['wfc_number']} = {info['orbital']}")
            #atomic_orbitals.append(f"{info['atom_type']}{info['atom_number']}{info['orbital']}")
            atomic_orbitals.append(f"{info['atom_type']}{info['orbital']}")
        
        print('\n PROCESSING INFO :: Listing found atomic_orbitals \n', atomic_orbitals, '\n')
        return atomic_orbitals

    def parse_projwfc_output(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        data = []
        all_projections=[]
        current_kpoint = None
        current_kpoint_magnitude = None
        ks_state_number = None
                
        recording = False
       
        for _, line in enumerate(lines):            
            # Match k-point line
            if line.startswith(' k ='):
                kpoint_values = line.split('=')[1].strip().split()
                current_kpoint = tuple(map(float, kpoint_values))
                current_kpoint_magnitude = math.sqrt(sum(k**2 for k in current_kpoint))
            # Match KS state energy line
            elif '==== e(' in line:
                ks_state_number = int(line.split('(')[1].split(')')[0].strip())
                ks_energy = float(line.split(' ')[-4])            
            # Match psi projection line
            elif line.strip().startswith('psi ='):
                current_block=[]
                psi_blocks = []
                projections = [0.0]*len(self.atomic_orbitals)
                recording = True
                current_block.append(line.strip())
            # keeps track of the projection block
            elif recording:    
                # Match |psi|^2 line
                if line.strip().startswith('|psi|^2 =') :
                    recording = False
                    psi_blocks.append("".join(current_block))
                    
                    psi_components = psi_blocks[0].split('=')[1].strip().split('+')

                    for component in psi_components:
                        weight, index = component.split('*[#')
                        weight = float(weight.strip())

                        state_index = int(index.strip(']')) - 1  # Convert to 0-based index
                        projections[state_index] = weight

                    current_block = []
                    psi_squared = float(line.split('=')[1].strip())
                    data.append((ks_state_number, ks_energy, current_kpoint, current_kpoint_magnitude, projections[:], psi_squared))
                    all_projections.append(projections)

                else:
                    current_block.append(line.strip())          
        
        return data
    
       
    def save_ks_states_individual(self, proj_data):
        print('\n SAVING INFO :: creating directory to save projection files ')
        KS_proj_file_dir = "./projectionFiles"

        # Create the directory if it doesn't exist
        if not os.path.exists(KS_proj_file_dir):
            os.makedirs(KS_proj_file_dir)
            print(f"     Directory '{KS_proj_file_dir}' created.")
        else:
            print(f"     Directory '{KS_proj_file_dir}' already exists.")

        ks_states = {}
        all_dataframes = []

        for state_info in proj_data:
            ks_state_number, ks_energy, kpoint, kpoint_magnitude, projections, psi_squared = state_info
            if ks_state_number not in ks_states:
                ks_states[ks_state_number] = []
            ks_states[ks_state_number].append((ks_energy, kpoint, kpoint_magnitude, projections, psi_squared))

        projection_headers = self.create_atomic_orbital_header().strip().split()
        print('\n SAVING INFO :: creating projection_headers \n', projection_headers)

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

        return ks_states, all_dataframes
    
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
        print('\n INSPECTING :: Sample of data saved to file \n',listDataframe[0])
        print('\n\n FINAL INFO :: Parsing finished with sucess, files saved to directory \n\n.')
        return True






if __name__ == '__main__':
    
    path = '/home/anibal/scratch/DFT/TransitionDipoleExplorer/Al_test/'
    prefix = 'Al'
    filename = '/home/anibal/scratch/DFT/TransitionDipoleExplorer/Al_test/al_projwfc.out'

    parser = projWFC_parser(filename=filename)
    parsing = parser.parse()




    

    
    


    


