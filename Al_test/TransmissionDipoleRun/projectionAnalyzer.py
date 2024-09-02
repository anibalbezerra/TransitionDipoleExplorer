import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import re
import os
import coloredlogs, logging
from multiprocessing import Pool, Manager

class analyzer:
    def __init__(self, parsedProjectionFile_dir, dipoleMatrixResults_dir, verbose = 'debug', loggerLevel = 'INFO') -> None:
        self.parsedProjectionFile_dir = parsedProjectionFile_dir
        self.dipoleMatrixResults_dir = dipoleMatrixResults_dir
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level=f'{loggerLevel}', logger=self.logger)


    def read_projections2dataframe(self, filenameRoot='_summed-projections.dat'):

        naming_convention = r'^KS.*_summed-projections.dat$'  # Example regex pattern
        
        # List and filter the files
        matching_files = []
        for filename in os.listdir(parsedProjectionFile_dir):
            if re.match(naming_convention, filename):
                matching_files.append(filename)
        #sorting names acendingly
        matching_files = sorted(matching_files, key=self._natural_sort_key)

        if self.verbose == 'debug':
            print(f'\t Inspecting INFO :: Full list of projections filenames \n {matching_files} \n')

        self.nbnd = len(matching_files) # defines the number of KS states by the amount of files

        self.logger.info(f'Number of Khom-Sham states (bands) = {self.nbnd}')
        self.logger.info(f'READING INFO :: Name of one of the files (to inspect) =  {matching_files[0]} \n')

        dataframes = {}
        for i, filename in enumerate(matching_files):
            istates = i + 1

            data = []
            with open(self.parsedProjectionFile_dir + filename, 'r') as f:
                first_line = next(f).strip().split()

                for line in f:
                    line = line.strip()
                    if line:
                        data.append(list(map(float, line.split()[1:])))
                    
            df = pd.DataFrame(data, columns=first_line[1:])
            dataframes[f'KS{istates}'] = df

        if self.verbose == 'debug':
            #print(f"\t Inspecting INFO :: Example of dataframe read from file \n {dataframes[f'KS{1}']} \n")
            self.logger.debug(f"\t Inspecting INFO :: Example of dataframe read from file \n {dataframes[f'KS{1}']} \n")

        self.logger.info('Dataframes from projection properly generated')

        return dataframes, first_line[1:]

    def read_DipoleMatrix_to_dataframes(self, in_state = 1, direction='x'):
        ''' in_state used only for inspecting
        Runs over all Khom Sham states projections
        '''
        def _natural_sort_key(filename):
            
            match = re.search(rf'DipoleMatrix_32-2575_{direction}_KS(\d+)\.dat', filename)
            if match:
                return int(match.group(1))
            return filename

        naming_convention = rf'^DipoleMatrix_32-2575_{direction}_KS.*.dat$'  # Example regex pattern
        
        # List and filter the files
        matching_files = []
        for filename in os.listdir(self.dipoleMatrixResults_dir):
            if re.match(naming_convention, filename):
                matching_files.append(filename)

        #sorting names acendingly
        matching_files = sorted(matching_files, key=_natural_sort_key)
        if self.verbose == 'debug':
            print(f'Filenames (ordered) of the dipole data read from file\n {matching_files}')

        dataframes = {}
        for i, filename in enumerate(matching_files):
            istates = i + 1

            data = []
            with open(self.dipoleMatrixResults_dir + filename, 'r') as f:
                first_line = next(f).strip().split()

                for line in f:
                    line = line.strip()
                    if line:
                        data.append(list(map(float, line.split()[:])))
                
            df = pd.DataFrame(data, columns=first_line[:])
            dataframes[f'KS{istates}'] = df

        if self.verbose == 'debug':
            print(f"\t Inspecting INFO :: Example of Dipole Matrix dataframe read from file \n {dataframes[f'KS{in_state}']} \n")
            # Save the DataFrame as a CSV file
            dataframes[f'KS{in_state}'].to_csv(f'DipoleMatrix_DATAFRAME_KS{in_state}.csv', index=False)  # Set index=False to exclude the index column
        
        self.logger.info('Dataframes from Dipole Moment properly generated')

        return dataframes

    def extract_energy_columns(self, dataframes):
        energy_data = []
        for _, df in dataframes.items():
            energy_data.append(df['e(ev)'])

        energy_df = pd.concat(energy_data, axis=1)
        energy_df.columns = dataframes.keys()
        return energy_df

    def extract_k_columns(self, dataframes):
        k_data = []
        for _, df in dataframes.items():
            k_data.append(df['abs(k)'])

        k_df = pd.concat(k_data, axis=1)
        k_df.columns = dataframes.keys()
        return k_df
            
    def _natural_sort_key(self, filename):
        match = re.search(r'KS(\d+)_summed-projections\.dat', filename)
        if match:
            return int(match.group(1))
        return filename

    def create_difference_dataframe(self, df, in_state=1):
        k_column = df.iloc[:, 0]
        first_column = df.iloc[:, in_state]
        new_df = pd.DataFrame({'abs(k)': k_column})

        diff_columns = {f'Diff_{in_state}{i}': df.iloc[:, i] - first_column for i in range(1, df.shape[1])}
        diff_df = pd.DataFrame(diff_columns)

        new_df = pd.concat([new_df, diff_df], axis=1)
        return new_df

    def order_dataframe(self, df, column=0):
        df_sorted = df.sort_values(by=df.columns[column])
        order = df_sorted.index
        return df_sorted, order

    def prepare_consolidated_dataframe(self, projection_header, data, e_diff_df, dipole_data , in_state=1):

        df_projs = data[[x for x in projection_header if x not in ["kx", "ky", "kz", "abs(k)"]]] #[['e(ev)','|psi|^2',  's', 'px', 'py', 'pz' ...]]
        df_energy = e_diff_df#[[f'abs(k)',f'Diff_{in_state}2',f'Diff_{in_state}3',f'Diff_{in_state}4',f'Diff_{in_state}5',f'Diff_{in_state}6']]

        self.logger.debug(f'\n Dipole Moment from KS{in_state} \n {dipole_data.head(3)}')
        df_dipole = dipole_data.drop(['nks'], axis=1)  #[['KS2','KS3','KS4','KS5','KS6']]

        merged_df = pd.concat([df_projs, df_energy, df_dipole], axis=1)

        self.orbitals_str = [x for x in df_projs.columns.tolist() if x not in ['e(ev)', '|psi|^2']]

        column_order = ['abs(k)', 'e(ev)', '|psi|^2'] + self.orbitals_str + \
                    [item for sublist in [[f'Diff_{in_state}{i}', f'KS#{i}'] for i in range(1, self.nbnd+1)] for item in sublist]

        self.logger.debug(f'Consolidated dataframe column order \n {column_order}')        
        
        return merged_df[column_order]
    
    def orbitals_weigthed_dataframe(self, df, in_state=1):
        new_columns_dict = {}

        for j in range(1, self.nbnd+1):
            #if j == in_state:
            #    continue

            for _, atm_orb in enumerate(self.orbitals_str):
                new_column_name = f'DipKS_{in_state}_{j}_{atm_orb}'
                new_columns_dict[new_column_name] = df[f'KS#{j}'] * df[atm_orb] * df[f'|psi|^2']

        # Create a new DataFrame with the calculated columns
        new_columns_df = pd.DataFrame(new_columns_dict)
        new_df = pd.concat([df, new_columns_df], axis=1)

        # Drop unnecessary columns
        columns_to_drop = ['e(ev)', '|psi|^2'] + [orb for orb in self.orbitals_str]
        new_df = new_df.drop(columns=columns_to_drop)

        # Rearrange columns
        column_order = ['abs(k)'] + \
                        [item for sublist in [[f'Diff_{in_state}{i}', f'KS#{i}'] +
                                            [f'DipKS_{in_state}_{i}_{orb}' for orb in self.orbitals_str]
                                            for i in range(1, self.nbnd+1)] for item in sublist]
        new_df = new_df[column_order]

        return new_df
    
    def apply_gaussian_smoothing(self, df, energy_column, response_column, sigma, range_start=0, range_end=70):
        def gaussian(x, mu, sigma):
            """Calculates the Gaussian value for a given x, mean, and standard deviation."""
            return np.exp(-(x - mu)**2 / (2 * sigma**2))

        # Create a grid of x values from range_start to range_end
        x_grid = np.linspace(range_start, range_end, num=10000)

        # Apply Gaussian smoothing to each data point
        smoothed_response = np.zeros_like(x_grid)
        for _, row in df.iterrows():
            energy, response = row[energy_column], row[response_column]
            smoothed_response += response * gaussian(x_grid, energy, sigma)

        # Normalize the result
        smoothed_response /= np.where(np.sum(smoothed_response) != 0, np.sum(smoothed_response) * (x_grid[1] - x_grid[0]), 1)

        # Create a new DataFrame
        smoothed_df = pd.DataFrame({energy_column: x_grid, response_column: smoothed_response})
        return smoothed_df

    def calculate_energy_contributions(self, df, energy_column='Diff_12', response_column='DipKS_16px'):
        """Calculates the sum of a response column for each unique energy value."""

        grouped_df = df.groupby(energy_column)[response_column].sum()
        grouped_df = grouped_df.reset_index()

        if self.verbose == 'debug':
            print(f'Grouped by energy dataframe energy_column={energy_column}, \
                    response_column={response_column}\n', grouped_df ,f'\n {grouped_df[response_column].describe()}')
        return grouped_df

    def calculate_and_smooth_data(self, df, energy_column, response_columns, sigma=0.1):
        """Calculates energy contributions and applies Gaussian smoothing for multiple response columns.
        """
        results = []
        non_smooth=[]
        for en_cycle, response_column in enumerate(response_columns):
            energy_contributions = self.calculate_energy_contributions(df, energy_column, response_column)

            if self.verbose =='debug':
                print(f'Energy Contributions in {response_column} :: cycle {en_cycle}\n')#, energy_contributions)
            smoothed_data = self.apply_gaussian_smoothing(energy_contributions, energy_column, response_column, sigma)
            results.append(smoothed_data)
            non_smooth.append(energy_contributions)
        return results, non_smooth
    
    def generate_response_columns(self, orbitals, in_state, final_state):
        response_columns = []
        for orbital in orbitals:
            response_columns.append(f'DipKS_{in_state}_{final_state}_{orbital}')
        return response_columns

    def run_over_KS_states(self, e_df, data, all_dipole_data, \
                           projection_header,  \
                            in_state, final_state, \
                            direction='x', verbosity = 'detailed', path2save_Figs = './figures/'):
        
        if not os.path.exists(path2save_Figs): # Create the directory if it doesn't exist
            os.makedirs(path2save_Figs)
            if self.verbose == 'debug':
                print(f"     Directory '{path2save_Figs}' created.")
        else:
            if self.verbose == 'debug':
                print(f"     Directory '{path2save_Figs}' already exists.")

        ''' ******************************************************************************************* '''
        ''' FROM NOW ON THE ANALYSIS WILL HAVE AS INITIAL STATE KS <in_state> AND DIRECTION <direction> '''
        ''' ******************************************************************************************* '''

        e_diff_df = self.create_difference_dataframe(e_df,in_state=in_state)
        if verbosity == 'detailed' :
            print(f'\n Energy values differences from KS{in_state} \n', e_diff_df)
        
            plt.plot(e_diff_df.iloc[:,final_state])
            plt.savefig(path2save_Figs + f'energy_diff_KS{in_state}{final_state}.png')

        ''' ordering dataframes following the energy difference of state KS<final_state>'''
        ordered_e_diff_df, order = self.order_dataframe(e_diff_df, column=final_state)
        if verbosity == 'detailed':
            print(f'\n ORDERED  Energy values differences from KS{in_state} to KS{final_state} \n', ordered_e_diff_df)
        
            plt.plot(ordered_e_diff_df.iloc[:,final_state],'-*')
            plt.savefig(path2save_Figs + f'ordered_energy_diff_KS{in_state}{final_state}.png')
    
        ''' consolidating data in a whole dataframe'''
        consolidated_df = self.prepare_consolidated_dataframe(projection_header, data[f'KS{in_state}'], \
                                                              e_diff_df, all_dipole_data[f'KS{in_state}'], in_state=in_state)
        if verbosity == 'detailed':
            print(f'\n Consolidated Data from transition starting on KS{in_state} \n', consolidated_df)
            # Save the DataFrame as a CSV file
        if self.verbose == 'debug':
            consolidated_df.to_csv(f'Consolidated_transition_from_KS{in_state}.csv', index=False)  

        ''' weighting dipoles with orbitals projections'''
        DipoloxProj = self.orbitals_weigthed_dataframe(consolidated_df, in_state=in_state)
        orderedDipoloxProj = DipoloxProj.iloc[order]

        if verbosity == 'detailed':
            print(f'\n Energy difference ordered Weigthed Dipole with projections from transition starting on KS{in_state} \n', orderedDipoloxProj)
            if self.verbose == 'debug':
                print(f'Energy difference ordered Weigthed Dipole with projections from transition starting on KS{in_state} - columns \n',orderedDipoloxProj.columns[:100] )


        ''' Calculating energy contributions for the projection over each atomic orbital
        Using response columns and the Dipole Matrix from <in_state> to <final_state> multiplied by the 
        projection probabilities of KS<in_state> over the atomic orbitals'''
        response_columns = self.generate_response_columns(self.orbitals_str, in_state, final_state)

        if final_state == 1:
            self.logger.info(f'Response columns from in_state {in_state} to final_state {final_state} \n {response_columns}')
        self.logger.info(f'Calculation from in_state {in_state} to final_state {final_state}')

        energy_contributions, _ = self.calculate_and_smooth_data(df=orderedDipoloxProj, \
                                                                 energy_column=f'Diff_{in_state}{final_state}', response_columns=response_columns)
        
        if verbosity == 'detailed':
            print('\t Plotting some figures for inspecting')
            self.plotting('dipole_projection', path2save_Figs, (response_columns,orderedDipoloxProj,in_state,final_state))
            self.plotting('energy_summed_dipole_projection', path2save_Figs, (direction, energy_contributions,in_state,final_state))

        return {f'projDipole_{in_state}_{final_state}': energy_contributions, \
                'orbitals': self.orbitals_str, \
                'in_state': in_state, \
                'final_state': final_state}

    def plotting(self, kind, path2save_Figs, data):
        if kind == 'dipole_projection':
            response_columns,orderedDipoloxProj,in_state,final_state = data
            plt.clf()
            for i_atm, atm_orb in enumerate(response_columns):
                plt.scatter(orderedDipoloxProj[f'Diff_{in_state}{final_state}'], orderedDipoloxProj[atm_orb], label=f'Dipole KS{in_state}-{final_state} - proj {atm_orb}')
            #plt.yscale('log')
            plt.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left', borderaxespad=0.0, fontsize=5)
            plt.savefig(path2save_Figs+'ordered_dipole_projected.png')

        elif kind == 'energy_summed_dipole_projection':
            if not os.path.exists(path2save_Figs + 'energy_summed'): # Create the directory if it doesn't exist
                os.makedirs(path2save_Figs + 'energy_summed')
                self.logger.debug(f"Directory '{path2save_Figs + 'energy_summed'}' created.")
            else:
                self.logger.debug(f" Directory '{path2save_Figs + 'energy_summed'}' already exists..")

            direction, energy_contributions,in_state,final_state = data
            plt.clf()
            for i_atm, atm_orb in enumerate(self.orbitals_str):
                plt.plot(energy_contributions[i_atm].iloc[:,0], energy_contributions[i_atm].iloc[:,1], label=f'Dipole KS{in_state}-{final_state} - proj {atm_orb}')
            plt.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left', borderaxespad=0.0, fontsize=5)
            #plt.yscale('log')
            plt.xlabel(f'\Delta E_{in_state}_{final_state} (eV)')
            plt.ylabel('Weigthed Dipole Transition (u.a.)')
            plt.savefig(path2save_Figs+ 'energy_summed/'+f'energySummed_dipole_projected_{direction}_KS{in_state}_to_KS{final_state}.png')

        elif kind == 'consolidated_projections':
            if not os.path.exists(path2save_Figs + 'consolidated_projections'): # Create the directory if it doesn't exist
                os.makedirs(path2save_Figs + 'consolidated_projections')
                self.logger.debug(f"Directory '{path2save_Figs + 'consolidated_projections'}' created.")
            else:
                self.logger.debug(f" Directory '{path2save_Figs + 'consolidated_projections'}' already exists..")

            final_df, in_state = data
            plt.clf()
            plt.figure(figsize=(16, 12))
            for i_atm, atm_orb in enumerate(final_df.columns[1:]):
                self.logger.debug(f"Printing loop {i_atm}")                
                plt.plot(final_df[final_df.columns[0]], final_df[atm_orb], label=f'KS{in_state}-proj{atm_orb}')
            plt.legend(bbox_to_anchor=(0.95, 1.0), loc='upper left', borderaxespad=0.0, fontsize=10)
            plt.xlabel(f'E (eV)', fontsize=18)
            plt.ylabel('Weigthed Dipole Transition (u.a.)', fontsize=18)
            plt.title(f'Initial state {in_state}', fontsize=14)
            plt.tick_params(axis='x', which='both', direction='in', labelsize=14)
            plt.tick_params(axis='y', which='both', direction='in', labelsize=14)
            plt.savefig(path2save_Figs+ 'consolidated_projections/'+f'consolidated_dipole_projected_instate={in_state}.png')
            plt.close()

            for i_atm, atm_orb in enumerate(final_df.columns[1:]):
                plt.clf()
                plt.figure(figsize=(16, 12))
                self.logger.debug(f"Printing loop {i_atm}")                
                plt.plot(final_df[final_df.columns[0]], final_df[atm_orb], label=f'KS{in_state}-proj{atm_orb}')
                plt.legend(loc='upper right', borderaxespad=0.0, fontsize=10)
                plt.xlabel(f'E (eV)', fontsize=18)
                plt.ylabel('Weigthed Dipole Transition (u.a.)', fontsize=18)
                plt.title(f'Initial state {in_state} - projected over {atm_orb}')
                plt.tick_params(axis='x', which='both', direction='in', labelsize=14)
                plt.tick_params(axis='y', which='both', direction='in', labelsize=14)
                plt.savefig(path2save_Figs+ 'consolidated_projections/' + f'consolidated_dipole_projected_instate={in_state}_orb{atm_orb}.png')
                plt.close()
        
        elif 'final_projections':
            if not os.path.exists(path2save_Figs + 'final_projections'): # Create the directory if it doesn't exist
                os.makedirs(path2save_Figs + 'final_projections')
                self.logger.debug(f"Directory '{path2save_Figs + 'final_projections'}' created.")
            else:
                self.logger.debug(f" Directory '{path2save_Figs + 'final_projections'}' already exists..")

            final_df = data
            plt.clf()
            plt.figure(figsize=(16, 12))
            for i_atm, atm_orb in enumerate(final_df.columns[1:]):
                self.logger.debug(f"Printing loop {i_atm}")                
                plt.plot(final_df[final_df.columns[0]], final_df[atm_orb], label=f'{atm_orb}')
            plt.legend(bbox_to_anchor=(0.95, 1.0), loc='upper left', borderaxespad=0.0, fontsize=10)
            plt.xlabel(f'E (eV)', fontsize=18)
            plt.ylabel('Weigthed Dipole Transition (u.a.)', fontsize=18)
            plt.title(f'Summed over all states', fontsize=14)
            plt.tick_params(axis='x', which='both', direction='in', labelsize=14)
            plt.tick_params(axis='y', which='both', direction='in', labelsize=14)
            plt.savefig(path2save_Figs+ 'final_projections/'+f'consolidated_dipole_projected.png')
            plt.close()

            for i_atm, atm_orb in enumerate(final_df.columns[1:]):
                plt.clf()
                plt.figure(figsize=(16, 12))
                self.logger.debug(f"Printing loop {i_atm}")                
                plt.plot(final_df[final_df.columns[0]], final_df[atm_orb], label=f'{atm_orb}')
                plt.legend(loc='upper right', borderaxespad=0.0, fontsize=16)
                plt.xlabel(f'E (eV)', fontsize=18)
                plt.ylabel('Weigthed Dipole Transition (u.a.)', fontsize=18)
                plt.title(f'Projected over {atm_orb}')
                plt.tick_params(axis='x', which='both', direction='in', labelsize=14)
                plt.tick_params(axis='y', which='both', direction='in', labelsize=14)
                plt.savefig(path2save_Figs+ 'final_projections/' + f'consolidated_dipole_projected_orb{atm_orb}.png')
                plt.close()

    def calculate_dipole_projections(self, in_state, all_data, direction, step = 200, verbosity = 'detailed'):
        e_df, data, projection_header, all_dipole_data = all_data
        results = []

        self.logger.critical(f'Run will be from {in_state} up to {self.nbnd} in steps of {step}')
        if in_state <= self.nbnd:
            for final_state in range(in_state+1, self.nbnd+1, step): # running over all Khom Sham states
                if final_state == in_state: continue # skipping intraband contribution - TODO verify how to implement results from intraband

                result = an.run_over_KS_states(e_df, data, all_dipole_data, projection_header = projection_header, \
                                               in_state=in_state, \
                                               final_state=final_state, \
                                               direction=direction, \
                                               verbosity = verbosity)
                
                # Extract relevant data
                proj_dipole = result[f'projDipole_{in_state}_{final_state}']                

                if verbosity == 'debug':
                    print('\n\t inspecting \n\n',proj_dipole)

                orbitals = result['orbitals']

                consolidate_energyDF = self.consolidate_energy_dataframe(proj_dipole)
                results.append(consolidate_energyDF)

            combined_dataframes = self.merge_and_sum_orbital_projections(results)
            print('\n\n\t Combined energies \n', combined_dataframes, '\n\n')

            self.logger.info(f'Combined projections for in_state {in_state} completed with success')
            
        return combined_dataframes

    def consolidate_energy_dataframe(self, list_dataframes):
        consolidated_df = pd.DataFrame()
        for _, df in enumerate(list_dataframes):            
            consolidated_df = pd.concat([consolidated_df, df], axis=1)
        consolidated_df = consolidated_df.loc[:,~consolidated_df.columns.duplicated()].copy()

        return consolidated_df
    
    def merge_and_sum_orbital_projections(self, dataframes, path2save_Figs = './figures/'):
        self.logger.debug(f'TOTAL OF DATAFRAMES in LIST  {len(dataframes)}')

        energy = dataframes[0].iloc[:,0]

        summed_projections = {}
        final_df = pd.DataFrame()
        df1 = dataframes[0]

        for col in df1.columns[1:]:
            # Extract orbital name (assuming the last part of the column name is the orbital)
            orbital = col.split('_')[-1]
            in_state = col.split('_')[1]
            string = col.split('_')[0] + in_state
            self.logger.debug(f"First Df col name {string} orbital {orbital}")
            
            summed_projections[orbital] = df1[col]

            for df2 in dataframes[1:]:
                # Find the corresponding column in the second dataframe
                matching_col = None
                for col2 in df2.columns[1:]:
                    if col2.split('_')[-1] == orbital:
                        matching_col = col2
                        self.logger.debug(f"Second Df matching column {matching_col} with {orbital}")
                        break

                if matching_col:
                    # Sum the corresponding projections
                    summed_projections[orbital] = summed_projections[orbital] + df2[matching_col] 

        # Add the summed projections to the final dataframe
        final_df[f'energy_{in_state}'] = energy
        for orbital, values in summed_projections.items():
            final_df[string+f'_{orbital}'] = values
        
        print('\n\n\tfinal df after summing two dfs\n',final_df)
        self.plotting('consolidated_projections', path2save_Figs, (final_df, in_state))

        return final_df   
    
    def _calculate_and_append(self,in_state, results_list, e_df, data, projection_header, all_dipole_data, step, direction, verbosity):
        self.logger.info(f'Calculating dipole projections varying in_states: in_state = {in_state}')
        projctDF = self.calculate_dipole_projections(in_state=in_state, all_data=(e_df, data, projection_header, all_dipole_data), step=step, direction=direction, verbosity = verbosity)
        results_list.append(projctDF)
        
    def run_over_instates(self, all_data, initial= 1, final = 3, step=200, direction='x', path2save_Figs = './figures/'):

        self.logger.info('Calculating dipole projections varying in_states')
        (data, e_df, projection_header, all_dipole_data) = all_data

        results = {}
        manager = Manager()
        results_list = manager.list()  # Create a shared list
        verbosity = self.verbose

        self.logger.critical(f'It will be considered INITIAL STATES ranging from {initial} up tp {final} :: See if is this correct.')
        '''for in_state in range(initial,final):
            self.logger.info(f'Calculating dipole projections varying in_states: in_state = {in_state}')

            projctDF = self.calculate_dipole_projections(in_state=in_state, data=(e_df, projection_header, all_dipole_data), step=step, direction=direction, verbosity = 'normal')
            results = {f'state_{in_state}': projctDF}
            results_list.append(projctDF)'''
        
        with Pool() as pool:
            # Create arguments for each function call
            args_list = [(i, results_list, e_df, data, projection_header, all_dipole_data, step, direction, verbosity) for i in range(initial, final)]

            # Parallelize the function calls
            pool.starmap(self._calculate_and_append, args_list)

        self.logger.critical(f'results_list length {len(results_list)}')
        energy = results_list[0].iloc[:,0]
        summed_projections = {}
        final_df = pd.DataFrame()
        df1 = results_list[0]

        self.logger.info('\n\nSumming over all in_states')
        for col in df1.columns[1:]:
            # Extract orbital name (assuming the last part of the column name is the orbital)
            orbital = col.split('_')[-1]
            string = col.split('_')[0] 
            self.logger.debug(f"First Df col name {string} orbital {orbital}")
            
            summed_projections[orbital] = df1[col]

            for df2 in results_list[1:]:
                # Find the corresponding column in the second dataframe
                matching_col = None
                for col2 in df2.columns[1:]:
                    if col2.split('_')[-1] == orbital:
                        matching_col = col2
                        self.logger.debug(f"Second Df matching column {matching_col} with {orbital}")
                        break

                if matching_col:
                    # Sum the corresponding projections
                    summed_projections[orbital] = summed_projections[orbital] + df2[matching_col] 

        # Add the summed projections to the final dataframe
        final_df[f'energy_{1}'] = energy
        for orbital, values in summed_projections.items():
            final_df[f'{orbital}'] = values

        self.logger.info(f'final df after summing all dfs\n {final_df}')

        self.plotting('final_projections', path2save_Figs, (final_df))

        self.logger.info(f'Saving consolidated Dipole Moment for direction {direction} to file')
        final_df.to_csv(f'Dipole_moment_dir_{direction}.csv', index=False) # saving Consolidated dipole momento for direction to file

        return results
    
    def run_all(self, initial_state, final_state, step, direction):
        verbosity = self.verbose

        '''Reading data from files - storing as dataframes
        *reading all projections file - Ordered in k indexes (1 to NK) (not the absolut value)
        *dataframes will have "e(ev) kx ky kz k-point magnitude <orbitals> |psi|^2"
        *Results in a dictionary containing all dataframes
        '''    
        data, projection_header = self.read_projections2dataframe()

        ''' creating a new dataframe with only the energies - Shaped (8000,natm)
            dataframe is ordered in k indexes (1 to nk) (not the absolut value)
        '''    
        e_df = self.extract_energy_columns(data)
        if verbosity == 'detailed':
            print('\n\t Energy values \n', e_df.head(10))

        ''' creating a new dataframe with only the k values - Ordered in k indexes (1 to 8000)  
            should have <total_KS> repeated columns 
        '''
        k_df = self.extract_k_columns(data)
        if verbosity == 'detailed':
            print('\n k values \n', k_df)

        ''' adding the k values to the energy dataframe as the first column  
        '''
        e_df.insert(0, 'abs(k)', k_df.iloc[:,0])
        if verbosity == 'detailed':
            print('\n Energy values with k values inserted \n', e_df)

        ''' reading dipole matrix for state KS1 and direction x'''
        all_dipole_data = self.read_DipoleMatrix_to_dataframes(in_state=50, direction=direction) # reads all dipoles from files generating dictionary of dataframes 

        if final_state == None:
            final_state = self.nbnd # Defining Final state as all states

        results = self.run_over_instates(all_data= (data, e_df, projection_header, all_dipole_data), initial= initial_state, final = final_state, step=step , direction=direction)


        return results


if __name__ == '__main__':

    verbosity = 'detailed' #options 'detailed' and 'debug'
    loggerLevel = 'INFO' #option 'INFO' and 'DEBUG'
    direction = ['x','y','z']
    initial_state = 1
    final_state = None
    step = 1

    parsedProjectionFile_dir = '/home/anibal/scratch/DFT/TransitionDipoleExplorer/Al_test/TransmissionDipoleRun/projectionFiles/'
    dipoleMatrixResults_dir = '/home/anibal/scratch/DFT/TransitionDipoleExplorer/Al_test/DipoleMatrix/'
    
    an = analyzer(parsedProjectionFile_dir=parsedProjectionFile_dir, dipoleMatrixResults_dir = dipoleMatrixResults_dir, verbose=verbosity, loggerLevel = 'INFO')

    for dir in direction:        
        result = an.run_all(initial_state, final_state, step, direction=dir)



  