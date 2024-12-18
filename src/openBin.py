"""
Module for opening binary files and processing their content.
"""

import numpy as np
import matplotlib.pyplot as plt
from coloredLogger import color_log as logger


class parse_eps_binaries:
    """
    A class to parse and process binary files containing electronic structure data.
    """
    
    def __init__(self, log_level='info'):
        """
        Initialize the parse_eps_binaries class with a specified log level.

        :param log_level: The log level to set ('info', 'debug', 'trace').
        :type log_level: str
        """
        self.logger = logger(log_level).logger  # Define logger to print code info during running    
        
    def unpack_binary(self, path, **kwargs):        
        """
        Unpack binary data from a file and return the parsed variables.

        :param path: The path to the binary file.
        :type path: str
        :param kwargs: Additional keyword arguments specifying which variables to extract.
        :return: A tuple containing the parsed variables.
        :rtype: tuple
        """
        filename = path.split('/')[-1]
        self.logger.warning(f'After unpacking file {filename}, the following variables will be get:\n {kwargs}')

        wproj_bool = kwargs.get('wproj')
        kproj_bool = kwargs.get('kproj')
        kproj_intra_bool = kwargs.get('kproj_intra')
        etrans_bool = kwargs.get('etrans')

        with open(path, 'rb') as f:
            nw = np.fromfile(f, dtype=np.int32, count=1)[0]
            nbnd = np.fromfile(f, dtype=np.int32, count=1)[0]
            nks = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.logger.info(f'Parsed values (file {filename}):\n\tnw={nw}\n\tnbnd={nbnd}\n\tnks={nks}')
            
            if wproj_bool and not kproj_bool and not kproj_intra_bool and not etrans_bool:            
                wgrid = np.fromfile(f, dtype=np.float64, count=nw)
                wproj = np.fromfile(f, dtype=np.float64).reshape((nw, nbnd, nbnd), order='F')
                self.logger.info(f'\n\twgrid shape ={wgrid.shape}\n\twproj shape={wproj.shape}')
                list2return = (nw, nbnd, wgrid, wproj)
            
            if kproj_bool and not wproj_bool and not kproj_intra_bool and not etrans_bool:            
                wgrid = np.fromfile(f, dtype=np.float64, count=nw)
                kproj = np.fromfile(f, dtype=np.float64).reshape((nks, nbnd, nbnd), order='F')
                self.logger.info(f'\n\twgrid shape ={wgrid.shape}\n\tkproj shape={kproj.shape}')
                list2return = (nw, nbnd, nks, wgrid, kproj)
            
            if kproj_intra_bool and not wproj_bool and not kproj_bool and not etrans_bool:            
                wgrid = np.fromfile(f, dtype=np.float64, count=nw)
                kproj_intra = np.fromfile(f, dtype=np.float64).reshape((nks, nbnd), order='F')
                self.logger.info(f'\n\twgrid shape ={wgrid.shape}\n\tkproj_intra shape={kproj_intra.shape}')
                list2return = (nw, nbnd, nks, wgrid, kproj_intra)
            
            if etrans_bool and not wproj_bool and not kproj_bool and not kproj_intra_bool:
                etrans = np.fromfile(f, dtype=np.float64).reshape((nks, nbnd, nbnd), order='F')
                self.logger.info(f'\n\tetrans shape={etrans.shape}')
                list2return = (nw, nbnd, nks, etrans)
        
        return list2return
    
    def get_eps_from_wproj(self, wproj, kind='real', normalization_const=0.0002):
        """
        Calculate the dielectric function from the wproj data.

        :param wproj: The wproj data.
        :type wproj: numpy.ndarray
        :param kind: The type of dielectric function to calculate ('real' or 'imag').
        :type kind: str
        :param normalization_const: The normalization constant.
        :type normalization_const: float
        :return: The calculated dielectric function.
        :rtype: numpy.ndarray
        """
        if kind == 'real':
            self.logger.info(f'epsr evaluated from summing wproj read from binary file')
            eps = 1.0 + np.sum(wproj, axis=(1, 2)) * normalization_const
        elif kind == 'imag':
            self.logger.info(f'epsi evaluated from summing wproj read from binary file')
            eps = np.sum(wproj, axis=(1, 2)) * normalization_const
        
        return eps
    
    def plot_eps(self, wgrid, eps, filename, kind='real', ylog=False):
        """
        Plot the dielectric function and save the plot to a file.

        :param wgrid: The frequency grid.
        :type wgrid: numpy.ndarray
        :param eps: The dielectric function data.
        :type eps: numpy.ndarray
        :param filename: The name of the file to save the plot.
        :type filename: str
        :param kind: The type of dielectric function ('real' or 'imag').
        :type kind: str
        :param ylog: Whether to use a logarithmic scale for the y-axis.
        :type ylog: bool
        """
        plt.clf()
        plt.plot(wgrid, eps, '--')
        
        if kind == 'real':
            string = 'epsr'
        elif kind == 'imag':
            string = 'epsi'

        plt.xlabel(r'$\hbar \omega$ (eV)')
        plt.ylabel(string + ' (u.a.)')
        
        if ylog:
            plt.yscale('log')
        
        plt.savefig(filename)
        self.logger.info(f'{string} graph plot with name {filename}')


if __name__ == '__main__':
    files_root = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/eps_files/'
    figures_root = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/'

    eps_bin_parser = parse_eps_binaries(log_level='info')

    data_w = eps_bin_parser.unpack_binary(path=files_root + 'proj_epsi_al_x_w.bin', wproj=True, kproj=False, kproj_intra=False, etrans=False)

    data_k = eps_bin_parser.unpack_binary(path=files_root + 'proj_al_x_k.bin', wproj=False, kproj=True, kproj_intra=False, etrans=False)

    data_k_intra = eps_bin_parser.unpack_binary(path=files_root + 'proj_al_x_k-intra.bin', wproj=False, kproj=False, kproj_intra=True, etrans=False)

    data_etrans = eps_bin_parser.unpack_binary(path=files_root + 'proj_al-etrans.bin', wproj=False, kproj=False, kproj_intra=False, etrans=True)

    nw, nbnd, wgrid, wproj = data_w
    epsi = eps_bin_parser.get_eps_from_wproj(wproj, kind='imag', normalization_const=0.0002)

    eps_bin_parser.plot_eps(wgrid, epsi, filename=figures_root + 'test3.png', kind='imag', ylog=True)
