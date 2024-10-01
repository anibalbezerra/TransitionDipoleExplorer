from coloredLogger import color_log
from projParser import projWFC_parser as parseProj
from openBin import parse_eps_binaries as parseEps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

class analyzer:

    def __init__(self, projFilename, log_level = 'info', prefix = 'al') -> None:
        self.log_level = log_level
        self.projFilename = projFilename
        self.prefix = prefix

        self.logger = color_log(log_level).logger # defining logger to print code info during running
        
        params = self.get_params()
        self.natomwfc = params['natomwfc']
        self.nbnd = params['nbnd']
        self.nkstot = params['nkstot']
        self.degauss = params['degauss']         


    def read_projections2dataframe(self):
        self.projData = parseProj(filename=None, log_level=self.log_level)._read_structured_projection_dataframe(self.projFilename)

        dataframes = {}
        for istates, df in enumerate(self.projData):
            dataframes[f'KS{istates+1}'] = df

            print(df.head(2))
        header = [key for key in df.keys()]

        self.logger.info('Dataframes from projection properly generated')
        self.logger.info(f'Dataframes from projection header:\n{header}')

        return dataframes, header
    
    def read_eps(self, filesRoot = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/eps_files/'):
        self.eps_bin_parser = parseEps(log_level = self.log_level) #parser for binaries files containing eps data
        
        for dir in ['x', 'y', 'z']:
            data_k = self.eps_bin_parser.unpack_binary(path=filesRoot+f'proj_{self.prefix}_{dir}_k.bin', wproj = False, kproj = True, kproj_intra=False, etrans = False)
            data_k_intra = self.eps_bin_parser.unpack_binary(path=filesRoot+f'proj_{self.prefix}_{dir}_k-intra.bin', wproj = False, kproj = False, kproj_intra=True, etrans = False)
            if dir == 'x':
                k_eps_x = data_k[4]
                k_eps_x_intra = data_k_intra[4]
            if dir == 'y':
                k_eps_y = data_k[4]
                k_eps_y_intra = data_k_intra[4]
            if dir == 'z':
                k_eps_z = data_k[4]
                k_eps_z_intra = data_k_intra[4]

        nw = data_k[0]
        nbnd =data_k[1]
        nks = data_k[2]
        wgrid =  data_k[3]

        data_etrans = self.eps_bin_parser.unpack_binary(path=filesRoot+f'proj_{self.prefix}-etrans.bin', wproj = False, kproj = False, kproj_intra=False, etrans = True)
        etrans = data_etrans[3]

        self.logger.info(f'Problem sizes after reading eps binary files: \
                         \n\tnw = {nw} \n\tnbnd ={nbnd} \n\tnks = {nks} \n\twgrid shape = {wgrid.shape}\
                         \n\tk_eps_x shape = {k_eps_x.shape} \n\tetrans shape = {etrans.shape}')
        self.logger.info('Successfully parsed binary eps files!')

        if not self.problems_sizes_sanity_check(nbnd, nks):
            self.logger.critical('Verify parsing of files: problems sizes sanity check failed...')
        else:
            return nw, nbnd, nks, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra

    def problems_sizes_sanity_check(self, nbnd, nks):
        if nbnd == self.nbnd and nks == self.nkstot:
            self.logger.info('Problem sizes sanity check passed!')
            return True
        else:
            return False


    def recover_summed_eps(self, nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra, plot = True):
        self.logger.info(f'Evaluating eps from data read from binary files')
        intersmear = 0.1360
        intrasmear = 0.1360
     
        cor = 1e-6

        k_eps = [k_eps_x, k_eps_y, k_eps_z]
        k_eps_intra = [k_eps_x_intra, k_eps_y_intra, k_eps_z_intra]

        epsi = np.zeros((3, self.nkstot, nw))
        epsr = np.zeros((3, self.nkstot, nw))
        epsi_intra = np.zeros((3, self.nkstot, nw))
        epsr_intra = np.zeros((3, self.nkstot, nw))

        wgrid = np.array(wgrid)
        wgrid_squared = wgrid**2

        etrans_squared = etrans**2
        etrans_diff_squared = etrans_squared[:, :, :, np.newaxis] - wgrid_squared
        aux = (etrans_diff_squared**2 + intersmear**2 * wgrid_squared) * etrans[:, :, :, np.newaxis]
        aux2 = etrans_diff_squared / (aux + cor)

        for dir, keps_dir in enumerate(k_eps):
            epsi[dir,:,:] += np.sum(keps_dir[:, :, :, np.newaxis] * intersmear * wgrid / (aux + cor), axis=(1, 2))
            epsr[dir,:,:] += np.sum(keps_dir[:, :, :, np.newaxis] * aux2, axis=(1, 2))

        
        aux3 = (wgrid**4 + intrasmear**2 * wgrid_squared) * self.degauss

        for dir, keps_dir in enumerate(k_eps_intra):
            epsi[dir,:,:] += np.sum(keps_dir[:, :, np.newaxis] * intrasmear * wgrid / (aux3 + cor), axis=1)
            epsi_intra[dir,:,:] += np.sum(keps_dir[:, :, np.newaxis] * intrasmear * wgrid / (aux3 + cor), axis=1)

            epsr[dir,:,:] -= np.sum(keps_dir[:, :, np.newaxis] * wgrid_squared / (aux3 + cor), axis=1)
            epsr_intra[dir,:,:] -= np.sum(keps_dir[:, :, np.newaxis] * wgrid_squared / (aux3 + cor), axis=1)

        epsi = np.sum(epsi, axis=1)
        epsr = 1. + np.sum(epsr, axis=1)

        epsi_intra = np.sum(epsi_intra, axis=1)
        epsr_intra = 1. + np.sum(epsr_intra, axis=1)

        self.save2csv(wgrid, epsi, filename='epsi(fromBinay).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        self.save2csv(wgrid, epsr, filename='epsr(fromBinay).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        self.save2csv(wgrid, epsi_intra, filename='epsi_intra(fromBinay).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        self.save2csv(wgrid, epsr_intra, filename='epsr_intra(fromBinay).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])

        if plot:
            for id, dir in enumerate(['x', 'y', 'z']):
                plt.plot(wgrid, epsi[id,:], label = f'epsi_{dir}')
                plt.plot(wgrid, epsr[id,:], label = f'epsr_{dir}')
            plt.ylim([-1000,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title('Real and Imag permitivity \nreconstructed from binary data \n(projected on k and on bands)', fontdict={'size':8})
            plt.legend()
            plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(reconstructed).png')

        return epsi, epsr, epsi_intra, epsr_intra 

   
    def recover_summed_eps_proj(self, nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra\
                                ,proj_dataframes_dict, proj_header, plot = True, metalCalc = True):
        intersmear = 0.1360
        intrasmear = 0.1360
        nproj = len(proj_header[6:-1])

        cor = 1e-6

        k_eps = [k_eps_x, k_eps_y, k_eps_z]
        k_eps_intra = [k_eps_x_intra, k_eps_y_intra, k_eps_z_intra]

        epsi_dir_proj = np.zeros((3,nproj,nw))
        epsr_dir_proj = np.zeros((3,nproj,nw))

        epsi_dir = np.zeros((3,nw))
        epsr_dir = np.zeros((3,nw))

        for id, dir in enumerate(['x','y','z']):
            self.logger.info(f'Running over direction {dir}')
            denominator = np.zeros(self.nkstot)
            numerator = np.zeros(self.nkstot)

            aux_imag = np.zeros((self.nkstot, nw))
            aux_real = np.zeros((self.nkstot, nw))

            dif = np.zeros(self.nkstot)
            et2 = np.zeros(self.nkstot)
            epsi = np.zeros((self.nkstot, nw))
            epsr = np.zeros((self.nkstot, nw))

            epsi_ = np.zeros((self.nkstot, nw, self.nbnd, self.nbnd))
            epsr_ = np.zeros((self.nkstot, nw, self.nbnd, self.nbnd))

            epsi_proj = np.zeros((self.nkstot, nproj, nw, self.nbnd))
            epsr_proj = np.zeros((self.nkstot, nproj, nw, self.nbnd))
            

            for iband1 in range(self.nbnd):
                for iband2 in range(self.nbnd):  
                    if iband2 == iband1: continue

                    et2[:] = etrans[:,iband1,iband2]**2
                    
                    for iw, w in enumerate(wgrid): 
                        dif[:] = (et2[:] - w**2)**2
                        denominator[:] = ((dif + intersmear**2 * w**2) * etrans[:,iband1,iband2]) + cor
                        numerator[:] = et2[:] - w**2

                        aux_imag[:, iw] =  k_eps[id][:,iband1,iband2] * intersmear * w / denominator[:]
                        aux_real[:, iw] =  k_eps[id][:,iband1,iband2] * numerator / denominator[:]

                        epsi[:, iw] += aux_imag[:, iw]
                        epsr[:, iw] += aux_real[:, iw]

                    epsi_[:, :, iband1, iband2] += aux_imag[:, :]
                    epsr_[:, :, iband1, iband2] += aux_real[:, :]

            epsi_ = np.sum(epsi_, axis=-1) # summing over iband2
            epsr_ = np.sum(epsr_, axis=-1) # summing over iband2
            
            if metalCalc:
                for iband1 in range(self.nbnd):            
                    for iw, w in enumerate(wgrid):
                        denominator[:] = (w**4 + intrasmear**2 * w**2 ) * self.degauss + cor

                        aux_imag[:, iw] = k_eps_intra[id][:,iband1] * intrasmear * w / denominator[:]
                        aux_real[:, iw] = k_eps_intra[id][:,iband1] * w**2 / denominator[:]

                        epsi[:, iw] += aux_imag[:, iw]
                        epsr[:, iw] -= aux_real[:, iw]

                    epsi_[:, :, iband1] += aux_imag[:, :]
                    epsr_[:, :, iband1] -= aux_real[:, :]
            

            for ia, atmwfc in enumerate(proj_header[6:-1]):
                for iband1 in range(self.nbnd):
                    filtered_proj, filtered_proj_header = self.get_atm_proj(iband1, proj_dataframes_dict, proj_header)          
                    proj = filtered_proj[:, ia] * filtered_proj[:, -1]

                    for iw, w in enumerate(wgrid):
                        epsi_proj[:, ia, iw, iband1] = epsi_[:, iw, iband1] * proj
                        epsr_proj[:, ia, iw, iband1] = epsr_[:, iw, iband1] * proj

            epsi = np.sum(epsi, axis=0)
            epsr = 1. + np.sum(epsr, axis=0)

            epsi_proj = np.sum(epsi_proj, axis=(0, -1)) # summing over k and iband 1
            epsr_proj = 1. + np.sum(epsr_proj, axis=(0, -1)) # summing over k and iband 1

            epsi_proj_sum = np.sum(epsi_proj, axis=(0))
            epsr_proj_sum = np.sum(epsr_proj, axis=(0))

            epsi_dir_proj[id,:,:] = epsi_proj[:,:]
            epsr_dir_proj[id,:,:] = epsr_proj[:,:]

            epsi_dir[id,:] = epsi
            epsr_dir[id,:] = epsr

        self.save2csv(wgrid, epsi, filename='epsi.csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        self.save2csv(wgrid, epsr, filename='epsr.csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        #self.save2csv(wgrid, epsi_intra, filename='epsi_intra(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        #self.save2csv(wgrid, epsr_intra, filename='epsr_intra(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])


        if plot:
            plt.clf()
            
            plt.plot(wgrid, epsi[:], label = f'epsi_x')
            plt.plot(wgrid, epsr[:], label = f'epsr_x')
            for ia, atmwfc in enumerate(proj_header[6:-1]):
                plt.plot(wgrid, epsi_proj[ia,:], label = f'epsi_x_{atmwfc}')
                plt.plot(wgrid, epsr_proj[ia,:], label = f'epsr_x_{atmwfc}')
            plt.plot(wgrid, epsi_proj_sum[:], label = f'epsi_x_summed')
            plt.plot(wgrid, epsr_proj_sum[:], label = f'epsr_x_summed')
            plt.ylim([-1000,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title('Real and Imag permitivity \nreconstructed from binary data \n(before projecting over bands)', fontdict={'size':8})
            plt.legend()
            plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(beforeprojection).png')

            plt.clf()
            plt.plot(wgrid, epsi[:], label = f'epsi_x')
            plt.plot(wgrid, epsr[:], label = f'epsr_x')
            for id, dir in enumerate(['x','y','z']):
                plt.plot(wgrid,  np.sum(epsi_dir_proj, axis=1)[id, :], label = f'epsi_{dir}_summed')
                plt.plot(wgrid,  np.sum(epsr_dir_proj, axis=1)[id, :], label = f'epsr_{dir}_summed')
            plt.ylim([-1000,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title('Real and Imag permitivity \nreconstructed from binary data \n(before projecting over bands)', fontdict={'size':8})
            plt.legend()
            plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(proj_dir).png')

        
        return epsi, epsr, None, None


    
    def recover_summed_eps_proj_wrong(self, nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra\
                                ,proj_dataframes_dict, proj_header, plot = True, metalCalc = True):
        self.logger.info(f'Evaluating eps from data read from binary files')
        intersmear = 0.1360
        intrasmear = 0.1360
     
        cor = 1e-6

        nproj = len(proj_header[6:-1])

        k_eps = [k_eps_x, k_eps_y, k_eps_z]
        k_eps_intra = [k_eps_x_intra, k_eps_y_intra, k_eps_z_intra]

        epsi = np.zeros((self.nkstot, nw))
        epsr = np.zeros((self.nkstot, nw))
        epsi_intra = np.zeros((self.nkstot, nw))
        epsr_intra = np.zeros((self.nkstot, nw))

        epsi_proj = np.zeros((self.nkstot, nproj, nw))
        epsr_proj = np.zeros((self.nkstot, nproj, nw))
        epsi_intra_proj = np.zeros((self.nkstot, nproj, nw))
        epsr_intra_proj = np.zeros((self.nkstot, nproj, nw))
       
        for ia, atmwfc in enumerate(proj_header[6:-1]):
             for iband1 in range(self.nbnd):      
                filtered_proj, filtered_proj_header = self.get_atm_proj(iband1, proj_dataframes_dict, proj_header)          
                proj = filtered_proj[:, ia] * filtered_proj[:, -1]
                proj = 1

                for iband2 in range(self.nbnd): 
                    if iband2 == iband1: continue
                    for iw, w in enumerate(wgrid):                        
                        aux = ((etrans[:,iband1,iband2]**2 - w**2)**2 + intersmear * w**2) * etrans[:,iband1,iband2]   
                                                
                        #for dir, keps_dir in enumerate(k_eps):
                        epsi[:, iw] += proj * k_eps_x[:,iband1,iband2] * intersmear * w / (aux + cor)
                        epsr[:, iw] += proj * k_eps_x[:,iband1,iband2] * (etrans[:,iband1,iband2]**2 - w**2) / (aux + cor)
                        epsi_proj[:, ia, iw] += proj * k_eps_x[:,iband1,iband2] * intersmear * w / (aux + cor)
                        epsr_proj[:, ia, iw] += proj * k_eps_x[:,iband1,iband2] * (etrans[:,iband1,iband2]**2 - w**2) / (aux + cor)

                #intraband
                if metalCalc:
                    for iw, w in enumerate(wgrid):  
                        aux3 = ( w**4 + intrasmear * w**2) * self.degauss     
                        epsi[:, iw] += proj * k_eps_x_intra[:,iband1] * intrasmear * w / (aux3 + cor)
                        epsr[:, iw] -= proj * k_eps_x_intra[:,iband1] *  w**2 / (aux3 + cor)
                        epsi_proj[:, ia, iw] += proj * k_eps_x_intra[:,iband1] * intrasmear * w / (aux3 + cor)
                        epsr_proj[:, ia, iw] -= proj * k_eps_x_intra[:,iband1] *  w**2 / (aux3 + cor)
                        epsi_intra[:,iw] += proj * k_eps_x_intra[:,iband1] * intrasmear * w / (aux3 + cor)
                        epsr_intra[:, iw] -= proj * k_eps_x_intra[:,iband1] *  w**2 / (aux3 + cor)
                        epsi_intra_proj[:, ia, iw] += proj * k_eps_x_intra[:,iband1] * intrasmear * w / (aux3 + cor)
                        epsr_intra_proj[:, ia, iw] -= proj * k_eps_x_intra[:,iband1] *  w**2 / (aux3 + cor)
       
        epsi = np.sum(epsi, axis=0)
        epsr = 1. + np.sum(epsr, axis=0)

        epsi_proj = np.sum(epsi_proj, axis=0)
        epsr_proj = 1. + np.sum(epsr_proj, axis=0)

        epsi_intra = np.sum(epsi_intra, axis=0)
        epsr_intra = 1. + np.sum(epsr_intra, axis=0)

        #self.save2csv(wgrid, epsi, filename='epsi(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        #self.save2csv(wgrid, epsr, filename='epsr(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        #self.save2csv(wgrid, epsi_intra, filename='epsi_intra(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])
        #self.save2csv(wgrid, epsr_intra, filename='epsr_intra(fromBinary).csv', directory = './results/csv', transpose=True,  columns=['epsi_x','epsi_y','epsi_z'])

        if plot:
            plt.clf()
            
            plt.plot(wgrid, epsi[:], label = f'epsi_x')
            plt.plot(wgrid, epsr[:], label = f'epsr_x')
            plt.ylim([-1000,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title('Real and Imag permitivity \nreconstructed from binary data \n(projected on k and on bands)', fontdict={'size':8})
            plt.legend()
            plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(reconstructed_afterBands).png')

            plt.clf()
            for ia, atmwfc in enumerate(proj_header[6:-1]):
                plt.plot(wgrid,epsi_proj[ia,:], label=f'epsi_{atmwfc}')
                plt.plot(wgrid,epsr_proj[ia,:], label=f'epsr_{atmwfc}')
            plt.plot(wgrid, epsi[:],   '-*', label = f'epsi_x')
            plt.plot(wgrid, epsr[:], '-*', label = f'epsr_x')
            plt.ylim([-1000,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title('Real and Imag permitivity \nreconstructed from binary data \n(projected on k and on bands)', fontdict={'size':8})
            #plt.legend()
            plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(projBands).png')

        
        return epsi, epsr, epsi_intra, epsr_intra   

    def atmwfc_projected_eps(self, nw, wgrid, etrans, k_eps_x, k_eps_x_intra, proj_dataframes_dict, proj_header, direction='x'):
        self.logger.info(f'Number of atomic projections = {self.natomwfc}')
        self.logger.info(f'Evaluating eps projections over atomic orbitals for direction {direction} (takes time...)')
        epsi_proj = np.zeros((self.natomwfc, self.nkstot, nw))
        epsr_proj = np.zeros((self.natomwfc, self.nkstot, nw))

        epsi_proj_intra = np.zeros((self.natomwfc, self.nkstot, nw))
        epsr_proj_intra = np.zeros((self.natomwfc, self.nkstot, nw))

        intersmear = 0.136
        intrasmear = 0.136
        cor = 1e-6 

        # Precompute wgrid squared
        wgrid_squared = wgrid**2

        for choose_projection in range(self.natomwfc):
            for iband1 in range(self.nbnd):
                filtered_proj, filtered_proj_header = self.get_atm_proj(iband1, proj_dataframes_dict, proj_header)
                if iband1 == 0:
                    self.logger.info(f'\tEvaluating projection {filtered_proj_header[choose_projection]}')
                projFactor = np.ones_like(filtered_proj[:, choose_projection] * filtered_proj[:, -1])

                for iband2 in range(self.nbnd):
                    if iband2 == iband1: continue

                    etrans_diff_squared = etrans[:, iband1, iband2][:, np.newaxis]**2 - wgrid_squared
                    aux = (etrans_diff_squared**2 + intersmear**2 * wgrid_squared) * etrans[:, iband1, iband2][:, np.newaxis]
                    aux2 = etrans_diff_squared / (aux + cor)

                    epsi_proj[choose_projection,:,:] += projFactor[:, np.newaxis] * k_eps_x[:, iband1, iband2][:, np.newaxis] * intersmear * wgrid / (aux + cor)
                    epsr_proj[choose_projection,:,:] += projFactor[:, np.newaxis] * k_eps_x[:, iband1, iband2][:, np.newaxis] * aux2

                #intraband
                aux3 = (wgrid**4 + intersmear**2 * wgrid_squared) * self.degauss

                epsi_proj[choose_projection,:,:] += projFactor[:, np.newaxis] * k_eps_x_intra[:, iband1][:, np.newaxis] * intrasmear * wgrid / (aux3 + cor)
                epsr_proj[choose_projection,:,:] -= projFactor[:, np.newaxis] * k_eps_x_intra[:, iband1][:, np.newaxis] * wgrid**2 / (aux3 + cor)

                epsi_proj_intra[choose_projection,:,:] += projFactor[:, np.newaxis] * k_eps_x_intra[:, iband1][:, np.newaxis] * intrasmear * wgrid / (aux3 + cor)
                epsr_proj_intra[choose_projection,:,:] -= projFactor[:, np.newaxis] * k_eps_x_intra[:, iband1][:, np.newaxis] * wgrid**2 / (aux3 + cor)

        epsi = np.sum(epsi_proj, axis = 0) #summing over the orbitals
        epsr = np.sum(epsr_proj, axis = 0) #summing over the orbitals

        epsi_proj = np.sum(epsi_proj, axis=1)        
        epsr_proj = 1. + np.sum(epsr_proj, axis=1)        

        epsi_proj_intra = np.sum(epsi_proj_intra, axis=1)
        epsr_proj_intra = 1. + np.sum(epsr_proj_intra, axis=1)

        epsi = np.sum(epsi, axis = 0)
        epsr = 1. + np.sum(epsr, axis = 0)

        epsi_intra = np.sum(epsi_proj_intra, axis = 0)
        epsr_intra = np.sum(epsr_proj_intra, axis = 0)

        self.save2csv(wgrid, epsi_proj, filename=f'epsi(projected)_{direction}.csv', directory = './results/csv', transpose=True,  columns=filtered_proj_header[:-1])
        self.save2csv(wgrid, epsr_proj, filename=f'epsr(projected)_{direction}.csv', directory = './results/csv', transpose=True,  columns=filtered_proj_header[:-1])
        self.save2csv(wgrid, epsi_proj_intra, filename=f'epsi_intra(projected)_{direction}.csv', directory = './results/csv', transpose=True,  columns=filtered_proj_header[:-1])
        self.save2csv(wgrid, epsr_proj_intra, filename=f'epsi_intra(projected)_{direction}.csv', directory = './results/csv', transpose=True,  columns=filtered_proj_header[:-1])

        self.save2csv(wgrid, epsi, filename=f'epsi(sum_over_projection)_{direction}.csv', directory = './results/csv', transpose=True,  columns=[f'epsi_{direction}'])
        self.save2csv(wgrid, epsr, filename=f'epsr(sum_over_projection)_{direction}.csv', directory = './results/csv', transpose=True,  columns=[f'epsi_{direction}'])

        return epsi_proj, epsr_proj, epsi_proj_intra, epsr_proj_intra, epsi, epsr, epsi_intra, epsr_intra   
    
    def save2csv(self, wgrid, array, filename, directory = './results/csv', **kwargs):
        # Create the directory if it doesn't exist
        if kwargs['transpose']:
            array = array.T
        if 'columns' in kwargs:
            columns = kwargs['columns']
        else:
            columns = None


        if not os.path.exists(directory):
            os.makedirs(directory)

        df = pd.DataFrame(array, columns=columns)
        df.insert(0, 'ev', wgrid)
        df.to_csv(os.path.join(directory, filename), index=False)


    def get_atm_proj(self, iband, proj_dataframes_dict, proj_header):
        proj = proj_dataframes_dict[f'KS{iband+1}']
        filtered_proj = proj.values[:,6:]
        filtered_proj_header = proj_header[6:]
        return filtered_proj, filtered_proj_header   

    def get_params(self, projWFCfilename = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/projWfc_files/al.projwfc.out') :
        with open(projWFCfilename, 'r') as f:
            lines = f.readlines()

        # Extract natomwfc, nbnd, nkstot
        for line in lines:
            if line.strip().startswith("natomwfc"):
                natomwfc = int(line.split("=")[1].strip())
            elif line.strip().startswith("nbnd"):
                nbnd = int(line.split("=")[1].strip())
            elif line.strip().startswith("nkstot"):
                nkstot = int(line.split("=")[1].strip())
            elif line.strip().startswith("Gaussian broadening"):
                degauss = float(line.split("=")[1].strip().split()[1])
        
        self.logger.info(f'Problem sizes read from prokwfc.out: \n\tnatomwfc =  {natomwfc} \n\tnbnd =  {nbnd} \n\tnkstot =  {nkstot} \n\tdegauss = {degauss}')
        return {'natomwfc': natomwfc, 'nbnd': nbnd, 'nkstot': nkstot, 'degauss': degauss}
    
    def run(self, plot = False):
        proj_dataframes_dict, proj_header = self.read_projections2dataframe() # returns a dictionary with the projection dataframes for each KS state-
        nw, nbnd, nks, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra = self.read_eps()

        filtered_proj_header = proj_header[6:]
    
        epsi, epsr, epsi_intra, epsr_intra = an.recover_summed_eps(nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra)
    
        #epsi_proj, epsr_proj, epsi_proj_intra, epsr_proj_intra, \
        #    epsi_total, epsr_total, epsi_intra, epsr_intra= an.atmwfc_projected_eps(nw, \
         #                                                                       wgrid, etrans, k_eps_x, k_eps_x_intra, proj_dataframes_dict, proj_header)
        
        epsi_proj, epsr_proj, epsi_intra, epsr_intra = an.recover_summed_eps_proj(nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra\
                                ,proj_dataframes_dict, proj_header, plot = True)
        
        if plot:
            plt.clf()
            for ic in range(len(filtered_proj_header)-1):
                plt.plot(wgrid, epsi_proj[:], label = f'epsi_{filtered_proj_header[ic]}')
                #plt.plot(wgrid, epsi_total, label = f'epsi_total')
                plt.plot(wgrid, epsr_proj[:], label = f'epsr_{filtered_proj_header[ic]}')
                #plt.plot(wgrid, epsr_total, label = f'epsr_total')
            plt.ylim([-100,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title(f'Real and Imag permitivity \n(projected onto atomic wavefunctions)', fontdict={'size':8})
            plt.legend(bbox_to_anchor=(0.8, 0.8), loc='upper left', borderaxespad=0.0, fontsize=6)
            plt.savefig('./results/figures/eps(projected).png')

            plt.clf()
            plt.plot(wgrid, epsi[0,:], label = f'epsi')
            plt.plot(wgrid, epsi_proj, label = f'epsi_projection')
            plt.plot(wgrid, epsr[0,:], label = f'epsr')
            plt.plot(wgrid, epsr_proj, label = f'epsr_projection')
            plt.ylim([-100,200])
            plt.xlim([0.01,12])
            plt.xlabel(r'$\hbar\omega$ (ev)')
            plt.ylabel(r'permitivity')
            plt.title(f'Real and Imag permitivity \n comparison of total and projected summed', fontdict={'size':8})
            plt.legend()
            plt.savefig('./results/figures/eps(comparison).png')


if __name__ == '__main__':

    log_level = 'info'
    prefix = 'al'
    projFilename = '/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/projectionFiles/structured_projection_dataframes.h5'
    an = analyzer(projFilename = projFilename, log_level = log_level, prefix = prefix)

    an.run()

    '''proj_dataframes_dict, proj_header = an.read_projections2dataframe() # returns a dictionary with the projection dataframes for each KS state-
    nw, nbnd, nks, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra = an.read_eps()
  

    filtered_proj_header = proj_header[6:]
    
    epsi, epsr, epsi_intra, epsr_intra = an.recover_summed_eps(nw, wgrid, k_eps_x, k_eps_y, k_eps_z, etrans, k_eps_x_intra, k_eps_y_intra, k_eps_z_intra)
    
    epsi_proj, epsr_proj, epsi_proj_intra, epsr_proj_intra, \
        epsi_total, epsr_total, epsi_intra, epsr_intra= an.atmwfc_projected_eps(nw, \
                                                                                wgrid, etrans, k_eps_x, k_eps_x_intra, proj_dataframes_dict, proj_header)'''

'''
    for ic in range(len(filtered_proj_header)-1):
        plt.plot(wgrid, epsi_proj[ic,:], label = f'epsi_{filtered_proj_header[ic]}')
        plt.plot(wgrid, epsi_total, label = f'epsi_total')
        plt.plot(wgrid, epsr_proj[ic,:], label = f'epsr_{filtered_proj_header[ic]}')
        plt.plot(wgrid, epsr_total, label = f'epsr_total')
    plt.ylim([-100,200])
    plt.xlim([0.01,12])
    plt.xlabel(r'$\hbar\omega$ (ev)')
    plt.ylabel(r'permitivity')
    plt.title(f'Real and Imag permitivity \n(projected onto atmwfc={filtered_proj_header[choose_projection]})', fontdict={'size':8})
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(projected).png')

    plt.clf()
    plt.plot(wgrid, epsi[0,:], label = f'epsi')
    plt.plot(wgrid, epsi_total, label = f'epsi_projection')
    plt.plot(wgrid, epsr[0,:], label = f'epsr')
    plt.plot(wgrid, epsr_total, label = f'epsr_projection')
    plt.ylim([-100,200])
    plt.xlim([0.01,12])
    plt.xlabel(r'$\hbar\omega$ (ev)')
    plt.ylabel(r'permitivity')
    plt.title(f'Real and Imag permitivity \n comparison of total and projected summed', fontdict={'size':8})
    plt.legend()
    plt.savefig('/home/anibal/scratch/DFT/ProjWFC/Al_4atoms/TransitionDipoleAnalyser/results/figures/eps(comparison).png')
    '''
