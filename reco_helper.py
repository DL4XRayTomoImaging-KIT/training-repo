from cil.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData
from cil.plugins.astra import ProjectionOperator
from cil.plugins.astra.processors import FBP
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm

def reconstruct_one_slice(sino, n_proj=120, n_channels=None, imsize=512):
    # crop data to centre
    sino = sino[:, :, :-28]
    n_channels = n_channels or sino.shape[0]
    
    # set-up ccpi objects
    # multi-channel geometry
    ag_MC = AcquisitionGeometry.create_Parallel2D()
    ag_MC.set_panel(sino.shape[2], pixel_size=0.055)
    ag_MC.set_channels(num_channels=n_channels)
    ag_MC.set_angles(np.linspace(0, 180, n_proj, endpoint=False), angle_unit='degree')
    # allocate AcquisitionGeometry and fill it in with actual sinogram
    ad_MC = ag_MC.allocate()
    ad_MC.fill(sino)
    # generate ImagGeometry
    ig_MC = ag_MC.get_ImageGeometry()
    
    # actual projection operator for iterative reconstruction
    op_MC = ProjectionOperator(ig_MC, ag_MC, 'gpu')

    # single slice geometry
    ag = ag_MC.get_slice(channel=0)
    ig = ag.get_ImageGeometry()

    #%% FBP recon
    FBP_recon = ig_MC.allocate()

    # FBP reconstruction per channel
    for i in tqdm(range(ig_MC.channels), leave=False, desc='reconstructing channels'):

        FBP_recon_2D = FBP(ig, ag, 'gpu')(ad_MC.get_slice(channel=i))
        FBP_recon.fill(FBP_recon_2D, channel=i)

        # print("Finish FBP recon for channel {}".format(i), end='\r')

    # print("\nFBP Reconstruction Complete!")
    # get actual numpy array
    FBP_recon = FBP_recon.as_array()
    
    return FBP_recon


#%% here is long and boring script to prepare the spectral axis and to parse
# text files with ground truth
# it was written in a rush and I never bothered to clean it up.
import csv
import sklearn.metrics
from matplotlib import pyplot as plt

def get_spectrum(reconstructed_slice_left, 
                 reconstructed_slice_right, 
                 name_left='FBP', 
                 name_right='Something else', 
                 binning=[16, 8, 8, 4]):

    # parse specral info
    # shutter values
    tof_lim_left = np.array([15e-3, 27e-3, 44e-3, 53e-3])
    tof_lim_right = np.array([26.68e-3, 43.68e-3, 52.68e-3, 72e-3])
    tof_bin_width = np.array([10.24, 20.48, 20.48, 40.96])

    # number of shutter intervals
    n_intervals = 4

    # binning in each shutter interval
    binning = np.array(binning)

    # calculate number of bins in each shutter interval
    # TOF is in seconds, bins in microseconds
    n_bins = np.int_(np.floor((tof_lim_right - tof_lim_left) / (tof_bin_width * binning * 1e-6)))
    n_bins_total = np.sum(n_bins)

    ## prepare spectral axis for plots
    tof_bins_left = np.zeros((n_bins_total), dtype = np.float32)
    tof_bins_right = np.zeros((n_bins_total), dtype = np.float32)
    counter = 0
    for i in range(n_intervals):
        tof_bins_left[counter:(counter+n_bins[i])] = np.arange(tof_lim_left[i], tof_lim_right[i]-tof_bin_width[i]*binning[i]*1e-6, tof_bin_width[i]*binning[i]*1e-6, dtype = np.float32)
        tof_bins_right[counter:(counter+n_bins[i])] = tof_bins_left[counter:(counter+n_bins[i])] + (tof_bin_width[i]*binning[i]*1e-6)
        counter = counter+n_bins[i]

    tof_bins_center = ((tof_bins_left + tof_bins_right) / 2)

    l = 56.428
    # full equation
    # angstrom_lim_1 = (tof_lim_1 * const.h) / (const.m_n * l) * 1e10
    # angstrom_lim_2 = (tof_lim_2 * const.h) / (const.m_n * l) * 1e10
    # and it's simplified form
    angstrom_lim_1 = (tof_lim_left * 3957) / l
    angstrom_lim_2 = (tof_lim_right * 3957) / l
    angstrom_bins_center = (tof_bins_center * 3957) / l
    angstrom_bin_width = (tof_bin_width * 1e-6 * 3957) / l

    # import table data with attenuation values for different wavelength
    attenuation_zn = np.zeros((3974, 2), dtype=np.float32)
    with open('Zn_new.txt', 'r') as attenuation_values_csv:
        csv_reader = csv.reader(attenuation_values_csv, delimiter = '\t')
        
        # skip first row with column names
        next(csv_reader)
        counter = 0
        for row in csv_reader:
            
            attenuation_zn[counter, :] = row
            counter += 1

    attenuation = np.zeros((5998, 11), dtype = np.float32)
    with open('attenuation.txt', 'r') as attenuation_values_csv:
        csv_reader = csv.reader(attenuation_values_csv, delimiter = '\t')
        
        # skip first row with column names
        next(csv_reader)
        counter = 0
        for row in csv_reader:
            attenuation[counter, :] = row
            counter += 1

    # wavelength - first column
    wavelength = np.sort(np.copy(attenuation_zn[:, 0]))

    attenuation_zn = attenuation_zn[:, 1]
    # interpolate attenuation on common axis
    attenuation_tmp = np.interp(angstrom_bins_center, wavelength, attenuation_zn)
    attenuation_zn = np.copy(attenuation_tmp)
    attenuation_zn[-1] = attenuation_zn[-2]

    # wavelength - first column
    wavelength = np.copy(attenuation[:, 0])
    attenuation[:, :-1] = attenuation[:, 1:]
    # add a column with attenuation for air - currently zero
    attenuation[:, -1] = 0
    # column names
    materials = np.array(["Ti", "Fe", "Ni", "Cu", "Cd", "W", "Pb", "Al", "Zn", "ZnO", "Air"])

    # interpolate attenuation on common axis
    attenuation_tmp = np.zeros((n_bins_total,11), dtype = np.float32)

    for i in range(11):
        attenuation_tmp[:,i] = np.interp(angstrom_bins_center, wavelength, attenuation[:,i])

    attenuation = attenuation_tmp
    attenuation[:, 8] = attenuation_zn

    #%% plot results - spectral comparison
    # now it compares FBP vs FBP :) in your case I guess it will be FBP vs denoised FBP
    # scale pixel values
    FBP_recon_l = reconstructed_slice_left / 0.055
    FBP_recon_r = reconstructed_slice_right / 0.055

    # set-up plots
    sample_materials = np.array(["Fe", "Ni", "Cu",  "Zn", "Al"])

    #green, orange, blue, black, terracota
    colors = [[44/255, 162/255, 95/255],\
              [227/255, 74/255, 51/255],\
              [44/255, 127/255, 184/255],\
              [99/255, 99/255, 99/255],\
              [191/255, 91/255, 23/255]]

    # this coefficients scale gray values 
    # the reason why we need to do this is because 
    # we used not solid materails but powders
    # i.e. each reconstucted voxel contains a mixture of
    # air and maetrial
    coeff = [0.65, 0.9, 0.55, 1, 0.9]

    # roi coordinates and size which we used to generate
    # spectral profiles
    # roi_row = np.array([362, 242, 120, 132, 252])
    # roi_col = np.array([182, 382, 150, 312, 100])

    roi_row = np.array([362, 242, 120, 132, 252])
    roi_col = np.array([182, 382, 150, 312, 100])

    roi_pix_v = 1
    roi_pix_h = 1

    nrows = 5
    ncols = 2

    fig, (axes) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 7), 
                               dpi=100, sharex=True, sharey=False)

    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                if i == 0:
                    axes[i,j].set_title(name_left)
                predicted_line = 1/coeff[i]*np.mean(np.mean(FBP_recon_l[:, roi_row[i]:(roi_row[i]+roi_pix_v), roi_col[i]:(roi_col[i]+roi_pix_h)], axis = 2), axis = 1)
                theoretical_line = attenuation[:, materials == sample_materials[i]]

                axes[i,j].plot(angstrom_bins_center, predicted_line, color=colors[0], linestyle='-', linewidth=1)
                axes[i,j].plot(angstrom_bins_center, theoretical_line, color = 'black', linestyle='--', linewidth=1)
                axes[i,j].set_ylabel(sample_materials[i]+f'\n{sklearn.metrics.mean_absolute_error(theoretical_line, predicted_line):.3f}', fontsize=13)
                axes[i,j].get_yaxis().set_label_coords(-0.2,0.5)      
            elif j == 1:
                if i == 0:
                    axes[i,j].set_title(name_right)
                predicted_line = 1/coeff[i]*np.mean(np.mean(FBP_recon_r[:, roi_row[i]:(roi_row[i]+roi_pix_v), roi_col[i]:(roi_col[i]+roi_pix_h)], axis = 2), axis = 1)
                theoretical_line = attenuation[:, materials == sample_materials[i]]
                
                axes[i,j].plot(angstrom_bins_center, predicted_line, color=colors[0], linestyle='-', linewidth=1)
                axes[i,j].plot(angstrom_bins_center, theoretical_line, color = 'black', linestyle='--', linewidth=1)
                axes[i,j].tick_params(axis='both', which='both', left=False)
                axes[i,j].set_yticklabels([])
                axes[i,j].yaxis.set_label_position("right")
                axes[i,j].set_ylabel(sample_materials[i]+f'\n{sklearn.metrics.mean_absolute_error(theoretical_line, predicted_line):.3f}', fontsize=13)
                
            axes[i,j].set_xlim(np.amin(angstrom_bins_center), np.amax(angstrom_bins_center))
            axes[i,j].set_ylim(0.8*np.amin(attenuation[:, materials == sample_materials[i]]), 1.1*np.amax(attenuation[:, materials == sample_materials[i]]))
            
            if i == 4:
                axes[i,j].set_xlabel(r'$\lambda, \mathrm{\AA}$', fontsize=13)

    fig.tight_layout()
    text = fig.text(-0.01, 0.5, r'$\Sigma_{\mathrm{tot}} (\lambda), [\mathrm{cm}^{-1}]$', ha='center', va='center', rotation='vertical', fontsize=13)
    # Some additional whitespace adjustment is needed
    fig.subplots_adjust(right=0.825, hspace=0.01, wspace=0.01)
    plt.show()

