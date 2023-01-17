#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#%%
data = io.imread('/ccpi/data/imat/001/ffcorr-corr-dead-px-all/scan_0031.tif')

ncols = data.shape[2]
nrows = data.shape[1]
n_frames = data.shape[0]

#%%
# do some precalculations for phase retrieval
# phase image
energy = 30.4
delta = 1e-7
thresholding_rate = 0.01 
pixel_size = 0.69e-6
propagation_distance_x = 0.22
propagation_distance_y = 0.6
frequency_cutoff = 1e30
regularize_rate = 2.0
padded_width = 4096

lam = 6.62606896e-34 * 299792458 / (energy * 1.60217733e-16)

if delta is not None:  
    thickness_conversion = -lam / (2 * np.pi * delta)
else:
    thickness_conversion = 1

thickness_conversion *= -10 ** regularize_rate / 2

tmp = np.pi * lam / (pixel_size * pixel_size)

prefac_x = tmp * propagation_distance_x
prefac_y = tmp * propagation_distance_y
    
pad_width_cols = (padded_width - ncols) // 2
pad_width_rows = (padded_width - nrows) // 2

x = np.fft.fftfreq(padded_width, d=1.0)
xx,yy = np.meshgrid(x, x)

sin_arg = prefac_x * (xx * xx) + prefac_y * (yy * yy)

filt = np.zeros((padded_width,padded_width), dtype=np.float32)
filt[sin_arg < frequency_cutoff] = 0.5 / (sin_arg[sin_arg < frequency_cutoff] + 10**(-regularize_rate))

frames_abs = np.zeros((n_frames,nrows,ncols), dtype=np.float32)
frames_phase = np.zeros((n_frames,nrows,ncols), dtype=np.float32)

for i in range(n_frames):
    print(i)

    tmp = data[i,:,:]
    noise = np.random.normal(0, 0.1, tmp.shape)
    tmp += noise
    
    nonzero = tmp>0
    # absorption image
    frames_abs[i, nonzero] = -np.log(tmp[nonzero])

    # phase image
    im_pad = np.pad(tmp, ((pad_width_rows, pad_width_rows), (pad_width_cols, pad_width_cols+1)), mode='edge')
    im_fft = np.fft.fft2(im_pad)
    im_phase = np.float32(np.real(np.fft.ifft2(filt * im_fft)))
    im_phase_crop = im_phase[pad_width_rows:-pad_width_rows, pad_width_cols:(-pad_width_cols-1)]
    im_phase_corr = np.zeros((nrows, ncols), dtype=np.float32)
    im_phase_corr[im_phase_crop > 0] = -np.log(2 / 10 ** regularize_rate * im_phase_crop[im_phase_crop > 0]) * thickness_conversion
    frames_phase[i, :, :] = im_phase_corr

# %% compare with tofu results

data_ref = io.imread('/ccpi/data/imat/001/pr-pag-all/scan_0031-pr-pag-2p0.tif')

plt.figure(figsize=(10,10))
plt.imshow(frames_phase[0, :, :])
plt.colorbar()
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(data_ref[0, :, :])
plt.colorbar()
plt.show()

# %%