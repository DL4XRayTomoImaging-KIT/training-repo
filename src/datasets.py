from builtins import breakpoint, isinstance
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from medpy.io import load as medload
from glob import glob
import os
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
import scipy

def convert_target(addr, converter):
    if isinstance(list(converter.keys())[0], str):
        # this is because of the restrictions in the OmegaConf. Should be resolved with 2.1 version.
        converter = {int(k):v for k,v in converter.items()}
    markup = ExpandedPaddedSegmentation(addr)
    markup.data = np.vectorize(converter.get)(markup.data)
    return markup

def supervised_segmentation_target_matcher(volumes, targets):
    label_ids = [os.path.basename(i).split('.')[-2] for i in glob(targets.format('*'))]

    if '-' in label_ids[0]:
        volume_ids = [i.split('-')[1] for i in label_ids]
    else:
        volume_ids = label_ids
    
    return list(zip([volumes.format(i) for i in volume_ids], [targets.format(i) for i in label_ids]))

def same_name_target_matcher(*addrs, names_differentiate=None):
    addr_dicts = []
    file_names = []
    names_differentiate = names_differentiate or [-1]
    for addr in addrs:
        current_addrs_dict = {}
        for fn in glob(addr):
            chunks = fn.split('/')
            f_key = '_'.join([chunks[i] for i in names_differentiate])
            current_addrs_dict[f_key] = fn
        addr_dicts.append(current_addrs_dict)
        file_names.append(set(current_addrs_dict.keys()))
    
    joint_filenames = set.intersection(*file_names)

    result_tuples = []
    for fn in joint_filenames:
        result_tuples.append(tuple([d[fn] for d in addr_dicts]))
    return result_tuples

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    if volumes_limit is not None:
        if isinstance(volumes_limit, float):
            volumes_limit = int(volumes_limit*len(gathered_data))
        gathered_data = gathered_data[:volumes_limit]
    train_data, test_data = train_test_split(gathered_data, random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def get_TVSD_datasets(data_addresses, aug=None, **kwargs):
    datasets = []
    for image_addr, label_addr in data_addresses:
        datasets.append(VolumeSlicingDataset(image_addr, segmentation=label_addr, augmentations=aug,
                                             **kwargs))
    return ConcatDataset(datasets)

def adaptive_choice(choose_from, choice_count):
    if choice_count <= len(choose_from):
        return np.random.choice(choose_from, choice_count, replace=False)
    else:
        subsample = [choose_from]*(choice_count//len(choose_from)) # all the full inclusions first
        subsample.append(np.random.choice(choose_from, choice_count%len(choose_from), replace=False)) # additional records
        return np.concatenate(subsample)

def multiple_dataset_resample(resampling_function):
    def wrapper_resampler(datasets, multiple_datasets_mode='all', **kwargs):
        if multiple_datasets_mode == 'first':
            return [resampling_function(datasets[0], **kwargs)] + datasets[1:]
        elif multiple_datasets_mode == 'all':
            return [resampling_function(dset, **kwargs) for dset in datasets]
        elif multiple_datasets_mode == 'default':
            return [resampling_function(datasets[0], **kwargs)] + [resampling_function(dset) for dset in datasets[1:]]
    
    return wrapper_resampler

@multiple_dataset_resample
def TVSD_dataset_resample(dataset, segmented_part=1.0, empty_part=0.1):
    is_marked = np.concatenate([d.segmentation._contains_markup() for d in dataset.datasets])
    if segmented_part is None:
        segmented_part = 1.0
    if isinstance(segmented_part, float):
        segmented_part = int(is_marked.sum() * segmented_part)        
    
    if isinstance(empty_part, float):
        empty_part = int(segmented_part * empty_part)
    elif empty_part is None:
        empty_part = (1-is_marked).sum()

    segmented_subsample = adaptive_choice(np.where(is_marked)[0], segmented_part)
    empty_subsample = adaptive_choice(np.where(1-is_marked)[0], empty_part)

    return Subset(dataset, np.concatenate([segmented_subsample, empty_subsample]))

from univread import read as imread

def get_paganin_converter(ncols, nrows, noise_sigma=None):
    #%%
    # do some precalculations for phase retrieval
    # phase image
    energy = 30.4
    delta = 1e-7
    thresholding_rate = 0.01 
    pixel_size = 0.8e-6
    propagation_distance_x = 0.22
    propagation_distance_y = 0.6
    frequency_cutoff = 1e30
    regularize_rate = 2.0
    padded_width = 1024

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

    
    def get_one_pr(tmp):
        if noise_sigma is not None:
            noise = np.random.normal(0, noise_sigma, tmp.shape)
            tmp += noise
        im_pad = np.pad(tmp, ((pad_width_rows, pad_width_rows), (pad_width_cols, pad_width_cols+1)), mode='edge')
        im_fft = scipy.fft.fft2(im_pad)
        im_phase = np.float32(np.real(scipy.fft.ifft2(filt * im_fft)))
        im_phase_crop = im_phase[pad_width_rows:-pad_width_rows, pad_width_cols:(-pad_width_cols-1)]
        im_phase_corr = np.zeros((nrows, ncols), dtype=np.float32)
        im_phase_corr[im_phase_crop > 0] = -np.log(2 / 10 ** regularize_rate * im_phase_crop[im_phase_crop > 0]) * thickness_conversion
        return im_phase_corr
    
    return get_one_pr

# class PaganinNoiseCollection:
#     def __init__(self, samples_count, noise_sigma, noise_shape) -> None:
#         self.paganin_converter = get_paganin_converter(noise_shape[1], noise_shape[0])
#         self.noise_shape = noise_shape
#         self.noise_sigma = noise_sigma
#         self.stack = [None] * samples_count
    
#     def __getitem__(self, i):
#         if self.stack[i] is None:
#             noise = np.random.normal(0, self.noise_sigma, self.noise_shape)
#             self.stack[i] = self.paganin_converter(noise+1) - 1
        
#         return self.stack[i]


# GLOBAL_NOISE_STACK = PaganinNoiseCollection(10_000, 0.1, 516, 787)

gau_ssim = lambda a, b: ssim(gaussian(a), gaussian(b))

from skimage.feature import canny
canny_ssim = lambda a,b: ssim(canny(a*1e4, sigma=5), canny(b*1e4, sigma=5))

invert_ssim = lambda a,b: ssim(a - gaussian(a, sigma=5), b - gaussian(b, sigma=5))

class AtrociousSliceSampling(Dataset):
    def __init__(self, volume_addr, atro_mask=[True, False, True], augmentation=None, lazy=False, double_center=False, ssim_threshold=None, ssim_function='gau_ssim', paganin_noise_collection=None, scale_on_load=None):
        self.atro_mask = np.array(atro_mask)
        self.double_center = double_center
        if isinstance(volume_addr, str):
            self.volume = imread(volume_addr, lazy=lazy)
        else:
            self.volume = volume_addr
        
        if scale_on_load is not None:
            self.volume *= scale_on_load

        self.pad = len(self.atro_mask)//2
        self.augmentation = augmentation

        self.ssim_threshold = ssim_threshold
        self.ssim_function = globals()[ssim_function]

        self.paganin_noise = paganin_noise_collection
        
    def __len__(self):
        return len(self.volume) - self.pad - 1

    def _get_one_pair(self, i):
        i = i + self.pad
        if self.double_center:
            inp = np.stack([self.volume[i], self.volume[i]])
        else:
            inp = self.volume[i-self.pad:i+self.pad+1][self.atro_mask]
        out = self.volume[None, i]

        if self.paganin_noise is not None:
            noise = imread(self.paganin_noise, lazy=True)
            inp += noise[np.random.randint(noise.shape[0])]
            out += noise[np.random.randint(noise.shape[0])]
            del noise
            # inp += np.roll(noise_in, (np.random.randint(512), np.random.randint(512)), axis=(0, 1))[None, ...]
            # out += np.roll(noise_out, (np.random.randint(512), np.random.randint(512)), axis=(0, 1))[None, ...]
        
        if self.augmentation is not None:
            mul = np.concatenate([inp, out])
            augmented = self.augmentation(image=np.moveaxis(mul, 0, -1))
            mul = np.moveaxis(augmented['image'], -1, 0)
            inp = mul[:-1]
            out = mul[[-1]]
        
        return inp, out
    
    def __getitem__(self, i):
        if self.ssim_threshold is None:
            return self._get_one_pair(i)
        else:
            pairs = []
            ssims = []
            for i in range(4):
                pair = self._get_one_pair(i)
                ssi = self.ssim_function(pair[0][0], pair[1][0])

                if ssi > self.ssim_threshold:
                    return pair
                else:
                    pairs.append(pair)
                    ssims.append(ssi)
            return pairs[np.argmin(ssims)]

def get_atro_denoising_datasets(data_adresses, aug=None, **kwargs):
    datasets = [AtrociousSliceSampling(da[0], augmentation=aug, **kwargs) for da in data_adresses]
    return ConcatDataset(datasets)


from scipy.signal import convolve2d

footprint = np.array([[0, 1, 0], 
                      [1, 0, 1], 
                      [0, 1 ,0]])

footmask = footprint / footprint.sum()

def regular_grid(shape, part):
    x, y = np.meshgrid(*[np.arange(i) for i in shape])
    mesh = int(1/part)
    if mesh < 4:
        return (x + y)%mesh == np.random.randint(mesh)
    else:
        mesh = int(np.sqrt(mesh))
        return (x%mesh == np.random.randint(mesh)) & (y%mesh == np.random.randint(mesh))

class Noise2SelfDataset(Dataset):
    def __init__(self, volume_addr, noised_part=0.5, value_replacer='interpolation_both', noise_position='regular', masked_output=False, augmentation=None, scale_on_load=None):
        if isinstance(volume_addr, str):
            self.volume = imread(volume_addr)
        else:
            self.volume = volume_addr
        self.augmentation = augmentation

        self.noised_part = noised_part
        self.value_replacer = value_replacer
        self.noise_position = noise_position
        self.masked_output = masked_output
        
        if scale_on_load is not None:
            self.volume *= scale_on_load
        
    def __len__(self):
        return len(self.volume)
    
    def __getitem__(self, i):
        inp = deepcopy(self.volume[i])
        out = deepcopy(self.volume[i])


        if self.noise_position == 'regular':
            mask = regular_grid(inp.shape, self.noised_part)
        elif self.noise_position == 'random':
            mask = np.random.random(inp.shape) < self.noised_part

        if self.value_replacer == 'interpolation_both':
            inp[mask] = 0
            out[~mask] = 0
            filtered_inp = convolve2d(inp, footmask, mode='same')
            filtered_out = convolve2d(out, footmask, mode='same')
            inp[mask] = filtered_inp[mask]
            out[~mask] = filtered_out[~mask]
        elif self.value_replacer == 'interpolation_input':
            inp[mask] = 0
            filtered_inp = convolve2d(inp, footmask, mode='same')
            inp[mask] = filtered_inp[mask]
        elif self.value_replacer == 'random':
            inp[mask] = np.random.choice(out.flatten(), mask.sum())
        
        if self.masked_output:
            out[~mask] = 0
        
        inp = inp[None, ...]
        out = out[None, ...]

        if self.augmentation is not None:
            mul = np.concatenate([inp, out])
            augmented = self.augmentation(image=np.moveaxis(mul, 0, -1))
            mul = np.moveaxis(augmented['image'], -1, 0)
            inp = mul[:inp.shape[0]]
            out = mul[inp.shape[0]:]
        
        return inp, out

def get_n2s_denoising_datasets(data_adresses, aug=None,  **kwargs):
    datasets = [Noise2SelfDataset(da[0], augmentation=aug, **kwargs) for da in data_adresses]
    return ConcatDataset(datasets)


class Noise2NoiseMultiTargetDataset(Dataset):
    def __init__(self, source_address, target_addresses, augmentation=None, lazy=False):
        if isinstance(source_address, str):
            self.source_volume = imread(source_address, lazy=lazy)
        else:
            self.source_volume = source_address
        self.target_volumes = []
        for target_address in target_addresses:
            if isinstance(target_address, str):
                self.target_volumes.append(imread(target_address, lazy=lazy))
            else:
                self.target_volumes.append(target_address)
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.source_volume)
    
    def __getitem__(self, i):
        target_id = np.random.randint(len(self.target_volumes))

        inp = deepcopy(self.source_volume[None, i])
        out = deepcopy(self.target_volumes[target_id][None, i])

        if self.augmentation is not None:
            mul = np.concatenate([inp, out])
            augmented = self.augmentation(image=np.moveaxis(mul, 0, -1))
            mul = np.moveaxis(augmented['image'], -1, 0)
            inp = mul[:inp.shape[0]]
            out = mul[inp.shape[0]:]

        return inp, out

def get_n2n_multitarget_denoising_datasets(data_adresses, aug=None, lazy=False):
    datasets = [Noise2NoiseMultiTargetDataset(da[0], da[1:], augmentation=aug, lazy=lazy) for da in data_adresses]
    return ConcatDataset(datasets)


class NoisyPaganinDataset(Dataset):
    def __init__(self, volume_addr, noisy_paganin_address=None, augmentation=None, lazy=False):
        if isinstance(volume_addr, str):
            volume = imread(volume_addr)
        else:
            volume = volume_addr
        
        if noisy_paganin_address:
            paganin = imread(noisy_paganin_address, lazy=True)
            
        
            self.v1 = np.float16(volume + np.pad(paganin[np.random.randint(0, len(volume), len(volume))], ((0, 0), (0, 544-516), (0, 800-787))))
            self.v2 = np.float16(volume + np.pad(paganin[np.random.randint(0, len(volume), len(volume))], ((0, 0), (0, 544-516), (0, 800-787))))
        else:
            # inference mode
            self.v1 = volume
            self.v2 = volume

        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.v1)

    def __getitem__(self, i):
        inp = np.float32(self.v1[None, i])
        out = np.float32(self.v2[None, i])

        if self.augmentation is not None:
            mul = np.concatenate([inp, out])
            augmented = self.augmentation(image=np.moveaxis(mul, 0, -1))
            mul = np.moveaxis(augmented['image'], -1, 0)
            inp = mul[:-1]
            out = mul[[-1]]
        
        return inp, out

def get_noisy_paganin_datasets(data_adresses, aug=None, **kwargs):
    datasets = [NoisyPaganinDataset(da[0], augmentation=aug, **kwargs) for da in data_adresses]
    return ConcatDataset(datasets)
