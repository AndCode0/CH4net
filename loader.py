import os
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torch


class BalancedPlumeSampler(torch.utils.data.Sampler):
    """
    Sampler that balances positive and negative samples each epoch
    
    Args:
        pos_indices: Indices of positive samples
        neg_indices: Indices of negative samples  
        mode: "train", "val", or "test"
        neg_ratio: Ratio of negative to positive samples (e.g., 2.0 means 2 negatives per positive)
        generator: Random generator for reproducibility
    """

    def __init__(self, pos_indices, neg_indices, mode="train", neg_ratio=1.0, generator=None):
        self.pos_indices = torch.tensor(pos_indices) if isinstance(pos_indices, list) else pos_indices
        self.neg_indices = torch.tensor(neg_indices) if isinstance(neg_indices, list) else neg_indices
        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        self.neg_ratio = neg_ratio
        self.mode = mode
        self.generator = generator

        if self.mode in ["test", "val"]:
            self._length = self.n_pos + self.n_neg
            # Pre-compute deterministic indices for test/val
            indices = torch.cat([self.pos_indices, self.neg_indices])
            perm = torch.randperm(len(indices), generator=self.generator)
            self._cached_indices = indices[perm].tolist()
        else:
            n_neg_to_sample = min(self.n_neg, int(self.n_pos * self.neg_ratio))
            self._length = self.n_pos + n_neg_to_sample

    def __iter__(self):
        if self.mode in ["test", "val"]:
             return iter(self._cached_indices)

        # For training: balance and shuffle
        # Calculate how many negatives to sample
        n_neg_to_sample = int(self.n_pos * self.neg_ratio)
        
        # Randomly sample negatives
        if self.n_neg >= n_neg_to_sample:
            neg_perm = torch.randperm(self.n_neg, generator=self.generator)[:n_neg_to_sample]
            sampled_neg = self.neg_indices[neg_perm]
        else:
            # If not enough negatives, use all of them
            sampled_neg = self.neg_indices
        
        # Combine and shuffle
        indices = torch.cat([self.pos_indices, sampled_neg])
        shuffle_perm = torch.randperm(len(indices), generator=self.generator)
        
        return iter(indices[shuffle_perm].tolist())

    def __len__(self):
        return self._length


class MethaneLoader(Dataset):

    def __init__(self, mode, path, base_seed, red=False, channels=12, neg_ratio=1.0, generator=None):
        self.mode = mode
        self.reduce = red
        self.channels = channels
        self.neg_ratio = neg_ratio
        self.base_seed = base_seed
        self.generator = generator
        self.per_worker_generators = {}  # Stores per-worker RNG states
        
        if not os.path.exists(path):
            raise FileNotFoundError(f'The specified path does not exist: {path}')
        
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        label_dir = os.path.join(path,mode,"label")
        image_dir = os.path.join(path,mode,"s2")

        if not os.path.exists(label_dir):
            raise FileNotFoundError(f'Label directory does not exist: {label_dir}')
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f'Image directory does not exist: {image_dir}')

        self.image_dir = image_dir

        label_files = sorted(glob(f"{label_dir}/*.npy"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not label_files:
            raise ValueError(f"No label files found in {label_dir}")

        self.labels = label_files
        self.pos_indices = []
        self.neg_indices = []

        for idx, lf in enumerate(label_files):
            label = np.load(lf)
            if label.sum() > 0:
                self.pos_indices.append(idx)
            else:
                self.neg_indices.append(idx)

        self.s_ = 50

    def get_sampler(self):
        """Return the custom sampler for this dataset"""
        return BalancedPlumeSampler(
            self.pos_indices, 
            self.neg_indices, 
            self.mode, 
            neg_ratio=self.neg_ratio,
            generator=self.generator
        )

    def __len__(self):
        return len(self.labels)
    
    def _get_generator(self):
        worker_info = torch.utils.data.get_worker_info()

        # Set up RNG: per-worker if in multi-process, else use instance generator
        if worker_info is not None:  # Multi-worker mode
            worker_id = worker_info.id
            if worker_id not in self.per_worker_generators:
                seed = self.base_seed + worker_id
                self.per_worker_generators[worker_id] = torch.Generator().manual_seed(seed)
            return self.per_worker_generators[worker_id]
        else:  # Single-process mode
            return self.generator if self.generator is not None else torch.Generator()
        
    def _select_channels(self, image):
        if self.channels == 12:
            return image
        elif self.channels == 2:
            return image[..., 10:12]
        elif self.channels == 5:
            return np.concatenate([image[..., 1:4], image[..., 10:12]], axis=-1)
        else:
            raise ValueError(f'Channels can be 2, 5 or 12, not {self.channels}')

    def __getitem__(self, index):
 
        generator = self._get_generator()

        label_path = self.labels[index]
        file_id = os.path.basename(label_path).replace(".npy", "")
        image_path = os.path.join(self.image_dir, f"{file_id}.npy")

        target = np.load(label_path)
        context = np.load(image_path)

        height, width = target.shape

        if self.mode == "train":
            mid_x = torch.randint(self.s_, width - self.s_, (1,), generator=generator).item()
            mid_y = torch.randint(self.s_, height - self.s_, (1,), generator=generator).item()

        else:
            mid_x = width // 2
            mid_y = height // 2

        x_slice, y_slice = slice(mid_x - self.s_, mid_x + self.s_), slice(
            mid_y - self.s_, mid_y + self.s_
        )

        target = target[y_slice, x_slice]
        context = context[y_slice, x_slice, :]

        context = self._select_channels(context)

        if self.reduce:
            target = np.array([int(target.any())])

        d = {
            "pred": torch.from_numpy(context).float().permute(2, 0, 1) / 255,
            "target": torch.from_numpy(target).float(),
        }
        return d



# class MethaneHDF5Loader(Dataset):
#     def __init__(self, mode, channels=12, crop_size=50, 
#                  generator=None, use_mbmp=False):
#         self.hdf5_path = "C:/Users/aconte/Desktop/AIRMO/data.h5"
#         self.mode = mode
#         self.channels = channels
#         self.crop_size = crop_size
#         self.generator = generator
#         self.use_mbmp = use_mbmp
        
#         self.file_handle = None
#         self.group = None
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             grp = f[mode]
#             self.n_samples = grp.attrs['n_samples']
#             self.img_shape = grp.attrs['target_image_shape']
#             self.label_shape = grp.attrs['target_label_shape']
            
#             # Scan for positive/negative indices
#             labels_data = grp['labels']
#             self.pos_indices = []
#             self.neg_indices = []
            
#             chunk_size = 1000
#             for i in range(0, self.n_samples, chunk_size):
#                 end_idx = min(i + chunk_size, self.n_samples)
#                 labels_chunk = labels_data[i:end_idx]
                
#                 for j, label in enumerate(labels_chunk):
#                     global_idx = i + j
#                     if label.sum() > 0:
#                         self.pos_indices.append(global_idx)
#                     else:
#                         self.neg_indices.append(global_idx)
    
#     def _init_file_handle(self):
#         """Initialize file handle - called in worker process"""
#         if self.file_handle is None:
#             self.file_handle = h5py.File(self.hdf5_path, 'r', swmr=True)
#             self.group = self.file_handle[self.mode]
    
#     def __getitem__(self, index):
#         # Initialize file handle in worker process if needed
#         self._init_file_handle()
        
#         H, W = self.label_shape[:2]
        
#         # Calculate crop coordinates
#         if self.mode == "train":
#             # Use numpy random state for reproducibility
#             rng = np.random.RandomState()
#             mid_x = rng.randint(self.crop_size, W - self.crop_size)
#             mid_y = rng.randint(self.crop_size, H - self.crop_size)
#         else:
#             mid_x = W // 2
#             mid_y = H // 2
        
#         # Define slices for partial loading
#         y_slice = slice(mid_y - self.crop_size, mid_y + self.crop_size)
#         x_slice = slice(mid_x - self.crop_size, mid_x + self.crop_size)
        
#         # Efficient partial loading with HDF5 slicing
#         label = self.group['labels'][index, y_slice, x_slice]
        
#         # Channel selection based on configuration
#         if self.channels == 2:
#             image = self.group['images'][index, y_slice, x_slice, 10:12]
#         elif self.channels == 5:
#             # Advanced indexing for non-contiguous channels
#             image = self.group['images'][index, y_slice, x_slice, :]
#             image = image[..., [1, 2, 3, 10, 11]]
#         else:
#             image = self.group['images'][index, y_slice, x_slice, :]
        
#         # Optional: load MBMP data
#         if self.use_mbmp:
#             mbmp = self.group['mbmp'][index, y_slice, x_slice, :]
        
#         # Ensure arrays are contiguous for torch
#         image = np.ascontiguousarray(image)
#         label = np.ascontiguousarray(label)
        
#         output = {
#             'pred': torch.from_numpy(image).float().permute(2, 0, 1) / 255,
#             'target': torch.from_numpy(label).float(),
#         }
        
#         if self.use_mbmp:
#             output['mbmp'] = torch.from_numpy(mbmp).float().permute(2, 0, 1)
        
#         return output
    
#     def __len__(self):
#         return self.n_samples
    
#     def get_sampler(self):
#         """Get the balanced sampler"""
#         return BalancedPlumeSampler(
#             self.pos_indices,
#             self.neg_indices,
#             self.mode,
#             generator=self.generator
#         )
    
#     def __del__(self):
#         """Cleanup file handles"""
#         if self.file_handle is not None:
#             try:
#                 self.file_handle.close()
#             except:
#                 pass  # Ignore errors during cleanup