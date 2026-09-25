"""Generate frame-local Euclidean SSIs for the UWA dataset.

Accept both reader outputs (samples, frames, joints, 3) and legacy flattened
coordinates (samples, frames, joints * 3). Output remains float32 with a final
singleton image channel. These offline SSIs use Euclidean distances; the
learned MetricLayer in the NTU models is a separate implementation.
"""

import os

import h5py
import numpy as np


def load_data(filepath, scale):
    """Load both raw HDF5 splits, closing each file after reading."""
    splits = []
    for part in ('train', 'test'):
        filename = part.capitalize() + '_Raw_cv' + str(scale) + '.h5'
        with h5py.File(os.path.join(filepath, filename), 'r') as file:
            splits.extend([file['x_' + part][:], file['y_' + part][:]])
    return splits


def _joint_sequences(x):
    """Validate and expose a joint axis without guessing ambiguous layouts."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] % 3 == 0:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // 3, 3)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError('Expected (samples, frames, joints, 3) or flattened XYZ coordinates.')
    if x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError('SSI inputs must contain at least one frame and one joint.')
    if not np.isfinite(x).all():
        raise ValueError('SSI inputs must contain finite coordinates.')
    return x


def _compute_ssi(x, y):
    """Compute each frame independently and preserve sample-label alignment."""
    x = _joint_sequences(x)
    y = np.asarray(y)
    if y.ndim == 0 or x.shape[0] != y.shape[0]:
        raise ValueError('Coordinates and labels must have the same sample count.')
    # Preserve UWA's removal of all-zero samples, including the matching labels.
    valid = np.any(x != 0, axis=(1, 2, 3))
    x, y = x[valid], y[valid]
    images = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[2], 1), dtype='float32')
    for sample, sequence in enumerate(x):
        for t, joints in enumerate(sequence):
            # Allocate fresh distances per frame: no previous frame/sample state.
            joints = np.asarray(joints, dtype='float64')
            offsets = joints[:, None, :] - joints[None, :, :]
            distances = np.sqrt(np.sum(offsets * offsets, axis=-1))
            # UWA first standardises each frame, then normalises the whole sequence.
            images[sample, t, :, :, 0] = (distances - distances.mean()) / (distances.std() + 1e-8)
        volume = images[sample, :, :, :, 0]
        images[sample, :, :, :, 0] = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return images, y


def calculate_ssm(filepath, scale):
    """Convert train/test raw files into the existing SSI HDF5 schema."""
    # Process one split at a time to avoid holding both dense inputs in memory.
    for part in ('train', 'test'):
        raw_name = part.capitalize() + '_Raw_cv' + str(scale) + '.h5'
        with h5py.File(os.path.join(filepath, raw_name), 'r') as file:
            x = file['x_' + part][:]
            y = file['y_' + part][:]
        images, labels = _compute_ssi(x, y)
        output_name = part.capitalize() + 'set' + str(scale) + '.h5'
        with h5py.File(os.path.join(filepath, output_name), 'w') as file:
            file.create_dataset('X_' + part, data=images)
            file.create_dataset('Y_' + part, data=labels)
        del x, y, images, labels


if __name__ == '__main__':
    filepath = '/home/data/UWA3DII/'
    joint_num = 15

    print('computing ssm images....')
    calculate_ssm(filepath, scale='1')
    calculate_ssm(filepath, scale='2')
    calculate_ssm(filepath, scale='3')
