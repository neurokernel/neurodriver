#!/usr/bin/env python

"""
Routines for reading/writing numpy arrays from/to HDF5 files.
"""

import numbers

import h5py
import numpy as np

def dataset_append(dataset, arr):
    """
    Append an array to an h5py dataset.

    Parameters
    ----------
    dataset : h5py.Dataset
        Dataset to extend. Must be resizable in its first dimension.
    arr : numpy.ndarray
        Array to append. All dimensions of `arr` other than the first 
        dimension must be the same as those of the dataset.        
    """
    
    assert isinstance(dataset, h5py.Dataset)
    assert isinstance(arr, np.ndarray)

    # Save leading dimension of stored array:
    maxshape = list(dataset.shape)
    old_ld_dim = maxshape[0]

    # Extend leading dimension of stored array to accommodate new array:
    maxshape[0] += arr.shape[0]
    dataset.resize(maxshape)

    # Compute slices to use when assigning `arr` to array extension:
    slices = [slice(old_ld_dim, None)]
    for s in maxshape[1:]:
        slices.append(slice(None, None))

    # Convert list of slices to tuple because __setitem__ can 
    # only handle "simple" indexes:
    slices = tuple(slices)
    dataset.__setitem__(slices, arr)        

def write_array(arr, filename, mode='w', complevel=0):
    """
    Write numpy array containing numerical data to HDF5 file.

    Parameters
    ----------
    arr : numpy.ndarray, pycuda.gpuarray.GPUArray, parray.PitchArray
        Array to store. Must contain numerical data.
    filename: str
        HDF5 file to write.
    mode : str
        Mode to use when opening file. 'w' creates a new file,
        'a' appends to the dataset in an existing file. If appending, 
        all dimensions of `arr` other than the first dimension must be the same        
        as those of the existing dataset.
    complevel : int
        Compression level. Must be between 0 and 10.

    Notes
    -----
    Files written with this routine can be opened in MATLAB using
    the `h5read` function (although complex values will be returned
    as a structure containing a real array and imaginary array).

    See Also
    --------
    read_array
    """

    if arr.__class__.__name__ in ['GPUArray', 'PitchArray']:
        arr = arr.get()
    elif not isinstance(arr, np.ndarray):
        TypeError('unsupported array type')
    if not issubclass(arr.dtype.type, numbers.Number):
        TypeError('unsupported array dtype')

    h5file = h5py.File(filename, mode)    
    if mode == 'w' or (mode == 'a' and 'array' not in h5file.keys()):
        # Set leading dimension to None to enable the created array to be 
        # resized:
        maxshape = list(arr.shape)
        maxshape[0] = None
        h5file.create_dataset('/array', data=arr, maxshape=maxshape,
                              compression=complevel)
    elif mode == 'a':
        dataset_append(h5file['/array'], arr)
    else:
        RuntimeError('invalid mode')

    h5file.close()

def read_array(filename):
    """
    Read numpy array containing numerical data from HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file to read.

    Returns
    -------
    a : numpy.array
        Array read from file.

    Notes
    -----
    Files written with `write_array` or MATLAB's `h5write` function can
    be read with this routine.

    See Also
    --------
    write_array
    """

    h5file = h5py.File(filename, 'r')
    result = h5file['/array'][:]
    h5file.close()
    return result
