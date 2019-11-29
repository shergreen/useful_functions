import numpy as np

def logbins(bin_min, bin_max, nbins):
    """
    Generates logarithmically-spaced bins given a min, max and bin count.
    Also returns an empty array for binning.

    TODO: Add functionality so that the given array is also binned accordingly
    """
    bin_edges = np.logspace(np.log10(bin_min), np.log10(bin_max), nbins+1)
    bin_centers = 10**((np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])) / 2.)
    bins = np.zeros(nbins)
    return bin_edges, bin_centers, bins

def autobin(bin_min, bin_max, nbins, arr):
    """
    Same as logbins, but automatically bins the values in arr into the bins return.
    This routine makes no assumption about the values of arr, so one needs that the
    requested bins cover the range of interest.

    It's possible that NumPy has a functionality that already does something like
    this, but I've written this code snippet enough times to make it a function.
    """
    bin_edges, bin_centers, bins = logbins(bin_min, bin_max, nbins)
    for i in range(0, nbins):
        bins[i] = np.sum(np.logical_and(arr >= bin_edges[i], arr <= bin_edges[i+1]))
    return bin_edges, bin_centers, bins