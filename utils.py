import numpy as np


def logbins(bin_min, bin_max, nbins):
    """
    Generates logarithmically-spaced bins given a min, max and bin count.
    Also returns an empty array for binning.

    TODO: Add functionality so that the given array is also binned accordingly
    """
    bin_edges = np.logspace(np.log10(bin_min), np.log10(bin_max), nbins+1)
    bin_centers = 10**((np.log10(bin_edges[1:]) +
                        np.log10(bin_edges[:-1])) / 2.)
    bins = np.zeros(nbins)
    return bin_edges, bin_centers, bins


def autobin(bin_min, bin_max, nbins, bin_arr, calc_arr, typ='std'):
    """
    Generates nbins bins, logarithmically-spaced, in the range bin_min to bin_max.
    Using these bins, then computes the sum/count/or average of the quantities
    in calc_arr corresponding to entries of bin_arr within each bin.

    It's possible that NumPy has a functionality that already does something like
    this, but I've written this code snippet enough times to make it a function.

    TODO: Add functionality for choosing log or linear bins.
    TODO: Get this to work with two arrays: one for the binning and the other for
          the computing
    """
    bin_edges, bin_centers, bins = logbins(bin_min, bin_max, nbins)
    for i in range(0, nbins):
        msk = np.logical_and(
            bin_arr >= bin_edges[i], bin_arr <= bin_edges[i+1])
        if(typ == 'sum'):
            bins[i] = np.sum(calc_arr[msk])
        elif(typ == 'count'):
            bins[i] = np.sum(msk)
        elif(typ == 'average'):
            bins[i] = np.average(calc_arr[msk])
        elif(typ == 'std'):
            bins[i] = np.std(calc_arr[msk])
        elif(typ == '1684'):
            bins[i] = (np.percentile(calc_arr[msk], 84) - np.percentile(calc_arr[msk], 16)) / 2.0
    return bin_edges, bin_centers, bins
