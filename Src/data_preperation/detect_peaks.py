"""Detect peaks in data based on their amplitude and other features."""

from __future__ import division, print_function
import numpy as np
import math

def detect_peaks(x, mph=None, mpd=0, threshold=None, edge='both',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    if threshold is None:
        threshold = (np.max(x) - np.min(x)) / 4

    # find indices of all peaks
    dx = np.diff(x)
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    dx[dx == 0] = -1e-6
    ind = np.where(dx[:-1] * dx[1:] < 0)[0] + 1 # Find where the derivative changes sign
    ind_vals = x[ind]
    '''
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    '''
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # go through peaks to check if they are maxima
    maxPeaks = math.ceil(len(ind) / 2)
    peak_values = x[ind]
    peakLoc = np.zeros((maxPeaks, 1))
    peakMag = np.zeros((maxPeaks, 1))
    leftMin = min(x[0], peak_values[0])
    minMag = np.min(peak_values)
    tempMag = minMag
    foundPeak = False

    # Skip the first point if it is smaller so we always start on a maxima
    ii = 0 if peak_values[0] > peak_values[1] else 1
    cInd = 0
    while ii < len(peak_values):
        # Reset peak finding if we had a peak and the next peak is bigger
        #   than the last or the left mipeak_valuesn was small enough to reset.
        if foundPeak:
            tempMag = minMag
            foundPeak = False

        # Make sure we don't iterate past the length of our vector
        if ii == len(ind) - 1:
            break  # We assign the last point differently out of the loop

        # Found new peak that was lager than temp mag and selectivity larger
        #   than the minimum to its left.
        if (peak_values[ii] > tempMag) and (peak_values[ii] > (leftMin + threshold)):
            tempLoc = ii
            tempMag = peak_values[ii]

        ii = ii+1 # Move onto the valley
        # Come down at least threshold from peak
        if (not foundPeak) and (tempMag > threshold + peak_values[ii]):
            foundPeak = True # We have found a peak
            leftMin = peak_values[ii]
            peakLoc[cInd] = tempLoc # Add peak to index
            peakMag[cInd] = tempMag
            cInd = cInd+1
        elif peak_values[ii] < leftMin: # New left minima
            leftMin = peak_values[ii]
        ii = ii + 1  # This is a peak
    # Check end point
    if peak_values[-1] > tempMag and (peak_values[-1] > (leftMin + threshold)):
        peakLoc[cInd] = len(peak_values) - 1
        peakMag[cInd] = peak_values[-1]
        cInd = cInd + 1
    elif (not foundPeak) and (tempMag > min(peak_values[-1], x[-1]) + threshold):
        # Check if we still need to add the last point
        peakLoc[cInd] = tempLoc
        peakMag[cInd] = tempMag
        cInd = cInd + 1

    peakLoc = peakLoc[:cInd]
    peakMag = peakMag[:cInd]

    # remove peaks < minimum peak height
    if mph is not None:
        peakLoc = peakLoc[peakMag >= mph]

    peakLoc = ind[peakLoc.astype(int)]
    '''
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    '''
    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, peakLoc)

    return peakLoc


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()