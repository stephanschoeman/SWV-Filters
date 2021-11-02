def firstToPeakSymmetrical(x, y, dir = -1):
    yMod = []
    yMiddle = 0
    if dir < 0:
        yMiddle = min(y)
    else:
        yMiddle = max(y)

    index = y.index(yMiddle) * 2
    # y = mx + c
    # m = (y2 - y1)/(x2 - x1)

    if index > len(y) - 1:
        index = len(y) - 1

    m = (y[index] - y[0])/(x[index] - x[0])
    c = y[index] - x[index]*m


    pos = 0
    for v in y:
        yMod.append(v - m*x[pos] - c)
        pos += 1

    return [x, yMod]

def firstToPeakSymmetricalWithPercent(x, y, p = 10, dir = -1):
    yMod = []
    yMiddle = 0
    if dir < 0:
        yMiddle = min(y)
    else:
        yMiddle = max(y)

    index = y.index(yMiddle) * 2
    # y = mx + c
    # m = (y2 - y1)/(x2 - x1)

    if index > len(y) - 1:
        index = len(y) - 1

    startIndex = returnPercentIndex(y, p, 'right')

    m = (y[index] - y[startIndex])/(x[index] - x[startIndex])
    c = y[index] - x[index]*m

    pos = 0
    for v in y:
        yMod.append(v - m*x[pos] - c)
        pos += 1

    return [x, yMod]

def returnPercentIndex(y, p = 10, dir = 'right'):
    index = 0
    yMod = []

    if dir == 'right':
        yMod = y
    else:
        for yval in reversed(y):
            yMod.append(yval)

    mx = len(yMod) - 1

    for yCurrent in yMod:
        if index < mx:
            pChange = abs(100 - (yMod[index + 1]/yCurrent) * 100)
            if(pChange < p):
                break
        index += 1

    if(index >= mx or index > 50):
        print('Gradient too large for dir: ' + dir + ' at a ' + str(p) + ' slope')
        index = 0

    return index

def leftAndRightMinimum(x, y, dir = -1):
    yMod = []

    minimumIndex = y.index(min(y))

    if minimumIndex > 0.1*len(y):
        firstHalf = y[:minimumIndex]
        secondHalf = y[minimumIndex:]

        indexRight = y.index(max(firstHalf)) 
        indexLeft = y.index(max(secondHalf)) 
    else:
        indexRight = 0
        indexLeft = len(y) - 1

    # y = mx + c
    # m = (y2 - y1)/(x2 - x1)

    m = (y[indexLeft] - y[indexRight])/(x[indexLeft] - x[indexRight])
    c = y[indexLeft] - x[indexLeft]*m

    pos = 0
    for v in x:
        yMod.append(y[pos] - m*v - c)
        pos += 1

    return [x, yMod, m, c]


def thirdPointsOrLRMin(x, y, dir = -1):
    yMod = []
    xM = leftAndRightMinimum(x, y)
    yMod = xM[1]

    minimumIndex = yMod.index(min(yMod))

    if minimumIndex > 0.2*len(y) and minimumIndex < 0.8*len(y):
        firstHalf = yMod[:minimumIndex]
        secondHalf = yMod[minimumIndex:]

        try:
            maxFirst = firstHalf.index(0)
        except:
            maxFirst = getCloseToZero(firstHalf)

        try:
            maxSecond = secondHalf.index(0)
        except:
            maxSecond = getCloseToZero(secondHalf)

        if maxFirst < 3:
            firstNewIndex = 3
        else:
            firstNewIndex = maxFirst

        if maxSecond + len(firstHalf) > (len(yMod) - 3):
            secondNewIndex = len(yMod) - 3
        else:
            secondNewIndex = maxSecond + len(firstHalf) 
    else:
        firstNewIndex = 0
        secondNewIndex = len(y) - 1

    # y = mx + c
    # m = (y2 - y1)/(x2 - x1)

    try:
        m = (yMod[secondNewIndex] - yMod[firstNewIndex])/(x[secondNewIndex] - x[firstNewIndex])
        c = yMod[secondNewIndex] - x[secondNewIndex]*m

        yModMod = []
        pos = 0
        for v in x:
            yModMod.append(yMod[pos] - m*v - c)
            pos += 1
    except:
        print('ERROR: ' + str(secondNewIndex) + ' ' + str(firstNewIndex) + ' ' + str(len(yMod)))
        return [x, yMod]


    return [x, yModMod]

def getCloseToZero(y):
    print('Using approximate zero')
    pos = 0
    for yVal in y:
        if pos < 0.5*len(y):
            if yVal >= 0 and y[pos + 1] <= 0:
                return yVal
        pos += 1
    print('Returning zero')

    return 0

def getPlainMinXY(x, y):
    xMin = 0
    yMin = 0

    p0V = 0
    p3V = 0

    pos = 0
    for xV in x:
        if xV < 0 and p0V == 0:
            p0V = pos
        if p3V < 0.3 and p3V == 0:
            p3V = pos

        pos += 1

    yMin = min(y[p3V:p0V])
    index = y.index(yMin)
    xMin = x[index]

    return [xMin, yMin]

def getGradient(x, y):
    try:
        return (y[len(y) - 1] - y[0])/(x[len(x) - 1] - x[0])
    except:
        return 0


def touchBaselineLeftToRight(x, y, softFilter = 1):

    yMod = []
    A = returnPercentIndex(y)
    rightIndex = hardDerivativeChangeIndex(x[A:],y[A:])
    rightIndex = rightIndex + A
    middleIndex = y.index(min(y))
    leftArray = y[middleIndex:]
    leftIndex = y.index(max(leftArray))
    pos = 0
    mx = leftIndex - middleIndex - softFilter

    for yval in reversed(y[middleIndex:leftIndex]):
        realIndex = y.index(yval)
        if pos < mx:
            mReal = getGradient([x[realIndex], x[realIndex + softFilter]],[y[realIndex], y[realIndex + softFilter]])
            mLine = getGradient([x[realIndex], x[rightIndex]],[y[realIndex], y[rightIndex]])
            if mReal * mLine > 0:
                if abs(mReal) > abs(mLine):
                    break
            leftIndex = realIndex
        pos += 1

    mLine = getGradient([x[leftIndex], x[rightIndex]],[y[leftIndex], y[rightIndex]])
    cLine = y[leftIndex] - mLine*x[leftIndex]

    yMod = []
    pos = 0
    for v in x:
        yMod.append(y[pos] - mLine*v - cLine)
        pos += 1


    return [x, yMod]

def touchBaselineRightToLeft(x, y, softFilter = 1):

    yMod = []
    A = returnPercentIndex(y)
    rightIndex = hardDerivativeChangeIndex(x[A:],y[A:])
    rightIndex = rightIndex + A
    middleIndex = y.index(min(y))
    leftArray = y[middleIndex:]
    leftIndex = y.index(max(leftArray))
    pos = 0
    mx = leftIndex - middleIndex - softFilter

    for yval in y[middleIndex:leftIndex]:
        realIndex = y.index(yval)
        mReal = getGradient([x[realIndex], x[realIndex - softFilter]],[y[realIndex], y[realIndex - softFilter]])
        if pos < mx and mReal < 0:
            mLine = getGradient([x[realIndex], x[rightIndex]],[y[realIndex], y[rightIndex]])
            if mReal * mLine > 0:
                if abs(mReal) < abs(mLine):
                    leftIndex = realIndex
                    break
            leftIndex = realIndex
        pos += 1

    mLine = getGradient([x[leftIndex], x[rightIndex]],[y[leftIndex], y[rightIndex]])
    cLine = y[leftIndex] - mLine*x[leftIndex]

    yMod = []
    pos = 0
    for v in x:
        yMod.append(y[pos] - mLine*v - cLine)
        pos += 1

    return [x, yMod]

def hardDerivativeChangeIndex(x, y, softSet = 1, consecutivePoints = 4):
    pos = 0
    savePos = 0
    successCount = 0
    mx = len(y) - 2*softSet
    for yVal in y:
        if pos < mx:
            mCur = getGradient([x[pos],x[pos + softSet]],[y[pos], y[pos + softSet]])
            mFut = getGradient([x[pos + softSet],x[pos + 2*softSet]],[y[pos + softSet], y[pos + 2*softSet]])
            if mCur*mFut > 0:
                if successCount == 0:
                    savePos = pos
                if successCount > consecutivePoints:
                    break
                successCount += 1
            else:
                successCount = 0
                savePos = 0
        pos += 1
    
    return savePos


def parse(x, y):
    yMin = getYMinimumBasedOnMaximum(y)
    yMod = modulatingSG(yMin)
    return baseline(x, yMod, yMin)
    
def baseline(x, y0, yOr, softFilter = 1):
    y = []
    for val in y0:
        y.append(val)
 
    yMod = []
    rightIndex = 3# getRightPositionBasedOnSparsity(y)
    #print(rightIndex)
    #print(len(y))
    middleIndex = y.index(min(y))
    leftArray = y[middleIndex:]
    leftIndex = y.index(max(leftArray))
    pos = 0
    mx = leftIndex - middleIndex - softFilter

    for yval in y[middleIndex:leftIndex]:
        realIndex = y.index(yval)
        mReal = getGradient([x[realIndex], x[realIndex - softFilter]],[y[realIndex], y[realIndex - softFilter]])
        if pos < mx and mReal < 0:
            mLine = getGradient([x[realIndex], x[rightIndex]],[y[realIndex], y[rightIndex]])
            if mReal * mLine > 0:
                if abs(mReal) < abs(mLine):
                    leftIndex = realIndex
                    break
            leftIndex = realIndex
        pos += 1

    mLine = getGradient([x[leftIndex], x[rightIndex]],[y[leftIndex], y[rightIndex]])
    cLine = y[leftIndex] - mLine*x[leftIndex]

    yMod = []
    pos = 0
    for v in x:
        yMod.append(yOr[pos] - mLine*v - cLine)
        pos += 1

    return [x, yMod]


def getRightPositionBasedOnSparsity(y, factor = 3):

    sp = getSparsityFactor(y)
    print(sp)
    pos = 1
    mx = len(y) - 2
    safetyFactor = 0
    saveThisPosition = 0

    for val in reversed(y):
        if pos < mx:
            localDiff = abs(val - y[pos])
            if localDiff < factor*sp:
                if safetyFactor == 0:
                    saveThisPosition = pos - 1
                
                if safetyFactor == 3:
                    break

                safetyFactor += 1
            else:
                safetyFactor = 0
                
        pos += 1

    return len(y) - 1 - saveThisPosition


def getSparsityFactor(y):
    pos = 1
    mx = len(y) - 2
    sp = 0
    for val in y:
        if pos < mx:
            diff = abs(val - y[pos])/2
            if pos == 1:
                sp = diff
            else:
                sp += (diff)
        pos += 1

    if pos == 1:
        print('Could not calculate the sparsity factor for array:')
        print(y)

    return sp

def getYMinimumBasedOnMaximum(y):
    mx = max(y)
    #print(mx)
    yMod = []
    for val in y:
        yMod.append(val - mx)
    return yMod

def modulatingSG(y):
    winSize = round(len(y)/10)

    if winSize % 2 == 0:
        winSize += 1

    return savitzky_golay(y, window_size=21, order=1)


def savitzky_golay(y0, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    y = np.asarray(y0)
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')