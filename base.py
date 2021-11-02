from filters import returnPercentIndex
import numpy as np
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt

def baseline(x, y, fs, order = 2, cutoff=1, addPlot=False):
    my = butter_lowpass_filter(y, cutoff, fs, order)
    return touchBaselineRightToLeft(x, y, my,addPlot=addPlot)

def getRightStartingPosition(x, y, my, addPlot=False):
    SelectedTPPos = RightTP(x, y, my, addPlot=addPlot)

    if SelectedTPPos == 0:
        # check that startup current is not present
        SelectedTPPos = returnPercentIndex(y)

    if SelectedTPPos < 2 or SelectedTPPos > 20:
        # if we selected a wacky position, make sure we select the maximum in this range
        SelectedTPPos = RightMaximum(x, y, SelectedTPPos)
    
    if SelectedTPPos == 0:
        # look for saddle
        SelectedTPPos = RightTP(x, y, my, 2, addPlot=addPlot)

   # print(SelectedTPPos)

    return SelectedTPPos

def RightMaximum(x, y, SelectedTPPos):

    V0p35 = 0
    pos = 0
    for xV in x:
        if xV < 0.3 and V0p35 == 0:
            V0p35 = pos
        pos += 1
    yMaxPos = y.index(max(y[1:V0p35]))

    if SelectedTPPos > 20:
        SelectedTPPos = 0

    if yMaxPos > SelectedTPPos:
        #print('Returning right maximum for pos: ' + str(yMaxPos))
        return yMaxPos

    return SelectedTPPos

def touchBaselineRightToLeft(x, y, my, softFilter = 1,addPlot=False):

    yMod = []
    rightIndex = getRightStartingPosition(x, y, my,addPlot=addPlot)

    # strong assumption that we have a minimum
    middleIndex = my.index(min(my))
    leftArray = my[middleIndex:]
    leftIndex = my.index(max(leftArray))
    pos = 0
    mx = len(my) - 1
    maxIndex = leftIndex
    #print('start')
    #print(mx)

    for yval in my[middleIndex:mx]:
        realIndex = my.index(yval)
        mReal = getGradient([x[realIndex], x[realIndex - softFilter]],[my[realIndex], my[realIndex - softFilter]])
        #print(str(mReal) + ' > ')
        if pos < mx and mReal < 0:
            mLine = getGradient([x[realIndex], x[rightIndex]],[my[realIndex], my[rightIndex]])
            #print(str(mLine))
            if mReal * mLine > 0:
                if abs(mReal) < abs(mLine) or realIndex == maxIndex:
                    leftIndex = realIndex
                    #print('leftIndex')
                    break
            leftIndex = realIndex
        pos += 1
        #print(pos)

    # check for a max between 0 and -0.2 V for leftside
    mxYindex = y.index(max(y[middleIndex:mx]))
    if x[mxYindex] < 0 and x[mxYindex] > -0.2:
        leftIndex = mxYindex

    mLine = getGradient([x[leftIndex], x[rightIndex]],[y[leftIndex], y[rightIndex]])
    cLine = y[leftIndex] - mLine*x[leftIndex]

    yMod = []
    pos = 0
    for v in x:
        yMod.append(y[pos] - mLine*v - cLine)
        pos += 1

    if addPlot:
        aa =[]
        for val in x:
            aa.append(mLine*val + cLine)

        plt.figure(figsize=(12,6),dpi=120)

        plt.plot(x, aa)
        plt.plot(x, y)
        plt.plot(x[leftIndex],y[leftIndex],'*')
        plt.plot(x[maxIndex],y[maxIndex],'.')
        plt.grid()

        return [x, yMod, mLine, cLine]

    return [x, yMod]

def getGradient(x, y):
    try:
        return (y[len(y) - 1] - y[0])/(x[len(x) - 1] - x[0])
    except:
        return 0

def returnPercentIndex(y, p = 20, dir = 'right'):
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
                #if index > 2:
                #    print('Returning percentage change at pos: ' + str(index) + '. Change: ' + '{:.2f}'.format(pChange) + '% < ' + str(p) + '%')
                break
        index += 1

    if(index >= mx or index > 50):
        print('Gradient too large for dir: ' + dir + ' at a ' + str(p) + ' slope')
        index = 0

    return index


def RightTP(dx, dy, my, df = 1, addPlot=False):
    CUTOFF_VOLTAGE = 0.35
    pos = 0
    dmy = my

    while pos < df:
        dmy = np.diff(dmy)
        pos += 1

    pos = 1
    mx = len(dmy) - 1
    turningPoints = []
    for val in dmy:
        if pos < mx:
            nxtVal = dmy[pos]
            if nxtVal * val < 0:
                turningPoints.append(pos)
        pos += 1
    # we might have more than one turning point
    # look for the one that looks the most promising between 0.3 and 0.4 V
    SelectedTPPos = 0
    # we also know the SelectedTPPos can in no way stay 0
    if len(turningPoints) > 1:
        prevTP = 0
        firstRun = True
        for val in turningPoints:
            if dx[val] >= CUTOFF_VOLTAGE:
                # we have a TP that we can select
                curTP = dy[val]
                if firstRun:
                    firstRun = False
                    prevTP = curTP
                    SelectedTPPos = val
                else:
                    if curTP > prevTP:
                        #print(SelectedTPPos)
                        prevTP = curTP
                        SelectedTPPos = val

    # get all the turning points
    if addPlot:
        plt.plot(dx[0:len(dmy)],dmy)
        plt.plot(dx[SelectedTPPos],dmy[SelectedTPPos],'*')
        if df == 1:
            plt.title(r'First differential')
        else:
            plt.title(r'Second differential')

        plt.grid()
        plt.show()

    return SelectedTPPos


def butter_lowpass_filter(data, cutoff, fs, order,addPlot=False):
    from scipy.signal import butter,filtfilt
    from scipy import signal
    
    nyq = 0.5 * fs  # Nyquist Frequency

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=True)
    y = filtfilt(b, a, data)

    if addPlot:
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        #plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(normal_cutoff, color='green') # cutoff frequency
        plt.xlim(normal_cutoff/10,normal_cutoff*10)
        plt.show()

    yMod = []
    for val in y:
        yMod.append(val)

    return yMod