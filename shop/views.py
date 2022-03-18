# shop/views.py
import numpy
import numpy as np
import scipy.ndimage
import scipy.interpolate as interp
import math
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count, F, Sum, Avg
from django.db.models.functions import ExtractYear, ExtractMonth
from django.http import JsonResponse
from django.shortcuts import render

from shop.models import Data
from utils.charts import months, colorPrimary, colorSuccess, generate_color_palette, get_year_dict, get_data_dict
from collections import OrderedDict
import matplotlib.pyplot as plt
import bisect
from scipy.ndimage.filters import uniform_filter1d


@staff_member_required
def get_filter_options(request):
    # grouped_purchases = Purchase.objects.annotate(year=ExtractYear('time')).values('year').order_by('-year').distinct()
    # options = [purchase['year'] for purchase in grouped_purchases]

    return JsonResponse({
        'Disconnected': 'Disconnected',
        'Freq.Syn-Off': 'Freq.Syn-Off',
        'Freq.Syn-818': 'Freq.Syn-818',
        'Freq.Syn-816': 'Freq.Syn-816',
        'Freq.Syn-880': 'Freq.Syn-880',
    })


@staff_member_required
def compute_data(request, year):
    data_dict = get_data_dict()
    data_dict2 = get_data_dict()
    data_dict3 = get_data_dict()
    data_dict4 = get_data_dict()
    data_dict5 = get_data_dict()
    p_dict = get_data_dict()
    p_dict2 = get_data_dict()
    p_dict3 = get_data_dict()
    p_dict4 = get_data_dict()
    p_dict5 = get_data_dict()
    peak_dict = get_data_dict()

    temps = year.split(",")
    valopt = str(temps[3])
    lab = str(temps[4])
    layer = str(temps[5])
    synthopt = str(temps[0])
    peaknum = str(temps[10])  # Penguin12 | Added Line | extract 'em
    current_marker = int(temps[11])


    if valopt == "center":
        center = float(temps[1]) * 1000000
        span = float(temps[2]) * 1000000
        if span <= 0:
            span = 2
        fstart = center - span / 2
        fstop = center + span / 2
    else:
        fstart = float(temps[1])  # * 1000000
        fstop = float(temps[2])  # * 1000000

    # if fstart < 0:
    #    fstart = 0
    maxf = 9999
    minf = 9999

    interpolatefreq = 818000000

    span = fstop - fstart

    if len(lab) > 18:
        templayer = lab
    else:
        templayer=""

    if lab == "L1":
        layer = "RF Output Off"
    elif span >= 2900000000:
        layer = "Layer1"
    elif span >= 1000000000:
        layer = "Layer2"
    elif span >= 500000000:
        layer = "Layer3"
    else:
        layer = "Layer4"

    if len(templayer) > 18:
        layer = templayer.replace("VariableX", layer)
        layer = layer.replace("Z", "|")
        lab = "L5"
    #  year = "Start:" + str(round(fstart / 1000000)) + "MHz,Stop:" + str(round(fstop / 1000000)) + "MHz,Layer:" + layer  # Penguin22 | commented line
    # ,frequency_gte=startf, frequency__lte= stopf 814000000, frequency__lte= 821000000)

    if synthopt == "Disconnected":
        synthoptdb = "Freq.Syn-Off"
    else:
        synthoptdb = synthopt

    # Penguin12 | Added block | Finally (partially) fixed min max by requering but without fstart and fstop to get data_range (still buggy when changing labs for the first time which halves max value, but user side fixed by loading twice, ) {
    """
    rangefilter = Data.objects.filter(lab=lab, c=synthoptdb, r=layer, trace=1)
    temparray = []
    for x in rangefilter:
        temparray.append(x.frequency)
    data_range = [temparray[0], temparray[len(temparray)-1]]
    """
    #Penguin12 }
    print (layer)
    print(fstart)
    print(fstop)


    nestedfilter = Data.objects.filter(lab=lab, c=synthoptdb, r=layer, frequency__gte=fstart, frequency__lte=fstop)
    data_range = [0, 3000000000]  # never give up

    # Penguin | added from here { //Penguin22 | moved/edited block | added else here to avoid reference before assignment{
    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        freq_x_value1 = float(temps[6]) * 1000000
    else:
        freq_x_value1 = (data_range[1] + data_range[0])/2
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        freq_x_value2 = float(temps[7]) * 1000000
    else:
        freq_x_value2 = (data_range[1] + data_range[0])/2
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        freq_x_value3 = float(temps[8]) * 1000000
    else:
        freq_x_value3 = (data_range[1] + data_range[0])/2
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        freq_x_value4 = float(temps[9]) * 1000000
    else:
        freq_x_value4 = (data_range[1] + data_range[0])/2
    # Penguin | to here } //Penguin22 }


    # datavalues = nestedfilter.values_list('frequency', 'amplitude', named=True)

    """
    xaxisdata1 = nestedfilter.filter(trace=1).values_list('frequency', 'amplitude', named=True)
    datavalues = xaxisdata1
    #[list(item) for item in xaxisdata1]
    if(len(datavalues) < 2000):
        xaxisdata1 = zoomArray(datavalues, np.arange(2000,2000))

    max=-9999
    maxi=-1
    pointrad1 = max;
        #xaxisdata1.order_by('-amplitude').first().amplitude

    #for i,j in xaxisdata1:
    for i in range(0, len(xaxisdata1)):
        for j in range(0, len(xaxisdata1[i])):
            data_dict[i] = j
            if j >= max:
                max = j
            p_dict[i] = 0

    if maxi != -1:
        p_dict[maxi] = 5

    """
    firstlast = 1
    lastval = 0
    pointrad = 1  # mark the peak

    # Penguin | added from here {
    amp_y_value1 = 0.00
    amp_y_value2 = 0.00
    amp_y_value3 = 0.00
    amp_y_value4 = 0.00
    # Penguin | to here }
    amp_y_value21 = 0.00
    amp_y_value22 = 0.00
    amp_y_value23 = 0.00
    amp_y_value24 = 0.00
    amp_y_value31 = 0.00
    amp_y_value32 = 0.00
    amp_y_value33 = 0.00
    amp_y_value34 = 0.00
    amp_y_value41 = 0.00
    amp_y_value42 = 0.00
    amp_y_value43 = 0.00
    amp_y_value44 = 0.00
    amp_y_value51 = 0.00
    amp_y_value52 = 0.00
    amp_y_value53 = 0.00
    amp_y_value54 = 0.00

    # data_dict = OrderedDict()

    xaxisdata1 = nestedfilter.filter(trace=1)
    # no need to plot peak automatically
    # pointrad1 = xaxisdata1.order_by('-amplitude').first().amplitude
    for temp in xaxisdata1:
        lastval = temp.amplitude
        if firstlast == 1:
            firstval = lastval
            data_dict[fstart] = lastval
            p_dict[fstart] = 0
            firstlast = 2

        # duplicate prevention
        # if x not in data_dict.keys():

        data_dict[temp.frequency] = lastval

        # no need to plot peak automatically
        # if temp.amplitude >= pointrad1:
        #    p_dict[temp.frequency] = pointrad
        # else:
        p_dict[temp.frequency] = 0

    # Penguin12 | moved peak search block | do peak search, override, then update amplitude so that I can override the frequency with peak search freq
    # peak search calculation
    # ymva is y moving average
    x, y = zip(*data_dict.items())  # need to do this again as marker point values have been added
    stdev = np.asarray(y).std()
    ymva = uniform_filter1d(y, size=33)
    ymvap = ymva + stdev

    # Penguin13 | Added block | jus' settin' som' variables{
    max_peak_value = -numpy.Inf  # set to smallest so that basically all values are bigger than this -((2 ^ 64)-1) # Penguin20 | edited line
    # min_peak_value = numpy.Inf # same as maxpeakvalue but opposite  # Penguin20 | edited line & commented redundancy
    temp_peak_num = peaknum  # so I don't lose peaknum value before I finish searching min max peak
    peak_i = 0
    prev_peak = -numpy.Inf  # Penguin16 | Added Line # Penguin20 | edited line
    # Penguin13 }
    prev_amp = 0  # Penguin20 | added line
    recharged = 1  # Penguin20 | added line
    ascend = 0  # Penguin20 | added line

    #Penguin21 | added/edited block {
    prev_x = 0
    prev_y = 0
    for x, y, amp in zip(x, ymva, y):
        # Penguin20 | added block {
        if amp >= prev_amp:  # Penguin20 || This only recharges when the slopes have gone up and down once already
            if ascend == 0:
                recharged = 1  # Penguin20 | added line
            ascend = 1
        else:
            ascend = 0
        # Penguin20 | removed redundancy
        # Penguin20 }
        if prev_amp > (prev_y + stdev) and amp < prev_amp and x >= prev_peak + 1000000 and recharged == 1:
            peak_dict[prev_x] = prev_amp  # peak dictionary for peak search usage
            # p_dict[x] = 7  # larger marker for peaks indicator but this needs to be reduced and put above the point #Penguin12 | temporarily disabled so I can see the smaller indicator
            # Penguin13 | Added block | check for "max min peak" time, comrade {
            if peaknum == "MaxPeak":
                if prev_amp > max_peak_value:  # if peak found (signified by adding to peak_dict) check if amp more than maxpeakvalue #Penguin14 | Edited Line | Check for if 0 # Penguin20 | edited line
                    max_peak_value = prev_amp
                    temp_peak_num = str(peak_i)
                peak_i += 1
            """  # Penguin20 | commented redundancy
            if peaknum == "MinPeak":
                if amp < min_peak_value:  # if peak found (signified by adding to peak_dict) check if amp less than minpeakvalue #Penguin14 | Edited Line | Check for if 0
                    min_peak_value = amp
                    temp_peak_num = str(peak_i)
                peak_i += 1
            """
            # Penguin13 }
            prev_peak = x  # Penguin16 | Added Line
            recharged = 0  # Penguin20 | added line
        prev_x = x
        prev_y = y
        prev_amp = amp
    # Penguin21 | }

    # Penguin14 | Added Block | PeakSearch time, I tried to avoid using peak_dict.values(), and while min and max search work, it seems this one isnt the case due to need of an already completed peak_dict{
    splitpeaknum = peaknum.split(".")
    if splitpeaknum[0] == "NextPeak":
        if splitpeaknum[1] == "None":
            splitpeaknum[1] = 0
        for amp in peak_dict.values():
            if max_peak_value < amp < list(peak_dict.values())[int(splitpeaknum[1])] and x != 0:   #search will be limited by highest val
                max_peak_value = amp
                temp_peak_num = str(peak_i)
            peak_i += 1
        if temp_peak_num == peaknum:  # if no change occurred
            temp_peak_num = 0
    # Penguin14 }

    peaknum = temp_peak_num  # Penguin13 | Added Line

    minimum = "None"
    # Penguin12 | Added block | Override enemy frequency {
    current_value = 0
    if peaknum != "None":
        # Penguin13 | Added block / Moved block into this block | So program doesn't crash when no elements in array {
        # Penguin20 | added block/edited block {
        if peaknum == "Min":
            peaknum = 0
            smallest = min(data_dict.values())
            for x, y in data_dict.items():
                if y == smallest:
                    minimum = x
            current_value = minimum  # Penguin20 | commented line
            # Penguin20 | added block/edited block {
        elif list(peak_dict):  # is peak_dict empty?
            current_value = list(peak_dict)[int(peaknum)]  # casting list on peak dict returns only the x values //Penguin20 | commented line
        else:
            peaknum = 0
            current_value = 0  # Penguin20 | commented line
            peak_dict = [0]
        # Penguin13 }
        match current_marker:
            case 0:
                freq_x_value1 = current_value
            case 1:
                freq_x_value1 = current_value
            case 2:
                freq_x_value2 = current_value
            case 3:
                freq_x_value3 = current_value
            case 4:
                freq_x_value4 = current_value
    # Penguin12 }

    result = sorted(data_dict.items())  # sorted by key, return a list of tuples
    x, y = zip(*result)  # unpack a list of pairs into two tuples
    f2 = interp.interp1d(x, y, kind='cubic', axis=0, fill_value="extrapolate")  # Penguin20/19 | edited line | cubic
    # Penguin12 | Removed data_range here and moved it

    # Penguin | added from here {
    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        amp_y_value1 = float(f2(freq_x_value1))
        #data_dict[freq_x_value1] = amp_y_value1  # <-- Your code that exposes the bug hiding since the beginning
        p_dict[freq_x_value1] = pointrad
        # Penguin18 | I cant solve the bug, the interpolation value is wrong
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        amp_y_value2 = float(f2(freq_x_value2))
        #data_dict[freq_x_value2] = amp_y_value2
        p_dict[freq_x_value2] = pointrad
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        amp_y_value3 = float(f2(freq_x_value3))
       # data_dict[freq_x_value3] = amp_y_value3
        p_dict[freq_x_value3] = pointrad
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        amp_y_value4 = float(f2(freq_x_value4))
        #data_dict[freq_x_value4] = amp_y_value4
        p_dict[freq_x_value4] = pointrad
    # Penguin | to here }
    #data_dict[fstart] = firstval
    #p_dict[fstart] = 0
    #data_dict[fstop] = lastval
    #p_dict[fstop] = 0


    """  
    # test code reference
    #fkeys = np.asarray(data_dict.keys())
    idx = (np.abs(np.asarray(x) - interpolatefreq)).argmin()

    #idx = np.searchsorted(fkeys, interpolatefreq, side="left")
    minfidx = 0
    maxfidx = len(x)

    #if idx == 0:
    #    interpolateval = fkeys[idx]

    #prevent out of range
    #if (idx == len(result) or math.fabs(interpolatefreq - result[idx-1]) < math.fabs(interpolatefreq - result[idx])):

    #else:

    if (idx - 16) > minfidx:
        minfidx = idx - 16
    if (idx + 16) < maxfidx:
        maxfidx = idx + 16

    startf = round(get_nth_key(data_dict,minfidx), -3)
    stopf = round(get_nth_key(data_dict,maxfidx), -3)

    xnew = np.arange(startf, stopf, 1000)
    ynew1 = f2(xnew)  # use interpolation function returned by `interp1d`

    plt.figure( "plotting index for : " + str(minfidx) + " to " + str(maxfidx))
    plt.plot(x, y, 'o', xnew, ynew1, 'r-')

    plt.show()
    """
    firstlast = 1
    lastval = 0
    xaxisdata2 = nestedfilter.filter(trace=2)
    pointrad2 = xaxisdata2.order_by('-amplitude').first().amplitude
    for temp in xaxisdata2:
        lastval = temp.amplitude
        if firstlast == 1:
            firstval = lastval
            data_dict2[fstart] = lastval
            p_dict2[fstart] = 0
            firstlast = 2
        data_dict2[temp.frequency] = lastval
        # if temp.amplitude >= pointrad2:
        #    p_dict2[temp.frequency] = pointrad
        # else:
        p_dict2[temp.frequency] = 0

    result = sorted(data_dict2.items())  # sorted by key, return a list of tuples
    x, y = zip(*result)  # unpack a list of pairs into two tuples
    f2 = interp.interp1d(x, y, kind='cubic', axis=0, fill_value="extrapolate")  # Penguin20/19 | edited line | cubic

    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        amp_y_value21 = float(f2(freq_x_value1))
        #data_dict2[freq_x_value1] = amp_y_value21
        p_dict2[freq_x_value1] = pointrad
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        amp_y_value22 = float(f2(freq_x_value2))
        #data_dict2[freq_x_value2] = amp_y_value22
        p_dict2[freq_x_value2] = pointrad
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        amp_y_value23 = float(f2(freq_x_value3))
        #data_dict2[freq_x_value3] = amp_y_value23
        p_dict2[freq_x_value3] = pointrad
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        amp_y_value24 = float(f2(freq_x_value4))
        #data_dict2[freq_x_value4] = amp_y_value24
        p_dict2[freq_x_value4] = pointrad

    #data_dict2[fstart] = firstval
    #data_dict2[fstop] = lastval
    #p_dict2[fstart] = 0
    #p_dict2[fstop] = 0
    firstlast = 1
    lastval = 0

    xaxisdata3 = nestedfilter.filter(trace=3)
    pointrad3 = xaxisdata3.order_by('-amplitude').first().amplitude
    for temp in xaxisdata3:
        lastval = temp.amplitude
        if firstlast == 1:
            firstval = lastval
            data_dict3[fstart] = lastval
            p_dict3[fstart] = 0
            firstlast = 2
        data_dict3[temp.frequency] = lastval
        # if temp.amplitude >= pointrad3:
        #    p_dict3[temp.frequency] = pointrad
        # else:
        p_dict3[temp.frequency] = 0

    result = sorted(data_dict3.items())  # sorted by key, return a list of tuples
    x, y = zip(*result)  # unpack a list of pairs into two tuples
    f2 = interp.interp1d(x, y, kind='cubic', axis=0, fill_value="extrapolate")  # Penguin20/19 | edited line | cubic

    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        amp_y_value31 = float(f2(freq_x_value1))
        #data_dict3[freq_x_value1] = amp_y_value31
        p_dict3[freq_x_value1] = pointrad
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        amp_y_value32 = float(f2(freq_x_value2))
        #data_dict3[freq_x_value2] = amp_y_value32
        p_dict3[freq_x_value2] = pointrad
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        amp_y_value33 = float(f2(freq_x_value3))
        #data_dict3[freq_x_value3] = amp_y_value33
        p_dict3[freq_x_value3] = pointrad
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        amp_y_value34 = float(f2(freq_x_value4))
        #data_dict3[freq_x_value4] = amp_y_value34
        p_dict3[freq_x_value4] = pointrad

    #data_dict3[fstart] = firstval
    #data_dict3[fstop] = lastval
    #p_dict3[fstart] = 0
    #p_dict3[fstop] = 0
    firstlast = 1
    lastval = 0

    xaxisdata4 = nestedfilter.filter(trace=4)
    pointrad4 = xaxisdata4.order_by('-amplitude').first().amplitude
    for temp in xaxisdata4:
        lastval = temp.amplitude
        if firstlast == 1:
            firstval = lastval
            data_dict4[fstart] = lastval
            p_dict4[fstart] = 0
            firstlast = 2
        data_dict4[temp.frequency] = lastval
        # if temp.amplitude >= pointrad4:
        #    p_dict4[temp.frequency] = pointrad
        # else:
        p_dict4[temp.frequency] = 0

    result = sorted(data_dict4.items())  # sorted by key, return a list of tuples
    x, y = zip(*result)  # unpack a list of pairs into two tuples
    f2 = interp.interp1d(x, y, kind='cubic', axis=0, fill_value="extrapolate")  # Penguin20/19 | edited line | cubic

    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        amp_y_value41 = float(f2(freq_x_value1))
        #data_dict4[freq_x_value1] = amp_y_value41
        p_dict4[freq_x_value1] = pointrad
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        amp_y_value42 = float(f2(freq_x_value2))
        #data_dict4[freq_x_value2] = amp_y_value42
        p_dict4[freq_x_value2] = pointrad
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        amp_y_value43 = float(f2(freq_x_value3))
        #data_dict4[freq_x_value3] = amp_y_value43
        p_dict4[freq_x_value3] = pointrad
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        amp_y_value44 = float(f2(freq_x_value4))
        #data_dict4[freq_x_value4] = amp_y_value44
        p_dict4[freq_x_value4] = pointrad
    #data_dict4[fstart] = firstval
    #data_dict4[fstop] = lastval
    #p_dict4[fstart] = 0
    #p_dict4[fstop] = 0
    firstlast = 1
    lastval = 0

    xaxisdata5 = nestedfilter.filter(trace=5)
    pointrad5 = xaxisdata5.order_by('-amplitude').first().amplitude
    for temp in xaxisdata5:
        lastval = temp.amplitude
        if firstlast == 1:
            firstval = lastval
            data_dict5[fstart] = lastval
            p_dict5[fstart] = 0
            firstlast = 2
        data_dict5[temp.frequency] = lastval
        # if temp.amplitude >= pointrad5:
        #    p_dict5[temp.frequency] = pointrad
        # else:
        p_dict5[temp.frequency] = 0

    result = sorted(data_dict5.items())  # sorted by key, return a list of tuples
    x, y = zip(*result)  # unpack a list of pairs into two tuples
    f2 = interp.interp1d(x, y, kind='cubic', axis=0, fill_value="extrapolate")  # Penguin20/19 | edited line | cubic

    if temps[6] != "None" and temps[6] != "":  # Penguin20 | edited line
        amp_y_value51 = float(f2(freq_x_value1))
        #data_dict5[freq_x_value1] = amp_y_value51
        p_dict5[freq_x_value1] = pointrad
    if temps[7] != "None" and temps[7] != "":  # Penguin20 | edited line
        amp_y_value52 = float(f2(freq_x_value2))
        #data_dict5[freq_x_value2] = amp_y_value52
        p_dict5[freq_x_value2] = pointrad
    if temps[8] != "None" and temps[8] != "":  # Penguin20 | edited line
        amp_y_value53 = float(f2(freq_x_value3))
        #data_dict5[freq_x_value3] = amp_y_value53
        p_dict5[freq_x_value3] = pointrad
    if temps[9] != "None" and temps[9] != "":  # Penguin20 | edited line
        amp_y_value54 = float(f2(freq_x_value4))
        #data_dict5[freq_x_value4] = amp_y_value54
        p_dict5[freq_x_value4] = pointrad
    #data_dict5[fstart] = firstval
    #data_dict5[fstop] = lastval
    #p_dict5[fstart] = 0
    #p_dict5[fstop] = 0
    # for group in xaxisdata1:
    # data_dict[group['frequency']] = round(group['amplitude'], 4)
    # for data1 in xaxisdata1:
    #    inp[data1['frequency']]=data1['amplitude']
    in_array = [0, math.pi / 2, np.pi / 3, np.pi]
    # sales_dict = np.sin(in_array)

    data_dict = dict(sorted(data_dict.items()))
    data_dict2 = dict(sorted(data_dict2.items()))
    data_dict3 = dict(sorted(data_dict3.items()))
    data_dict4 = dict(sorted(data_dict4.items()))
    data_dict5 = dict(sorted(data_dict5.items()))

    """ # Penguin20 | commented redundancy
    # Penguin18 | added block {
    pop_value1 = []
    pop_value2 = []
    pop_value3 = []
    pop_value4 = []
    pop_value5 = []
    for values in data_dict:
        if values > fstop:
            pop_value1.append(values)
    for values in data_dict2:
        if values > fstop:
            pop_value2.append(values)
    for values in data_dict3:
        if values > fstop:
            pop_value3.append(values)
    for values in data_dict4:
        if values > fstop:
            pop_value4.append(values)
    for values in data_dict5:
        if values > fstop:
            pop_value5.append(values)

    for a in pop_value1:
        data_dict.pop(a)
    for a in pop_value2:
        data_dict2.pop(a)
    for a in pop_value3:
        data_dict3.pop(a)
    for a in pop_value4:
        data_dict4.pop(a)
    for a in pop_value5:
        data_dict5.pop(a)
    # Penguin18 }
    """

    p_dict = dict(sorted(p_dict.items()))
    p_dict2 = dict(sorted(p_dict2.items()))
    p_dict3 = dict(sorted(p_dict3.items()))
    p_dict4 = dict(sorted(p_dict4.items()))
    p_dict5 = dict(sorted(p_dict5.items()))

    # inp = np.linspace(-np.pi, np.pi*100, 1200)
    # opt = np.sin(inp)*10
    inp = data_dict.keys()
    # opt = [x+10 for x in data_dict.values()]

    # data_dict.values()  #np.array([x for x in data_dict.values()])+500
    # 'title': f'Atten { year/100-0.21 } dB',
    return JsonResponse({
        # 'title': f'{year}',  # Penguin22 | commented line
        # Penguin | added from here {
        'amp_y_value1': f'{amp_y_value1}',
        'amp_y_value2': f'{amp_y_value2}',
        'amp_y_value3': f'{amp_y_value3}',
        'amp_y_value4': f'{amp_y_value4}',
        # Penguin | to here }
        'amp_y_value21': f'{amp_y_value21}',
        'amp_y_value22': f'{amp_y_value22}',
        'amp_y_value23': f'{amp_y_value23}',
        'amp_y_value24': f'{amp_y_value24}',
        'amp_y_value31': f'{amp_y_value31}',
        'amp_y_value32': f'{amp_y_value32}',
        'amp_y_value33': f'{amp_y_value33}',
        'amp_y_value34': f'{amp_y_value34}',
        'amp_y_value41': f'{amp_y_value41}',
        'amp_y_value42': f'{amp_y_value42}',
        'amp_y_value43': f'{amp_y_value43}',
        'amp_y_value44': f'{amp_y_value44}',
        'amp_y_value51': f'{amp_y_value51}',
        'amp_y_value52': f'{amp_y_value52}',
        'amp_y_value53': f'{amp_y_value53}',
        'amp_y_value54': f'{amp_y_value54}',
        'peaknum': f'{peaknum}',  # Penguin13 | Added line | I tried to avoid this but well
        'freq_data_range': list(data_range),  # Penguin3 | Added
        'peak_list': list(peak_dict),  # Penguin12 | Added
        'min': f'{minimum}',
        'data': {
            'labels': list(inp),
            'datasets': [
                {
                    'fill': 'false',
                    'label': ' ',
                    'lineTension': '0.1',
                    'tension': '0',
                    'pointRadius': 0,
                    'borderWidth': 0.5,
                    'pointStyle': 'triangle',
                    'rotation': 60,
                    'pointBackgroundColor': colorSuccess,
                    'pointBorderColor': colorSuccess,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'data': list(data_dict.values()),  # ist(opt),
                },
                {
                    'fill': 'false',
                    'hidden': 'false',
                    'label': ' ',
                    'lineTension': '0',
                    'tension': '0',
                    'borderWidth': 0.5,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'pointRadius': 0,
                    'data': list(data_dict.values()),  # sales_dict.values()),
                },
                {
                    'fill': 'false',
                    'hidden': 'false',
                    'label': ' ',
                    'lineTension': '0',
                    'tension': '0',
                    'borderWidth': 0.5,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'pointRadius': 0,
                    'data': list(data_dict2.values()),  # sales_dict.values()),
                },
                {
                    'fill': 'false',
                    'hidden': 'false',
                    'label': ' ',
                    'lineTension': '0',
                    'tension': '0',
                    'borderWidth': 0.5,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'pointRadius': 0,
                    'data': list(data_dict3.values()),  # sales_dict.values()),
                },
                {
                    'fill': 'false',
                    'hidden': 'false',
                    'label': ' ',
                    'lineTension': '0',
                    'tension': '0',
                    'borderWidth': 1,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'pointRadius': 0,
                    'data': list(data_dict4.values()),  # sales_dict.values()),
                },
                {
                    'fill': 'false',
                    'hidden': 'false',
                    'label': ' ',
                    'lineTension': '0',
                    'tension': '0',
                    'borderWidth': 0.5,
                    'backgroundColor': colorPrimary,
                    'borderColor': colorPrimary,
                    'pointRadius': 0,
                    'data': list(data_dict5.values()),  # sales_dict.values()),
                }
            ]
        },
    })


@staff_member_required
def statistics_view(request):
    return render(request, 'statistics.html', {})


def zoomArray(inArray, finalShape, zorder=3, sameSum=False,
              zoomFunction=scipy.ndimage.zoom):
    """
    Parameters
    ----------
    inArray: n-dimensional numpy array
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    inArray = np.asarray(inArray, dtype="f,f")
    inShape = inArray.shape
    if len(inShape) >= len(finalShape):
        return inArray

    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)

    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, order=zorder)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")