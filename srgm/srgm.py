# Copyright 2019 Nokia
# Licensed under the BSD 3 Clause Clear license
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import division
from math import exp, sqrt
import json
import copy
import sys
import numpy as np
from scipy.optimize import fmin, minimize
import json
import datetime
from datetime import timedelta 
from pprint import pprint


def srgm(defect_totals, inputs, p7, p8, calculate_closure_rate=True):

    #print("*************INPUTS**************************")
    #print('defect_totals', defect_totals)
    #print('inputs:', inputs)
    #print('p7:', p7)
    #print('p8:', p8)
    #print("*************OUTPUTS**************************")


    #print("Totals....")

    #print(defect_totals)

    weekly_created = defect_totals['created'][:-1]
    weekly_resolved = defect_totals['resolved'][:-1]
    cumulative_created = defect_totals['cumulative_created'][:-1]
    cumulative_resolved = defect_totals['cumulative_resolved'][:-1]

    # Caclculate horizon
    horizon=p8
    if len(defect_totals['created'])>p8:
        horizon=len(defect_totals['created'])

    output_array_length = horizon + inputs['beyondp8']

    inputs['percentage_rate_reduction_p7'] = int(inputs['percentage_rate_reduction_p7']) / 100
    inputs['percentage_rate_reduction_p7_p8'] = int(inputs['percentage_rate_reduction_p7_p8']) / 100
    inputs['b_0'] = inputs['b_0']
    inputs['ssq_condition'] = inputs['ssq_condition']
    inputs['minpoints'] = int(inputs['minpoints'])
    inputs['beyondp8'] = int(inputs['beyondp8'])
    inputs['percentage_of_high_severity'] = int(inputs['percentage_of_high_severity']) / 100
    inputs['percentage_high_severity_post_P8'] = int(inputs['percentage_high_severity_post_P8']) / 100
    inputs['defect_conversion_factor'] = inputs['defect_conversion_factor']
    inputs['software_fault_coverage'] = int(inputs['software_fault_coverage']) / 100
    inputs['average_manual_recovery_time'] = int(inputs['average_manual_recovery_time'])
    inputs['software_reboot_time'] = int(inputs['software_reboot_time'])
    inputs['high_severity_found_not_fixed_p8'] = int(inputs['high_severity_found_not_fixed_p8'])
    inputs['in_service_duration'] = int(inputs['in_service_duration'])
    #inputs['training_period'] = int(inputs['training_period'])


    b_0 = inputs['b_0']
    beyondp8 = inputs['beyondp8']
    minpoints = inputs['minpoints']
    ssq_condition = inputs['ssq_condition']

    #print("minpoints: ", minpoints)

    metrics = {
        'software_availability_estimate': None,
        'software_availability_lowerlimit': None,
        'software_availability_upperlimit': None,
        'software_downtime_estimate': None,
        'software_downtime_lowerlimit': None,
        'software_downtime_upperlimit': None,
        'software_failure_rate_estimate': None,
        'software_failure_rate_lowerlimit': None,
        'software_failure_rate_upperlimit': None,
        'defect_rate_at_delivery': None,
        'defect_rate_at_p7': None,
        'defect_rate_at_p8': None,
        'defects_at_p8': None,
        'defects_at_p7': None,
        'residual_defects_at_p7_estimate': None,
        'residual_defects_at_p7_lowerlimit': None,
        'residual_defects_at_p7_upperlimit': None,
        'defects_post_p8_estimate': None,
        'defects_post_p8_lowerlimit': None,
        'defects_post_p8_upperlimit': None,
        'total_defects_estimate': None,
        'total_defects_lowerlimit': None,
        'total_defects_upperlimit': None
    }

    

    outputs = {
        'created': [0] * output_array_length, 
        'resolved': [0] * output_array_length, 
        'cumulative_created': [0] * output_array_length,
        'cumulative_resolved': [0] * output_array_length,
        'open': [0] * output_array_length,
        'cumulative_at_p7': [0] * output_array_length,
        'cumulative_created_post_p7': [0] * output_array_length,
        # 'metrics': {} 
    }

    # Initializing Curves
    curves = get_initial_curves()

    minimum_week = 1
    maximum_week = len(cumulative_created)
    ssq_array, b_array, a_array = [], [], []

    x_s = minimum_week
    stop = min(p7, maximum_week)
    start = minimum_week + minpoints

    for i in range(start, stop + 1):
        y_s = cumulative_created[x_s - 1]
        y_i = cumulative_created[i - 1]

        if y_i > y_s:
            b = optimize(cumulative_created, x_s, i, b_0)
            a = calculate_a(cumulative_created, x_s, i, b)

            ssq_i = calculate_ssq(cumulative_created, x_s, i, y_s, a, b, p7, outputs)
            #track_i.append(i)
            #track_ys.append(y_s)
            #track_curves[str(i)]['p7'] = 
            ssq_array.append(ssq_i)
            size = len(ssq_array)

            b_array.append(b)
            a_array.append(a)
            
            nextcurve = checkssq(ssq_array, ssq_condition, minpoints, x_s, i)

            if nextcurve == 1:
                b_0 = inputs['b_0']

                b = b_array[len(b_array)-3]
                a = a_array[len(a_array)-3]
                update_curves(i, x_s, y_s, a, b, curves)

                x_s = i - 2
                i = x_s + minpoints - 1

                ssq_array = []
                b_array = []
                a_array = []

            elif x_s == stop - 2:
                adjust_last(stop, curves)

            elif i >= stop:
                add_last_curve(stop, x_s, y_s, a, b, curves)
                
                if 2 < stop - x_s < 7:
                    add_data(x_s, stop, cumulative_created, curves, minpoints)

                if b < 0.005:
                    add_data_edited(cumulative_created, curves)



    preP7_cumulative_predictions(cumulative_created, p7, curves, outputs)

    #print(outputs['cumulative_created'])

    #exit()
    # preP7_cumulative_predictions(cumulative_created, p7, curves, outputs, 'cumulative_created_post_p7')
    preP7_weekly_predictions(p7, outputs)
    #print(outputs['cumulative_created'])
    postp7_weekly_predictions(weekly_created, p7, p8, outputs, beyondp8, inputs, metrics) # cumulative / weekly predictions after p7
    #print(outputs['cumulative_created'])
    postp7_cumulative_predictions(weekly_created, p7, p8, outputs, beyondp8, cumulative_created)

    #print(outputs['cumulative_created'])

   # exit()
    # preP7_cumulative_predictions(outputs['cumulative_created_post_p7'], p7, curves, outputs, 'cumulative_created')
    if (calculate_closure_rate): closure_rate_predictions(weekly_resolved, cumulative_resolved, outputs, p8, beyondp8)
    # populate_reliability_metrics(curves, outputs, parameters['actual_data'], inputs, p7, p8, cumulative_created, metrics)

    # outputs['metrics'] = metrics

    # Calculating Open Values
    for x in range(len(outputs['cumulative_created'])):
        val = outputs['cumulative_created'][x] - outputs['cumulative_resolved'][x]

        if val > 0:
            outputs['open'][x] = val

        else:
            for i in range(x, len(outputs['cumulative_created'])):
                outputs['open'][i] = 0
            break

    #print("outputs: ", outputs)

    return outputs



def add_data_edited(cumulative_created, curves):
    y = cumulative_created[-4:]
    increment = (y[3]-y[0])/3
    flag = 1
    b_0 = 0.05
    while flag != 0:
        m = 1
        incr_rate = 0.9-0.05*(m-1)
        for j in range(1, 4):
            y[j] = y[j-1]+increment*(incr_rate-(1-incr_rate)*(j-1))

        # Verify with Rashid whether or not it is okay to remove these +1s. 
        # b = optimize(y, 1, len(y)+1, b_0)
        # a = calculate_a(y, 1, len(y)+1, b)
        b = optimize(y, 1, len(y), b_0)
        a = calculate_a(y, 1, len(y), b)
        b_0 = b
        m += 1

        if b > 0.005:
            flag = 0
            update_last_curve(b, a, curves)


# ------------------------------------------------------------------------------#
def postp7_weekly_predictions(weekly_created, p7, p8, outputs, beyondp8, inputs, metrics):

    w_t = 0
    previousweek = 0
    previousrate = 0

    for week in range(p7, len(outputs['cumulative_created']) + 1):

        if week == p7:
            previousweek = week

            if (week <= len(weekly_created)):
                w_t = weekly_created[week - 1]
                

            else:
                w_t = outputs['created'][week - 1]

    r1 = inputs['percentage_rate_reduction_p7']
    r2 = inputs['percentage_rate_reduction_p7_p8']

    w_7 = w_t * (1 - r1)
    w_8 = w_7 * (1 - r2)

    metrics['defect_rate_at_delivery'] = float("{0:.3f}".format(w_t))
    metrics['defect_rate_at_p7'] = float("{0:.3f}".format(w_7))
    metrics['defect_rate_at_p8'] = float("{0:.3f}".format(w_8))

    for i in range(previousweek, p8):
        outputs['created'][i] = int(w_7)
        #outputs['cumulative_created'][i] = outputs['cumulative_created'][i-1] + int(w_t)

    for i in range(p8, len(outputs['cumulative_created'])):
        outputs['created'][i] = int(w_8)
        #outputs['cumulative_created'][i] = outputs['cumulative_created'][i-1] + int(w_t)


#------------------------------------------------------------------------------#
def get_initial_curves():
    curves = {}
    curves[1] = {}
    curves[1]['a'] = 0
    curves[1]['b'] = 0
    curves[1]['startweek'] = 1
    curves[1]['endweek'] = 1
    curves[1]['lastcurve'] = 1
    return curves


#------------------------------------------------------------------------------#
def optimize(cumulative_created, min_week, max_week, b_0):
    y = adjustvalues(cumulative_created, min_week, max_week)
    x = create_values(y)
    p = max_week - min_week
    y_p = y[p]
    x_p = x[p]

    y = json.dumps(y)
    x = json.dumps(x)

    x_p_y_p = x_p * y_p

    b = calc(x, y, x_p_y_p, x_p)
    return b


#------------------------------------------------------------------------------#
def adjustvalues(cumulative_created, min_week, max_week):
    new_array = []

    y_0 = cumulative_created[min_week - 1]

    for x in range(min_week - 1, len(cumulative_created)):
        adj = cumulative_created[x] - y_0
        new_array.append(adj)

    return new_array


#------------------------------------------------------------------------------#
def create_values(y):
    x = []
    count = len(y)
    for i in range(count):
        x.append(i)
    return x


#------------------------------------------------------------------------------#
def calculate_a(cumulative_created, min_week, max_week, b):
    y_0 = cumulative_created[(min_week - 1)]
    x_p = max_week - min_week
    y_p = cumulative_created[max_week - 1] - y_0
    a = y_p / (1 - exp(-b * x_p))
    return a


#------------------------------------------------------------------------------#
def calculate_ssq(cumulative_created, x_s, i, y_s, a, b, p7, outputs):
    ssq_sum = 0

    for j in range(x_s, i + 1):
        x_i = j - x_s
        y_i = cumulative_created[j - 1]
        prediction = a * (1 - exp(-b * x_i)) + y_s
        ssq = pow(y_i - prediction, 2)
        ssq_sum += ssq

    ssq_i = ssq_sum / (i - x_s)

    try:
        pred_at_p7 = int(a * (1 - exp(-b * p7)) + y_s)

    except:
        pred_at_p7 = None

    if (b > 0.0):
        outputs['cumulative_at_p7'][i - 1] = pred_at_p7

    return ssq_i


#------------------------------------------------------------------------------#
def checkssq(ssq_array, ssq_condition, minpoints, x_s, i):
    delta = exp(-50)
    nextcurve = 0

    if len(ssq_array) > 2:
        SSQ = ssq_array[len(ssq_array)-1]
        SSQ_1 = ssq_array[len(ssq_array)-2]
        SSQ_2 = ssq_array[len(ssq_array)-3]
        delta1 = abs(SSQ_2 - SSQ_1) /  (SSQ_2 + delta)
        delta2 = abs(SSQ_2 - SSQ) /  (SSQ_2 + delta)
        trans = i - x_s - 2

        if delta1 > ssq_condition and delta2 > ssq_condition and trans > minpoints and SSQ_2 > 1:
            nextcurve = 1

    return nextcurve


#------------------------------------------------------------------------------#
def update_curves(i, x_s, y_s, a, b, curves):
    curve = 1
    x_i_2 = i - 2

    if x_s != 1:
        curve = get_curve_number(curves, x_s)

    curves[curve] = {}
    curves[curve]['a'] = a
    curves[curve]['b'] = b
    curves[curve]['startweek'] = x_s
    curves[curve]['endweek'] = x_i_2
    curves[curve]['lastcurve'] = 1

    if curve >= 2:
        cur = curve - 1
        curves[cur]['lastcurve'] = 0


#------------------------------------------------------------------------------#
def get_curve_number(curves, x_s):
    maxcurve = 0
    curvenumber = 0
    all = curves

    for curv in all:
        week = all[curv]['startweek']

        if x_s == week:
            curvenumber = curv

        if curv > maxcurve:
            maxcurve = curv

    if curvenumber == 0:
        curvenumber = maxcurve + 1

    return curvenumber


#------------------------------------------------------------------------------#
def adjust_last(stop, curves):
    curve = get_last_curve(curves)
    curves[curve]['endweek'] = stop
    curves[curve]['lastcurve'] = 1


#------------------------------------------------------------------------------#
def get_last_curve(curves):
    maxcurve = 0
    curvs = curves

    for curv in curvs:
        if curv > maxcurve:
            maxcurve = curv
    return maxcurve


#------------------------------------------------------------------------------#
def add_last_curve(i, x_s, y_s, a, b, curves):
    curve = 1
    total_defects = a + y_s
    x_i_2 = i

    if x_s != 1:
        curve = get_curve_number(curves, x_s)

    curves[curve] = {}
    curves[curve]['a'] = a
    curves[curve]['b'] = b
    curves[curve]['startweek'] = x_s
    curves[curve]['endweek'] = x_i_2
    curves[curve]['lastcurve'] = 1

    if curve >= 2:
        cur = curve - 1
        curves[cur]['lastcurve'] = 0


#------------------------------------------------------------------------------#
def add_data(start, stop, cumulative_created, curves, minpoints):
    num = len(cumulative_created)
    end = 0

    if stop < num < start + 5:
        end = num

    elif num > stop and num >= start + 5:
        end = start + 5

    else:
        end = stop

    numpoints = end - start + 1

    y = []

    for i in range(start, end + 1):
        y.append(cumulative_created[i - 1])

    y_s = y[0]
    y_start = y[0]
    y_stop = y[-1]
    delta = end - start
    avg = (y_stop - y_start) / delta

    for j in range(1, 4):
        if numpoints < 6:
            y.append(y_stop + avg * j* 0.95)
            numpoints += 1

    for j in range(4, 7):
        if numpoints < 6:
            y.append(y_stop + avg * j *(0.95-0.05 *j))
            numpoints += 1

    k = json.dumps(y)

    b_0 = 0.05
    a = b = 0

    for k in range(minpoints, 7):
        b = optimize(y, 1, k, b_0)
        a = calculate_a(y, 1, k, b)
        b_0 = b

    update_last_curve(b, a, curves)


#------------------------------------------------------------------------------#
def update_last_curve(b, a, curves):
    curve = get_last_curve(curves)
    curves[curve]['a'] = a
    curves[curve]['b'] = b













#------------------------------------------------------------------------------#
#--------------------------------- SRGM Stuff ---------------------------------#
#------------------------------------------------------------------------------#
def calc(var1, var2, var3, var4):
    delta = 1e-6  
   
    try:
        x = json.loads(var1)
        y = json.loads(var2)
        x_p_y_p = float(var3)
        x_p = float(var4)
        y = np.array(y)
        x = np.array(x)

    except:
        print('Could Not Decode Passed Parameter')
        sys.exit(1)


    def srgm(b):
        y_diff = (y[1:] - y[:-1])

        term1 = x[1:] * np.exp(-x[1:] * b[0])

        term2 = x[:-1] * np.exp(-x[:-1] * b[0])

        a_i = y_diff * (term1 - term2)

        term3 = np.exp(-x[:-1] * b[0])

        term4 = np.exp(-x[1:] * b[0])

        b_i = term3 - term4

        a_i_b_i = sum(a_i / (b_i + delta)  )
        return np.absolute(a_i_b_i - (x_p_y_p / (np.exp(x_p * b[0]) - 1 + delta)))  # delta added to avoid dision by zero


    b0 = np.array([0.05])
    res = fmin(srgm, b0, xtol=1e-15, disp=0) 
    xopt = res[0]
    fopt = srgm(np.array([xopt]))
    return xopt



#------------------------------------------------------------------------------#
def preP7_cumulative_predictions(cumulative_created, p7, curves, outputs, field=None):
    act = []
    pred = []
    ssq = 0
    ceiling = cumulative_created[-1] * 2

   #print(cumulative_created)
    #print("*************************************")
    #print(curves)
    #print("*************************************")
    #print(outputs)

    #exit()


    for curv, x in zip(curves, range(len(curves))):
        
        start = curves[curv]['startweek']
        end = curves[curv]['endweek']
        end_firm = end
        last = curves[curv]['lastcurve']
        if last == 1: end = len(outputs['created'])

        a = curves[curv]['a']
        b = curves[curv]['b']

        x_s = start
        y_s = cumulative_created[x_s - 1]


        for j in range(start, end + 1):

            predicted = a * (1 - exp(-b*(j - x_s))) + y_s
            
            predicted = min(int(predicted), ceiling)

            if j <= end_firm:
                actual = cumulative_created[j - 1]

            else:
                actual = predicted

            act.append(actual)
            pred.append(predicted)

            ssq_j = pow((actual - predicted), 2)
            ssq += ssq_j

  
            outputs['cumulative_created'][j - 1] = int(predicted)
            outputs['cumulative_created_post_p7'][j - 1] = int(predicted)
            # outputs[field][j - 1] = int(predicted)
    #print(len(outputs['cumulative_created']))
    #print(len(cumulative_created))
    #print(outputs['cumulative_created'])     
    #exit()


#------------------------------------------------------------------------------#
def preP7_weekly_predictions(p7, outputs):

    previous = previous_predicted = 0
    weekly_actual = weekly_predicted = 0

    for week in range(p7):
        predicted = outputs['cumulative_created'][week]

        if week == 0:
            weekly_predicted = predicted


        else:
            weekly_predicted = predicted - previous_predicted
            
        outputs['created'][week] = weekly_predicted
    
        previous_predicted = predicted


#------------------------------------------------------------------------------#
def postp7_cumulative_predictions(weekly_created, p7, p8, outputs, beyondp8, cumulative_created):
    limit = len(outputs['cumulative_created'])

    ceiling = cumulative_created[-1] * 2

    previous = 0
    actual = 0

    for week in range(p7 - 1, limit):
        weekly = outputs['created'][week]
      
        if week == p7 - 1:
            previous = outputs['cumulative_created_post_p7'][week]
        
        else:
            actual = previous + weekly
            actual = min(actual, ceiling)
            outputs['cumulative_created_post_p7'][week] = actual
            previous = actual


#------------------------------------------------------------------------------#
def closure_rate_predictions(weekly_resolved, cumulative_resolved, outputs, p8, beyondp8):
    predicted_cumulative_created = outputs['cumulative_created']
    endweek = 0
    weekly_resolved.append(None)
    i = 0

    while i < min(len(weekly_resolved), len(predicted_cumulative_created)):

        week = i + 1
        if i >= 2:
            z1 = weekly_resolved[i-1]
            z2 = weekly_resolved[i-2]
            z3 = weekly_resolved[i-3]
            if z1 is not None and z2 is not None and z3 is not None and weekly_resolved[i] is None:
                endweek = week
                i = len(weekly_resolved) - 1
        i += 1

    cum_pred = cumulative_resolved[endweek - 2]

    pred = weekly_resolved[endweek - 2]
    del weekly_resolved[-1]
    k = endweek - 1
    outputs['resolved'][k] = pred
    outputs['cumulative_resolved'][k] = cum_pred

    k = endweek
    while k <= len(outputs['cumulative_created']):

        z1z2 = (z1 + z2) / 2

        z1z2z3 = (z1 + z2 + z3) / 3
        #pred = max(z1, z1z2, z1z2z3)
        pred = z1z2z3   #changed the closure rate prediction to only be based on the average of the last three closure rates.
        cum_pred += pred
        z3 = z2
        z2 = z1
        z1 = pred
        endweek += 1
        wk = endweek - 1
        pred_found = predicted_cumulative_created[wk - 1]
        outputs['resolved'][k - 1] = int(pred)
        outputs['cumulative_resolved'][k - 1] = int(cum_pred)

        if cum_pred > pred_found:
            k = len(outputs['cumulative_created']) + 1
        k += 1

def get_metrics(total_defects, defects_at_p8, inputs):
    percentage_of_high_severity = inputs['percentage_of_high_severity']
    a_11 = total_defects * percentage_of_high_severity   # total high severity defects
    a_12 = defects_at_p8 * percentage_of_high_severity  # high severity defects found at service-in
    a_1 = a_11 - a_12   # High severity defects not yet found at service-in
    a_2 = inputs['high_severity_found_not_fixed_p8']
    a = a_1 + a_2   # high severity defects delivered at service-in
    b_0 = inputs['percentage_high_severity_post_P8']  # percentage of high severity defects to be found during service
    b_1 = inputs['in_service_duration']
    b = a*b_0*(b_1/52)   # high severity defects per year to be found in service
    c = inputs['defect_conversion_factor']   # defect_conversion_factor
    d = b * c  # software failure rate
    e = inputs['software_fault_coverage']     # software_fault_coverage
    f = inputs['average_manual_recovery_time']   # average_manual_recovery_time
    g = inputs['software_reboot_time']   # software_reboot_time

    # NetAct 17.2
    # print()
    # print('a_11:', a_11)
    # print('a_12:', a_12)
    # print('a_1:', a_1)
    # print('a_2:', a_2)
    # print('a:', a)
    # print('b_0:', b_0)
    # print('b_1:', b_1)
    # print('b:', b)
    # print('c:', c)
    # print('d:', d)
    # print('e:', e)
    # print('f:', f)
    # print('g:', g)
    # print()
    annual_downtime = d * (((1 - e) * f) + (e * g))
    availability = (1 - (annual_downtime / (60 * 24 * 365))) * 100
    return d, availability, annual_downtime

#------------------------------------------------------------------------------#
def populate_reliability_metrics(curves, outputs, processed_issue, inputs, p7, p8, arr, metrics):

    number_of_curves = len(curves)
    curve = curves[number_of_curves]
    a = curve['a']
    b = curve['b']
    week = curve['startweek']

    y_s = arr[week - 1]

    total_defects = outputs['cumulative_created_post_p7'][-1]
    defects_at_p8 = outputs['cumulative_created_post_p7'][p8]
    defects_at_p7 = outputs['cumulative_created_post_p7'][p7]
    failure_rate, availability, annual_downtime = get_metrics(total_defects, defects_at_p8, inputs)

    # print(failure_rate)
    # print('total_defects',total_defects)
    # print('defects_at_p8',defects_at_p8)
    # print('a',a)
    # print('y_s',y_s)

    metrics['total_defects_estimate'] = max(0, int(total_defects))
    metrics['defects_at_p8'] = max(0, int(defects_at_p8))
    metrics['defects_at_p7'] = max(0, int(defects_at_p7))
    metrics['residual_defects_at_p7_estimate'] = max(0, int(total_defects - defects_at_p7))
    metrics['defects_post_p8_estimate'] = max(0, int((total_defects - defects_at_p8)*(inputs['percentage_high_severity_post_P8'])))
    metrics['software_availability_estimate'] = min(100, max(0, float("{0:.3f}".format(availability))))
    metrics['software_downtime_estimate'] = max(0, float("{0:.3f}".format(annual_downtime)))

    # print(max(0, float("{0:.3f}".format(failure_rate))))
    metrics['software_failure_rate_estimate'] = max(0, float("{0:.3f}".format(failure_rate)))


    total_defects_ucl, total_defects_lcl = 0, 0 
    try:
        total_defects_ucl, total_defects_lcl = calculatelimits(b, y_s, week, arr)
    except: 
        pass

    if total_defects_ucl is not None:

        metrics['total_defects_upperlimit'] = max(0, int(total_defects_ucl))
        metrics['total_defects_lowerlimit'] = max(0, int(total_defects_lcl))

        metrics['residual_defects_at_p7_upperlimit'] = max(0, int(total_defects_ucl - defects_at_p7))
        metrics['residual_defects_at_p7_lowerlimit'] = max(0, int(total_defects_lcl - defects_at_p7))

        metrics['defects_post_p8_upperlimit'] = max(0, int((total_defects_ucl - defects_at_p8)*(inputs['percentage_high_severity_post_P8'])))
        metrics['defects_post_p8_lowerlimit'] = max(0, int((total_defects_lcl - defects_at_p8)*(inputs['percentage_high_severity_post_P8'])))

        failure_rate, availability, annual_downtime = get_metrics(total_defects_ucl, defects_at_p8, inputs)
        #metrics['software_availability_lowerlimit'] = max(0, float("{0:.3f}".format(max(0, availability))))
        #metrics['software_downtime_upperlimit'] = max(0, float("{0:.3f}".format(annual_downtime)))
        #metrics['software_failure_rate_upperlimit'] = max(0, float("{0:.3f}".format(annual_downtime)))

        failure_rate, availability, annual_downtime = get_metrics(total_defects_lcl, defects_at_p8, inputs)
        #metrics['software_availability_upperlimit'] = max(0, float("{0:.3f}".format(min(100, availability))))
        #metrics['software_downtime_lowerlimit'] = max(0, float("{0:.3f}".format(max(0, annual_downtime))))
        #metrics['software_failure_rate_lowerlimit'] = max(0, float("{0:.3f}".format(max(0, failure_rate))))

#------------------------------------------------------------------------------#
def calculatelimits(b, y_s, week, arr):

    max_wk = len(arr)
    y = adjustvalues(arr, week, max_wk)
    y_p = y[len(y)-1]
    x = create_values(y)
    x_p = x[len(x)-1]

    temp4 = y_p * x_p * x_p / (exp(b * x_p) - 1 + 0.00000001) # Delta added to avoid division by zero. 
    temp3 = 0

    for i in range(1, len(x)):
        temp3 += (y[i] - y[i-1])*(x[i] - x[i-1])*(x[i] - x[i-1])*exp(-b * x[i])

    total_defects_ucl = None
    total_defects_lcl = None

    if temp4 > temp3:
        sigma = sqrt(temp4 - temp3)

        b_lcl = b - 1.64 / sigma

        if b_lcl >= 0:
            b_ucl = b + 1.64 / sigma
            total_defects_ucl = int(y_p / (1 - exp(-b_lcl * x_p)) + y_s)
            total_defects_lcl = int(y_p / (1 - exp(-b_ucl * x_p)) + y_s)

    return total_defects_ucl, total_defects_lcl