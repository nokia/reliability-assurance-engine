# Copyright 2019 Nokia
# Licensed under the BSD 3 Clause Clear license
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd
import numpy as np
from datetime import datetime
import math


# increments = 0
# search_range = 0
# P7_NUM = 0
# current_date = 0
# qq_plot_start = 5
# qq_plot_end = 100
# qq_plot_increment = 5
# qq_plot_limit  = 0.3


def run_edpm(feature_data, defect_data, P7, inc, date, srange, qq_start, qq_end, qq_increment, qq_limit):

	global P7_NUM, increments, current_date, search_range, qq_plot_start, qq_plot_end, qq_plot_increment, qq_plot_limit

	#feature_data.to_csv("feature_data.csv")

	#defect_data.to_csv("defect_data.csv")

	P7_NUM = P7
	increments = inc
	current_date = date
	search_range = srange
	qq_plot_start = qq_start
	qq_plot_end = qq_end
	qq_plot_increment = qq_increment
	qq_plot_limit  = qq_limit

	# print('P7_NUM =', P7,
	# 'increments =', inc,
	# 'current_date =', date,
	# 'search_range =' ,srange)

	defects_start_date = defect_data['Date_Ending'].values[0]

	features_start_date = feature_data['Month_Ending'].values[0]

	defects_end_date = defect_data['Date_Ending'].values[-1]

	features_end_date = feature_data['Month_Ending'].values[-1]

	defect_data['X'] = 1+(defect_data['Date_Ending'] - defects_start_date).dt.days.astype(int)

	feature_data['X'] = 1+(feature_data['Month_Ending'] - features_start_date).dt.days.astype(int)

	feature_data.reset_index(inplace=True)


	feature_new_days = list(range(feature_data['X'].values[0], feature_data['X'].values[-1], increments))

	defect_new_days = list(range(defect_data['X'].values[0], defect_data['X'].values[-1], increments))

	gap = int(((defects_start_date - features_start_date).astype('timedelta64[D]').astype(int))/increments)

	#print(feature_data)

	#print(defect_data)

	#exit()

	feature_new_data = perform_interpolation(feature_new_days, feature_data['X'].values, feature_data['Sub-feature_Arrival'].values)

	defect_new_data = perform_interpolation(defect_new_days, defect_data['X'].values, defect_data['Created'].values)

	resolved_new_data = perform_interpolation(defect_new_days, defect_data['X'].values, defect_data['Resolved'].values)


	#print(feature_new_days)

	#print(final_index)

	#print("XXXXXXX")
	#exit()


	final_data = get_data(feature_new_days, defect_new_days, feature_new_data, defect_new_data, resolved_new_data)

	final_data['WEEK'] = final_data.index.values + 1



	#print(final_data)

	#final_data.to_csv("data_new.csv")

	#print("m: ", gap)

	#print("p7: ", P7_NUM)

	#print("increments: ", increments)


	a, b, c = create_qq_plot(final_data['FEATURES'].values, final_data['ARRIVALS'].values)

	final_data['WEEK_(X_NEW)'] = a + b * final_data['WEEK']

	final_data['ARRIVALS_(Y_NEW)'] = c * final_data['FEATURES']

	ssq = get_ssq(final_data['ARRIVALS'].values, final_data['WEEK_(X_NEW)'].values, final_data['ARRIVALS_(Y_NEW)'].values)

	#print("SSQ:", ssq)

	#print(final_data)


	#exit()

	N_p = current_date/(P7_NUM)

	F_p = int(N_p*len(final_data['FEATURES'].dropna().values))

	start_week = max(0, (F_p - search_range))

	end_week = min((F_p + search_range), (len(final_data['FEATURES'].dropna().values)))


	evaluation = []
	for index in range(start_week, end_week):
		feature_data = final_data['FEATURES'].values[:index]
		arrivals = final_data['ARRIVALS'].values

		week_data = np.asarray([i+1 for i in range(len(feature_data))])

		#print(week_data)

		a, b, c = create_qq_plot(feature_data, arrivals)

		x_new = a + b * week_data

		y_new = c * feature_data


		#print("x_new: ", len(x_new))
		#print("y_new: ", len(y_new))
		#print("week_data: ", len(week_data))
		#print("arrivals: ", len(arrivals))
		#exit()

		ssq = get_ssq(arrivals, x_new, y_new)

		evaluation.append([index, a, b, c, ssq])


	df = pd.DataFrame(evaluation, columns=['index', 'intercept', 'slope', 'ratio', 'ssq'])

	#df.to_csv('SSQ_CHECK.csv')

	best_index = df.loc[df['ssq'].idxmin()]
	best_index['gap'] = gap
	best_index = best_index.round(2)
	result = best_index.to_dict()
	result['defects_start_date'] = pd.Timestamp(defects_start_date)
	result['features_start_date'] = pd.Timestamp(features_start_date)
	
	#best_index['defects_start_date'] = defects_start_date
	#best_index['features_start_date'] = features_start_date

	#print(final_data)

	#print(current_date)


	#time_from_P7 = P7_NUM - current_date

	#print(time_from_P7)

	#print(final_data['FEATURES'].values)

	feature_data = final_data['FEATURES'].dropna().values[int(best_index['index']):]

	#predict_range = np.asarray([i+1 for i in range(current_date, P7_NUM)])

	#print(len(feature_data))

	#print(len(predict_range))

	#exit()
	#print(final_data)
	#print(best_index)
	

	#x_new = best_index['intercept'] + best_index['slope'] * predict_range
	
	#print(x_new)


	#exit()

	#required_range = [i for i in predict_range if i > np.min(x_new) and i < np.max(x_new)]

	#print(required_range)

	y_new = best_index['ratio'] * feature_data

	x_new = [current_date+i for i in range(len(y_new))]

	#print(current_date)
	#print(feature_data)
	#print(y_new)


	

	#y_new = perform_interpolation(required_range, x_new, y_new)
	#x_new = required_range

	df = pd.DataFrame({'y_new': y_new, 'x_new': x_new})

	#print(df)


	#exit()

	

	final_data = final_data.merge(df, left_on='WEEK', right_on='x_new', how='outer')

	#print(final_data)

	#print(result)

	#final_data.to_csv("FINAl_DATA.csv")

	#print(len(final_data))

	#print(len(pd.date_range(start=defects_start_date, periods=len(df), freq=str(increments)+'D')))

	final_data['defect_dates'] = pd.date_range(start=defects_start_date, periods=len(final_data), freq=str(increments)+'D')

	final_data['feature_dates'] = pd.date_range(start=features_start_date, periods=len(final_data), freq=str(increments)+'D')


	result['dates'] = list(final_data['defect_dates'].append(final_data['feature_dates']).sort_values().drop_duplicates().astype(str).values)

	final_data['defect_dates'] = final_data['defect_dates'].astype(str)

	final_data['feature_dates'] = final_data['feature_dates'].astype(str)

	

	#print(final_data)


	#exit()


	#exit()

	#result['dates'] = list(set(list(final_data['defect_dates']) + list(final_data['feature_dates'])))





	result['predictions'] = final_data[['defect_dates', 'y_new']].rename(columns={'defect_dates': 'date', 'y_new':'value'}).dropna().to_dict(orient='records')
	result['features'] = final_data[['feature_dates', 'FEATURES']].rename(columns={'feature_dates': 'date', 'FEATURES':'value'}).dropna().to_dict(orient='records')
	result['actual'] = final_data[['defect_dates', 'ARRIVALS']].rename(columns={'defect_dates': 'date', 'ARRIVALS':'value'}).dropna().to_dict(orient='records')
	#print(features)

	#exit()

	#print(final_data)

	#print(best_index)

	#print(defects_start_date)
	#print(features_start_date)


	#exit()

	#p7_week = P7_NUM

	#P7_Prediction = perform_interpolation([p7_week], x_new, y_new)[0]

	#print(P7_Prediction)

	

	return result

	#print(final_data)
	#final_data.to_csv("FINAl_DATA.csv")


def get_ssq(arrivals, x_new, y_new):

	df1 = pd.DataFrame({'WEEK':[i+1 for i in range(len(arrivals))], 'ARRIVALS':arrivals})

	min_week = int(math.ceil(np.min(x_new)))

	max_week = int(math.floor(np.max(x_new)))

	week_range = [i for i in range(min_week, max_week+1)]

	#x_new = x_new[:len()]

	#print("k: ", len(week_range))

	#print(week_range)

	#print(x_new)

	#print("l: ", len(x_new))

	#print("m: ", len(y_new))

	new_values = perform_interpolation(week_range, x_new, y_new, roundoff=False)

	#print(new_values)
	#print(len(new_values))

	df2 = pd.DataFrame({'D2':week_range, 'ARRIVALS_(Y_NEW)':new_values})

	df = df1.merge(df2, how='outer', left_on='WEEK', right_on='D2')

	df['ERROR'] = (df['ARRIVALS'] - df['ARRIVALS_(Y_NEW)'])**2

	p = df.count()['ERROR']

	#print("p: ", p) 

	ssq = round(math.sqrt(df['ERROR'].sum()/(p-2)), 3)

	del df['D2']

	return ssq


# def determine_ssq(final_data):

# 	#final_data['ARRIVALS_NEW'] = c * final_data['ARRIVALS']
# 	#print(len())
	
# 	min_week = int(math.ceil(final_data['WEEK_(X_NEW)'].min()))
# 	max_week = final_data['WEEK'].max()

# 	week_range = [i for i in range(min_week, max_week+1)]
# 	#print(max_week, min_week, len(week_range), week_range)

# 	row_data = []
# 	if len(week_range) < len(final_data):
# 		diff = len(final_data) - len(week_range)
# 		row_data = [None for i in range(diff)]

# 	row_data += perform_interpolation(week_range, final_data['WEEK_(X_NEW)'].values, final_data['ARRIVALS_(Y_NEW)'].values, roundoff=False)

# 	#print(row_data)

# 	if len(row_data) < len(final_data):
# 		diff = len(final_data) - len(row_data)
# 		nones = [None for i in range(diff)]

# 	row_data += nones

# 	#print(len(row_data))
# 	#print(len(final_data))

# 	#exit()

# 	final_data['SHIFTED_Y'] = row_data

# 	final_data['(Y_ACT-Y_PRED)^2'] = final_data['ARRIVALS'] - final_data['SHIFTED_Y']

# 	final_data['(Y_ACT-Y_PRED)^2'] = final_data['(Y_ACT-Y_PRED)^2']**2



# 	p = final_data.count()['(Y_ACT-Y_PRED)^2']

# 	print("p: ", p)

# 	ssq = round(math.sqrt(final_data['(Y_ACT-Y_PRED)^2'].sum()/(p-2)), 3)

# 	#print(final_data)

# 	#print("SSQ: ", ssq)

# 	return ssq, final_data


def create_qq_plot(feature_data, arrival_data):

	# qq_plot_start = 5
	# qq_plot_end = 100
	# qq_plot_increment = 5
	# qq_plot_limit  = 0.3

	max_feature = np.nanmax(feature_data)
	max_defect = np.nanmax(arrival_data)

	FEATURES_CDF = (feature_data/max_feature).round(5)

	ARRIVALS_CDF = (arrival_data/max_defect).round(5)

	w = [(i/100) for i in range(qq_plot_start,qq_plot_end,qq_plot_increment) if ((i/100) > np.nanmin(FEATURES_CDF)) and ((i/100) > np.nanmin(ARRIVALS_CDF))]

	#print("w: ", w)

	#prinr("W: ", w)

	#print("CDF: ", FEATURES_CDF)
	Q_features = perform_interpolation(w, FEATURES_CDF, [i+1 for i in range(len(feature_data))], roundoff=False)

	Q_arrivals = perform_interpolation(w, ARRIVALS_CDF, [i+1 for i in range(len(arrival_data))], roundoff=False)

	#print(Q_arrivals)

	#print(Q_features)

	#exit()

	arrivals_95pct = perform_interpolation([0.95], ARRIVALS_CDF, arrival_data, roundoff=False)[0]

	features_95pct = perform_interpolation([0.95], FEATURES_CDF, feature_data, roundoff=False)[0]

	c = arrivals_95pct/features_95pct	#ratio

	QQ = pd.DataFrame([[i] for i in w], columns=['p'])

	#print(QQ)
	#print(Q_features)
	QQ['x'] = Q_features
	QQ['y'] = Q_arrivals
	QQ['xx'] = QQ['x']*QQ['x']
	QQ['xy'] = QQ['x']*QQ['y']

	#print(QQ)

	#print(QQ)

	QQ = QQ[QQ['p'] >= qq_plot_limit]

	#print(QQ)

	n = len(QQ)

	a = (QQ['y'].sum()*QQ['xx'].sum() - QQ['x'].sum()*QQ['xy'].sum())/(n*QQ['xx'].sum()-QQ['x'].sum()*QQ['x'].sum())  #intercept

	b = (n*(QQ['xy'].sum()) - QQ['x'].sum()*QQ['y'].sum())/(n*QQ['xx'].sum() - QQ['x'].sum()*QQ['x'].sum())	#slope


	#print("n: ", n)

	#print(("a: %f, b: %f, c: %f") %(a, b, c))

	return a, b, c



def read_data():

	feature_data = pd.read_csv(directory+'feature_data.csv')

	defect_data = pd.read_csv(directory+'defect_data.csv')

	feature_data['Month_Ending'] = pd.to_datetime(feature_data['Month_Ending'], format='%d/%m/%Y')

	defect_data['Date_Ending'] = pd.to_datetime(defect_data['Date_Ending'], format='%d/%m/%Y')

	feature_data = feature_data[feature_data['Sub-feature_Arrival'] >= min_number_features]

	feature_data = feature_data[feature_data['Month_Ending'] >= feautures_start_date]

	feature_data['Month'] = list(range(1,len(feature_data)+1))

	defect_data = defect_data[defect_data['Created'] >= min_number_defects]

	return feature_data, defect_data






def get_data(feature_new_days, defect_new_days, feature_new_data, 
		defect_new_data, resolved_new_data):

	data = []
	for i in range(0, len(feature_new_days)):
		row = []
		row.append(feature_new_days[i])
		row.append(feature_new_data[i])
		data.append(row)
	features = pd.DataFrame(data, columns=['DAY', 'FEATURES'])


	data = []
	for i in range(0, len(defect_new_days)):
		row = []
		row.append(defect_new_days[i])
		row.append(defect_new_data[i])
		row.append(resolved_new_data[i])
		data.append(row)
	defects = pd.DataFrame(data, columns=['DAY', 'ARRIVALS', 'CLOSURES'])


	df = features.merge(defects, left_on='DAY', right_on='DAY', how='outer')

	return df


def convert_p7_to_number(feature_data, defect_data, P7, k):

	defect_start = defect_data['Date_Ending'].values[0]

	feature_start = feature_data['Month_Ending'].values[0]

	min_ = min(defect_start, feature_start)

	diff = math.ceil((np.datetime64(P7) - min_).astype('timedelta64[D]').astype(int)/k)


	return diff



def perform_interpolation(u, x, y, roundoff=True):
	#print(u)
	#u = [i+1 for i in u]
	#print("u: ", u)
	#print("x: ", x)
	#print("y: ", y)
	v = []
	for i in range(0,len(u)):
		#print(i)
		found = False
		for j in range(1,len(x)):
			
			if u[i] >= x[j-1] and u[i] < x[j]:
				found = True
				value = y[j-1]+((y[j]-y[j-1])/(x[j]-x[j-1]))*(u[i]-x[j-1])
				try:
					if roundoff:
						v.append(math.ceil(value))
					else:
						v.append(round(value,2))
				except:
					v.append(np.nan)

		if found == False:
			print("The value: ", u[i], " cannot be found..........")
				#print()
				#print(i, j, x[j-1], u[i], x[j], v[i], len(v), v)
	#print(v)
	#exit()
	return v




if __name__ == '__main__':
	run_edpm(feature_data, defect_data, P7, inc, date, srange, qq_start, qq_end, qq_increment, qq_limit)