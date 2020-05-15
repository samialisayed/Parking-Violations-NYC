# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:32:55 2020

@author: sami
"""

from pyspark import SparkContext
import sys


def FilterVio(partId, records):
    if partId==0:
        next(records)
    import csv
    import re
    reader = csv.reader(records)
    for row in reader:
        if (row[23] not in (None, "") and row[23][-1].isnumeric() and 
            row[24] not in (None, "") and row[21] not in (None, "") and row[4] not in (None, "")):# to filter our bad data
            
            if row[21] in ['MAN','MH','MN','NEWY','NEW Y','NY']:
                county = '1'
            elif row[21] in ['BRONX','BX','PBX']:
                county = '2'
            elif row[21] in ['BK','K','KING','KINGS']:
                county = '3'
            elif row[21] in ['Q','QN','QNS','QU','QUEEN']:
                county = '4'
            elif row[21] in ['R','RICHMOND']:
                county = '5'
            
            if str(row[4].split('/')[-1]) in ['2015','2016','2017','2018','2019']:
                ticket = int(row[0])
                year = str(row[4].split('/')[-1])
                house = re.sub('[^0-9-]','',row[23])
                house = tuple(map(int, house.replace('--','-').split('-')))
                street = row[24].lower()
                yield ((street,county),(year,house,ticket))
            
def FilterCent(partId, records):
    if partId==0:
        next(records)
    import csv
    reader = csv.reader(records)
    for row in reader:
        if ((row[2] not in (None, "") and row[3] not in (None, "")) or 
            (row[4] not in (None, "") and row[5] not in (None, ""))):
            
            ID = int(row[0])
            L_low = tuple(map(int, row[2].split('-')))
            L_high = tuple(map(int, row[3].split('-')))
            R_low = tuple(map(int, row[4].split('-')))
            R_high = tuple(map(int, row[5].split('-')))
            Full_street = row[28].lower()
            ST_label = row[10].lower()
            boro = row[13]
            yield ((Full_street,boro),(ID,L_low,L_high,R_low,R_high,ST_label))

def collecting(records):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    for record in records:
        count15 = 0
        count16 = 0
        count17 = 0
        count18 = 0
        count19 = 0
        
        ID = record[0]
        years = record[1].keys()
        
        if '2015' in years:
            count15 = record[1]['2015']
        if '2016' in years:
            count16 = record[1]['2016']
        if '2017' in years:
            count17 = record[1]['2017']
        if '2018' in years:
            count18 = record[1]['2018']
        if '2019' in years:
            count19 = record[1]['2019']
            
        lr = LinearRegression()
        lr.fit(np.array((2015,2016,2017,2018,2019)).reshape(-1,1), (count15 ,count16, count17, count18, count19))
        slope = round(lr.coef_[0],3)
        
        yield(ID, (count15 ,count16, count17, count18, count19, slope))
        
def allid(partId, records):
    if partId==0:
        next(records)
    import csv
    reader = csv.reader(records)
    for row in reader:
        yield(int(row[0]),(0, 0, 0, 0, 0, 0))
        
        
def to_csv(x):
    import csv, io
    output = io.StringIO("")
    csv.writer(output).writerow(x)
    return output.getvalue().strip()
    
    
if __name__ == '__main__':
    sc = SparkContext()
    vio = sc.textFile('/data/share/bdm/nyc_parking_violation/*.csv')
    cent = sc.textFile('/data/share/bdm/nyc_cscl.csv')
    vio_rdd = vio.mapPartitionsWithIndex(FilterVio).distinct()
    cent_rdd = cent.mapPartitionsWithIndex(FilterCent).distinct()
    full = vio_rdd.join(cent_rdd)\
         .filter(lambda x: ((x[1][0][1][-1] % 2 != 0 and x[1][1][1] <= x[1][0][1] <= x[1][1][2] and 
                             len(x[1][1][1]) == len(x[1][0][1]) == len(x[1][1][2]) and x[1][1][1] != (0,) and
                            x[1][1][2] != (0,) and x[1][1][3] != (0,) and x[1][1][4] != (0,)) or
                           (x[1][0][1][-1] % 2 == 0 and x[1][1][3] <= x[1][0][1] <= x[1][1][4] and 
                             len(x[1][1][3]) == len(x[1][0][1]) == len(x[1][1][4]) and x[1][1][1] != (0,) and
                            x[1][1][2] != (0,) and x[1][1][3] != (0,) and x[1][1][4] != (0,))))\
         .distinct()\
         .map(lambda x: (x[1][0][2],(x[1][1][0],x[1][0][0])))\
         .reduceByKey(lambda x,y: x+y).map(lambda x: ((x[1][0],x[1][1]),1))
    
    cent_label = cent_rdd.map(lambda x: ((x[1][5],x[0][1]),(x[1][0],x[1][1],x[1][2],x[1][3],x[1][4],x[0][0])))\
        .filter(lambda x: x[0][0] != x[1][5]).distinct()
    
    label = vio_rdd.join(cent_label)\
         .filter(lambda x: ((x[1][0][1][-1] % 2 != 0 and x[1][1][1] <= x[1][0][1] <= x[1][1][2] and 
                             len(x[1][1][1]) == len(x[1][0][1]) == len(x[1][1][2]) and x[1][1][1] != (0,) and
                            x[1][1][2] != (0,) and x[1][1][3] != (0,) and x[1][1][4] != (0,)) or
                           (x[1][0][1][-1] % 2 == 0 and x[1][1][3] <= x[1][0][1] <= x[1][1][4] and 
                             len(x[1][1][3]) == len(x[1][0][1]) == len(x[1][1][4]) and x[1][1][1] != (0,) and
                            x[1][1][2] != (0,) and x[1][1][3] != (0,) and x[1][1][4] != (0,))))\
         .distinct()\
         .map(lambda x: (x[1][0][2],(x[1][1][0],x[1][0][0])))\
         .reduceByKey(lambda x,y: x+y).map(lambda x: ((x[1][0],x[1][1]),1))
         
    all = (full + label).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0][0],(x[0][1],x[1])))\
                    .groupByKey().sortBy(lambda x: x[0]).mapValues(dict)
                    
    matchedID = all.mapPartitions(collecting)
    
    allid = cent.mapPartitionsWithIndex(allid)
    
    final = (matchedID + allid).distinct().groupByKey().mapValues(lambda x: sorted(list(x))[-1]).sortByKey()\
                    .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5]))
                    
    final.map(to_csv)\
        .saveAsTextFile(sys.argv[1])
    
    
    
    
    