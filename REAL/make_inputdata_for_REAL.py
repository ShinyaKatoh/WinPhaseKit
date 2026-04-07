#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import datetime
import math


# In[7]:


prob_thres = 0.7

sf = 250

reference_time = datetime.datetime(2015, 1, 1)

year = ["15"]

mo = ["01"]

day_list = ["01"]

day1 = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]
day2 = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
day3 = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28"]


for i in range(len(day_list)):

    file = "pick_" + str(sf) + "Hz_1501" + day_list[i] + ".csv"
    
    print(file)

    data = pd.read_csv(file)

    fname = np.sort(data['fname'].values)
    
    # print(type(fname))

    for l in range(len(fname)):

        station = fname[l][12:].replace('.npz','')
        
        # print(fname[l])
        
        year = fname[l][:2]
        month = fname[l][2:4]
        day = fname[l][4:6]
        hour = fname[l][6:8]
        minu = fname[l][9:11]
        
        # print(year, month, day, hour, minu)

        out_P_path = "./for_REAL/WI." + station + ".P.txt"
        out_S_path = "./for_REAL/WI." + station + ".S.txt"

        out_P = open(out_P_path, 'a')
        out_S = open(out_S_path, 'a')

        p_idx = data['itp'][l]
        p_prob = data['tp_prob'][l]
        s_idx = data['its'][l]
        s_prob = data['ts_prob'][l]

        pp_idx = p_idx[1:-1].split(' ')
        pp_prob = p_prob[1:-1].split(' ')
        ss_idx = s_idx[1:-1].split(' ')
        ss_prob = s_prob[1:-1].split(' ')
   

        for m in range(len(pp_idx)):
        
            if len(pp_idx[0]) > 1:

                if float(pp_prob[m]) >= prob_thres:

                    microsecond, second = math.modf((int(pp_idx[m])-1)/sf)

                    p_arrival = datetime.datetime(2000+int(year), int(month), int(day), int(hour), int(minu), int(second), int(microsecond*10**6) )
                    lapse_time = (p_arrival - reference_time).total_seconds()

                    out = "{:.2f} {:} {:}\n".format(lapse_time,pp_prob[m],0.0)
                    out_P.write(out)

        out_P.close()

        for m in range(len(ss_idx)):
            
            if len(ss_idx[0]) > 1:

                if float(ss_prob[m]) >= prob_thres:
                    
                    microsecond, second = math.modf((int(ss_idx[m])-1)/sf)

                    s_arrival = datetime.datetime(2000+int(year), int(month), int(day), int(hour), int(minu), int(second), int(microsecond*10**6) )
                    lapse_time = (s_arrival - reference_time).total_seconds()

                    out = "{:.2f} {:} {:}\n".format(lapse_time,ss_prob[m],0.0)
                    out_S.write(out)

        out_S.close()


# In[ ]:




