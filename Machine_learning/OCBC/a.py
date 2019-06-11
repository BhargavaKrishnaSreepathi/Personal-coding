from pytrends import request
import pandas as pd
import numpy as np

dfs = {}
for i in dates:  # weekly dates file where column are years
    dfs[i] = {}
    for j in dates[i]:  # each week in the year
        if type(j) == float and math.isnan(j):
            continue
        endpoints = j.split(" ")
        key = endpoints[0] + "T00 " + endpoints[1] + "T00"
        p = request.TrendReq(guser, gpas, hl='en-US')
        p.build_payload([keyword], timeframe=key)
        dfs[i][j] = p.interest_over_time()