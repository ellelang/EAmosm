#data_folder = Path('/Users/ellelang/Documents/github/EAmosm/data')
from pathlib import Path
data_folder = Path('/Users/ellelang/Downloads')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import array
import random
import json
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import altair as alt

dat = pd.read_csv (data_folder/'model_summary.csv')
dat.head()
summary = dat
source = summary
base = alt.Chart(summary).encode(
   x = alt.X('model',title = '',
            sort=alt.EncodingSortField(field="percent_error", op = 'mean', order='ascending')
           ),
    y = alt.Y('percent_error',  title='',\
           axis=alt.Axis(grid=False, tickMinStep=0.01)
    )
)


bars = base.mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    color = alt.condition(
      "datum.percent_error > 0",  
      alt.value("steelblue"), # positive color
      alt.value("orange")     # negative color
    ),
    opacity = alt.value(0.9)
)

text = base.mark_text(
    align='center',
    baseline='top',
    dx = 2,
    dy = -8,
    fontSize=7
).encode(
   text= alt.Text('percent_error:Q', format=',.2r')
)

g1 = (bars + text).facet('eval_window:O', columns=3).properties(
    title='Mean Percent Errors within Evaluation Window').resolve_axis(
    x='independent',
    y='independent'
).resolve_scale(
    x='independent', 
    y='independent'
)
    
from altair_saver import save

save(g1, data_folder/"chart.vl.json") 
save(g1, data_folder/"chart.png")   
g1.save(data_folder/'mychart.png', scale_factor=2.0)
