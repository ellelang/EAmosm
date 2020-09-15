#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt


# In[2]:


##conda install -c conda-forge altair vega_datasets


# In[36]:


import altair as alt


# In[8]:


hbo_seg = pd.read_csv('output/seg_hbo.csv')
#hbo_seg_id = hbo_seg_info['base_seg_id'].tolist()


# In[84]:


name_list = ['cinemax','hbo', 'epix', 'cbs', 'starz', 'showtime', 'pbsmaster', 'tcrime', 'shudder', 'pbskid']
#name_list = ['cinemax','epix', 'cbs', 'starz', 'showtime', 'pbsmaster', 'tcrime', 'shudder', 'pbskid']


# In[85]:


file_list = []
seg_id = []
for i in name_list:
    filename  = 'output/seg_' + i + '.csv'
    segfile = pd.read_csv(filename)
    segfile['ad_name'] = i
    seg_id.append(segfile['base_seg_id'].tolist())
    file_list.append(segfile)


# In[86]:


intersect = list(set.intersection(*[set(x) for x in seg_id]))
intersect


# In[87]:


df_comp = pd.concat(file_list)
df_comp['intersect'] = df_comp['base_seg_id'].astype(str).isin(intersect)


# In[88]:


df_comp.head(3)


# In[78]:


CH1 = alt.Chart(df_comp[df_comp.ad_name =='cinemax']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y'),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'Cinemax' ,
             width=1000)


CH2 = alt.Chart(df_comp[df_comp.ad_name =='epix']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y'),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'HBO' ,
             width=1000)


CH3 = alt.Chart(df_comp[df_comp.ad_name =='cbs']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y'),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'CBS' ,
             width=1000)


CH4 = alt.Chart(df_comp[df_comp.ad_name =='starz']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y'),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'Starz' ,
             width=1000)

CH1&CH2&CH3&CH4


# In[89]:


my_theme = alt.themes.get()()  # Get current theme as dict.
my_theme.setdefault('encoding', {}).setdefault('color', {})['scale'] = {
    'scheme': 'bluepurple',
}
alt.themes.register('my_theme', lambda: my_theme)
alt.themes.enable('my_theme')


# In[90]:


start= df_comp.marg_imp.min()
end = df_comp.marg_imp.max()


# In[91]:


selector = alt.selection_single(on='mouseover', nearest=True, empty='all', fields=['base_seg_id'])


# In[92]:


base = alt.Chart(df_comp).mark_point(filled =  True).encode(
    alt.X('Coef_value'),
    alt.Y('cr'),
    size = alt.Size('impressions', scale=alt.Scale(domain=[100, 100000])),
    color = alt.Color('marg_imp', scale=alt.Scale(scheme='bluepurple', domain=[start, end])),
    tooltip = [alt.Tooltip('base_seg_id'),
               alt.Tooltip('Coef_value'),
               alt.Tooltip('marg_imp')
              ],
    opacity = alt.OpacityValue(0.7)
).properties(
    width=320,
    height=280
).facet(
    facet='ad_name:N',
    columns=2
).configure_axis(
    #grid=False
)

base


# In[61]:


df_hbo_run = pd.read_csv ('output/Runtime_HBO.csv')
df_hbo_run = df_hbo_run[df_hbo_run.Method!= 'Benchmark']


# In[64]:


a1 = alt.Chart(df_hbo_run).mark_line(strokeWidth=3).encode(
    alt.X('N', title= 'Iteration of Cross-Validation'),
    alt.Y('Overlap',scale=alt.Scale(domain=[0.5, 0.8])),
    color = alt.Color('Method', scale = alt.Scale(scheme="dark2"))
)

a2 = alt.Chart(df_hbo_run).mark_bar().encode(
    x = alt.X('N:O', title= 'Iteration of Cross-Validation'),
    y = alt.Y('Second:Q', title= 'Runtime (second)'),
    color = alt.Color('Method', scale = alt.Scale(scheme="dark2")),
)

a1 | a2


# In[ ]:




