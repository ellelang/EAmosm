#!/usr/bin/env python
# coding: utf-8

# In[211]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt


# In[212]:


import altair as alt


# In[ ]:





# In[ ]:





# In[227]:


name_list = ['hbo','cmax']
#name_list = ['cmax','epix', 'cbs', 'showtime', 'pbsmaster', 'tcrime', 'shudder', 'sundance']


# In[228]:


file_list = []
seg_id = []
for i in name_list:
    filename  = 'data_2019/' + i + '_detarget.csv'
    segfile = pd.read_csv(filename)
    segfile['ad_name'] = i
    seg_id.append(segfile['base_seg_id'].tolist())
    file_list.append(segfile)


# In[229]:


intersect = list(set.intersection(*[set(x) for x in seg_id]))
len(intersect)


# In[230]:


df_comp = pd.concat(file_list)
df_comp['intersect'] = df_comp['base_seg_id'].astype(str).isin(intersect)


# In[231]:


df_comp[df_comp.ad_name == 'hbo'].shape


# In[232]:


df_comp['intersect']


# In[234]:


CH1 = alt.Chart(df_comp[df_comp.ad_name =='cmax']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y', axis=alt.Axis(labels=False)),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'Cinemax' ,
             width=1000)


CH2 = alt.Chart(df_comp[df_comp.ad_name =='hbo']).mark_bar().encode(
    x=alt.X('base_seg_id:N', sort = 'y', axis=alt.Axis(labels=False)),
    y=alt.Y('Coef_value:Q', title='coefficient'),
    color=alt.condition(
        alt.datum.intersect == True, 
        alt.value('orange'),     
        alt.value('steelblue')
    )
).properties(title= 'HBO' ,
             width=1000)


# CH3 = alt.Chart(df_comp[df_comp.ad_name =='starz']).mark_bar().encode(
#     x=alt.X('base_seg_id:N', sort = 'y', axis=alt.Axis(labels=False)),
#     y=alt.Y('Coef_value:Q', title='coefficient'),
#     color=alt.condition(
#         alt.datum.intersect == True, 
#         alt.value('orange'),     
#         alt.value('steelblue')
#     )
# ).properties(title= 'CBS' ,
#              width=1000)


# CH4 = alt.Chart(df_comp[df_comp.ad_name =='showtime']).mark_bar().encode(
#     x=alt.X('base_seg_id:N', sort = 'y', axis=alt.Axis(labels=False)),
#     y=alt.Y('Coef_value:Q', title='coefficient'),
#     color=alt.condition(
#         alt.datum.intersect == True, 
#         alt.value('orange'),     
#         alt.value('steelblue')
#     )
# ).properties(title= 'showtime' ,
#              width=1000)

CH1&CH2
# CH1&CH2&CH3&CH4


# In[222]:


my_theme = alt.themes.get()()  # Get current theme as dict.
my_theme.setdefault('encoding', {}).setdefault('color', {})['scale'] = {
    'scheme': 'blue',
}
alt.themes.register('my_theme', lambda: my_theme)
alt.themes.enable('my_theme')


# In[223]:


start= (df_comp.marg_imp.min())
end = df_comp.marg_imp.max()/4


# In[224]:


selector = alt.selection_single(on='mouseover', nearest=True, empty='all', fields=['base_seg_id'])


# In[225]:


#df_comp = df_comp[df_comp['intersect'] == True]


# In[235]:


base = alt.Chart(df_comp).mark_point(filled =  True).encode(
    alt.X('Coef_value'),
    alt.Y('cr'),
    size = alt.Size('impressions', scale=alt.Scale(domain=[100000, 10000000])),
    color = alt.Color('marg_imp', scale=alt.Scale(scheme='purples', domain=[start, end])),
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


# In[ ]:




