#!/usr/bin/env python
# coding: utf-8

# # The Battle of the Neighborhoods - A clustering approach to understanding business popularity in London.

# London is one of the world's most popular tourist destinations, with a plethora of well-known tourist attractions. This city is well-known for its large range of beautiful sites to visit, not just on weekends but also throughout the week. Despite the fact that there are hundreds of such locations around the city, investors have plenty of options. Understanding the characteristics of these well-known locations may provide investors with insight into potential business prospects in a certain London neighbourhood. In order to choose the ideal business opportunity, the best location must be combined with a thorough awareness of the environment.
# As a result, this study used a clustering method to categorise all of London's boroughs, then went on to analyse the top common places in each cluster based on their popularity, giving investors a glimpse at a wide range of top business selections for different areas and less popular ones with the potential to do well in all of them. Tourists may also use this information to choose which locations of London to visit depending on their interests.

# The k-means clustering of all the boroughs in London was used in this study to look for common venues and companies in distinct clusters of London boroughs. Three clusters were utilised to divide boroughs into three distinct groupings, each with its own set of characteristics. The groupings were shown to have a link with the region's population density, with each of the three clusters mostly belonging to the low, mid, or high population density categories. The popularity of venues in each cluster was found to be uniquely matching to the features observed in the region using data received from the foursquare API of the popular place.
# This will provide both investors and tourists with an overview of business prospects and tourism attractions in various London zones. Despite the fact that this study has provided several business ideas, it is limited since an ideal site with a very good estimate is required. As a result, further study is required to determine the specific location of the proposed enterprises in each zone.

# In[3]:


import numpy as np 

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json 

get_ipython().system('conda install -c conda-forge geopy --yes')
get_ipython().system('conda install -c conda-forge geocoder --yes')
from geopy.geocoders import Nominatim
import geocoder

import requests
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans

get_ipython().system('conda install -c conda-forge folium --yes')
import folium 

get_ipython().system('conda install -c conda-forge lxml --yes')
import lxml.html as lh 
import urllib.request

get_ipython().system('conda install -c anaconda beautifulsoup4 --yes')
from bs4 import BeautifulSoup

get_ipython().system('conda install -c conda-forge wordcloud --yes')
from wordcloud import WordCloud, STOPWORDS

print('Libraries imported.')


# # Scraping data from Wikipedia

# In[4]:


url = requests.get('https://en.wikipedia.org/wiki/List_of_London_boroughs').text
soup = BeautifulSoup(url,'lxml')
table = soup.find('table')
df_r = pd.read_html(str(table))
data = pd.read_json(df_r[0].to_json(orient='records'))
data = data.iloc[1:]


# In[5]:


data.head()


# # Preprocessing of the data

# In[6]:


new_data = data.iloc[:,[0,6,7]]
new_data.columns = ['Borough', 'Area', 'Population']
density = new_data['Population']/new_data['Area']
new_data = new_data.assign(Density=density.values)
new_data['Borough'] = new_data['Borough'].str.replace('\[note 2]', '')
new_data['Borough'] = new_data['Borough'].str.replace('\[note 4]', '')
new_data['Borough'] = new_data['Borough'].str.replace(' +$', '', regex = True)


# # Extraction of the coordinates

# In[7]:


column_names = ['Borough','Latitude', 'Longitude'] 
co_ord = pd.DataFrame(columns=column_names)
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='myapplication')
for i in range(1,32):
    city = new_data['Borough'][i]
    location = geolocator.geocode(city + " UK")
    lat = location.raw["lat"]
    long = location.raw["lon"]
    co_ord = co_ord.append({'Borough':location, 'Latitude': lat,'Longitude': long}, ignore_index=True)
print(co_ord.shape)


# In[8]:


co_ord.head()


# # Merging the data frame and correcting the wrong coordinates

# In[9]:


new_data["Latitude"] = co_ord.iloc[:,1].astype(float).values
new_data["Longitude"] = co_ord.iloc[:,2].astype(float).values
new_data.at[28, 'Latitude'] = 51.5203
new_data.at[28, 'Longitude'] = 0.0293
new_data.at[29, 'Latitude'] = 51.5886
new_data.at[29, 'Longitude'] = 0.0118
new_data.head()


# # To Verify the coordinates on a map

# In[10]:


#first the London coordinate needs to be accessed
address = 'London, UK'
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of London are {}, {}.'.format(latitude, longitude))
map_london = folium.Map(location=[latitude, longitude], zoom_start=10.5)

# Then adding markers to the map
for lat, lng, label in zip(new_data['Latitude'], new_data['Longitude'], new_data['Borough']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=8,
        popup=label,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(map_london)


# In[11]:


map_london


# # To extract the top venues from foursquare

# In[12]:


CLIENT_ID = 'LWSCQ2P00NUQ4LET3V51GVNIK4R4NFS3C0NWBZ0QYPLVT4R2' # your Foursquare ID
CLIENT_SECRET = 'YO55MQ0KYRQBKHLTWA35LSTD1MUUM05PJJ4UXUHY5VWCOVX3' # your Foursquare Secret
VERSION = '20210205' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[13]:


def getNearbyVenues(names, latitudes, longitudes, radius=5000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Borough', 
                  'Borough Latitude', 
                  'Borough Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

london_venues = getNearbyVenues(names=new_data['Borough'],
                                   latitudes=new_data['Latitude'],
                                   longitudes=new_data['Longitude']
                                  )


# In[14]:


print(london_venues.shape)
london_venues.head()


# # Venue's preprocessing

# In[15]:


london_venues.groupby('Borough').count()
print('There are {} uniques categories.'.format(len(london_venues['Venue Category'].unique())))
# one hot encoding
london_onehot = pd.get_dummies(london_venues[['Venue Category']], prefix="", prefix_sep="")
# add Borough column back to dataframe
london_onehot['Borough'] = london_venues['Borough'] 
# move Borough column to the first column
fixed_columns = [london_onehot.columns[-1]] + list(london_onehot.columns[:-1])
london_onehot = london_onehot[fixed_columns]
print(london_onehot.shape)
london_grouped = london_onehot.groupby('Borough').mean().reset_index()
print(london_grouped.shape)


# In[16]:


num_top_venues = 10

for hood in london_grouped['Borough']:
    print("----"+hood+"----")
    temp = london_grouped[london_grouped['Borough'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[17]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Borough']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
boroughs_venues_sorted = pd.DataFrame(columns=columns)
boroughs_venues_sorted['Borough'] = london_grouped['Borough']

for ind in np.arange(london_grouped.shape[0]):
    boroughs_venues_sorted.iloc[ind, 1:] = return_most_common_venues(london_grouped.iloc[ind, :], num_top_venues)


# In[18]:


boroughs_venues_sorted.head()


# # implimenting K-means Clustering

# In[21]:


# setting number of clusters = 3
kclusters = 3
london_grouped_clustering = london_grouped.drop('Borough', 1)
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(london_grouped_clustering)

# checking cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[23]:


# add clustering labels
boroughs_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
london_merged = new_data
london_merged = london_merged.join(boroughs_venues_sorted.set_index('Borough'), on='Borough')


# In[24]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(london_merged['Latitude'], london_merged['Longitude'], london_merged['Borough'], london_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=20,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color= 'black',
        fill_opacity= 0.5).add_to(map_clusters)


# In[25]:


map_clusters


# In[26]:


#setting each cluster to a variable
london_cluster_1 = london_merged.loc[london_merged['Cluster Labels'] == 0, london_merged.columns[[0] + list(range(5, london_merged.shape[1]))]]
london_cluster_2 = london_merged.loc[london_merged['Cluster Labels'] == 1, london_merged.columns[[0] + list(range(5, london_merged.shape[1]))]]
london_cluster_3 = london_merged.loc[london_merged['Cluster Labels'] == 2, london_merged.columns[[0] + list(range(5, london_merged.shape[1]))]]


# In[27]:


#formatting new dataframe for vuisualisation
new_clus = london_merged.iloc[:, 6:17]
df3 = pd.concat([new_data,new_clus], axis=1)
df_density = df3.sort_values('Density')
df_cluster = df3.sort_values('Cluster Labels')


# # The density Population

# In[28]:


clrs = ['red' if (t == 0) else 'purple' if (t == 1) else 'green' for t in df_density['Cluster Labels']]
clr = ['red' if (t == 0) else 'purple' if (t == 1) else 'green' for t in df_cluster['Cluster Labels']]

ax = df_density['Density']
ay = df_density['Borough']

ax_1 = df_cluster['Density']
ay_1 = df_cluster['Borough']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
axes[0].barh(ay, ax, color = clrs)
axes[1].barh(ay_1, ax_1, color = clr)
axes[0].set_title("Population Density of Boroughs \n (Descending Order/ Coloured Clustered)", fontsize=30)
axes[1].set_title("Population Density of Boroughs \n (Grouped/Coloured by Clusters)", fontsize=30)

axes[0].set_ylabel('Boroughs', fontsize=30)
axes[0].set_xlabel('Population Density',  fontsize=30)
axes[1].set_xlabel('Population Density',  fontsize=30)
axes[1].get_yaxis().set_visible(False)

for label in (axes[0].get_yticklabels()+ axes[0].get_xticklabels() + axes[1].get_xticklabels()):
    label.set_fontsize(20)
         
fig.tight_layout()
fig.savefig('density.png')


# In[29]:


#extracting all words from each cluster
df_word_1 = london_cluster_1.iloc[:, 3:13]
fd_1 = pd.concat([df_word_1, df_word_1.unstack().reset_index(drop=True).rename('All words')], axis=1)
word_1 = '+ '.join([i for i in fd_1['All words']])

df_word_2 = london_cluster_2.iloc[:, 3:13]
fd_2 = pd.concat([df_word_2, df_word_2.unstack().reset_index(drop=True).rename('All words')], axis=1)
word_2 = '+ '.join([str(i) for i in fd_2['All words']])

df_word_3 = london_cluster_3.iloc[:, 3:13]
fd_3 = pd.concat([df_word_3, df_word_3.unstack().reset_index(drop=True).rename('All words')], axis=1)
word_3 = '+ '.join([i for i in fd_3['All words']])


# In[30]:


# instantiate a word cloud object
graph_instance = WordCloud(
    background_color='white',
    regexp=r"\w[\w' ]+",
   collocations=False,
   width=3000, height=3000
)


# # Results for the first cluster

# In[31]:


graph_instance.generate(word_1)
clr_1 = ['red' if (t == 0) else 'gray' for t in df_density['Cluster Labels']] 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(28, 14))
ax = df_density['Density']
ay = df_density['Borough']

axes[1].imshow(graph_instance, interpolation='bilinear')
axes[1].axis('off')
axes[0].barh(ay,ax, color=clr_1)
axes[1].set_title("Frequency of Venues", fontsize=40)
axes[0].set_title("Corresponding Boroughs in red (Cluster 1)", fontsize=40)
axes[0].set_ylabel('Boroughs', fontsize=40)
axes[0].set_xlabel('Population Density',  fontsize=40)
for label in (axes[0].get_yticklabels()+ axes[0].get_xticklabels()):
    label.set_fontsize(25)
fig.tight_layout()
fig.savefig("cluster_1.png")


# # Results for the second cluster

# In[32]:


graph_instance.generate(word_2)
clr_2 = ['red' if (t == 1) else 'gray' for t in df_density['Cluster Labels']] 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(28, 14))
ax = df_density['Density']
ay = df_density['Borough']

axes[1].imshow(graph_instance, interpolation='bilinear')
axes[1].axis('off')
axes[0].barh(ay,ax, color=clr_2)
axes[1].set_title("Frequency of Venues", fontsize=40)
axes[0].set_title("Corresponding Boroughs in red (Cluster 2)", fontsize=40)
axes[0].set_ylabel('Boroughs', fontsize=40)
axes[0].set_xlabel('Population Density',  fontsize=40)
for label in (axes[0].get_yticklabels()+ axes[0].get_xticklabels()):
    label.set_fontsize(25)
fig.tight_layout()
fig.savefig("cluster_2.png")


# # Results for the third cluster

# In[33]:


graph_instance.generate(word_3)
clr_3 = ['red' if (t == 2) else 'gray' for t in df_density['Cluster Labels']] 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(28, 14))
ax = df_density['Density']
ay = df_density['Borough']

axes[1].imshow(graph_instance, interpolation='bilinear')
axes[1].axis('off')
axes[0].barh(ay,ax, color=clr_3)
axes[1].set_title("Frequency of Venues", fontsize=40)
axes[0].set_title("Corresponding Boroughs in red (Cluster 3)", fontsize=40)
axes[0].set_ylabel('Boroughs', fontsize=40)
axes[0].set_xlabel('Population Density',  fontsize=40)
for label in (axes[0].get_yticklabels()+ axes[0].get_xticklabels()):
    label.set_fontsize(25)
fig.tight_layout()
fig.savefig("cluster_3.png")


# In[ ]:




