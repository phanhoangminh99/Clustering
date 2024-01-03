# Clustering Project
## I. Introduce Dataset

Dataset show average life expectancy of countries and relate variables(2020):

![image](https://user-images.githubusercontent.com/110837675/202417099-cd649697-1fdb-431f-85fe-ca947791a05b.png)

With this dataset, I will Subgroups of countries by GDP per capital.

## II. Data Processing And Visualization
   #### A. Data Processing
   
  Check columns to see if any have NAN values?
```php
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/110837675/202418356-61a2f6ee-b08a-4b4c-b2a8-f190e079981d.png)

Data doesn't have NAN values.

  #### B. Visualization
  Data visualization by scatter chart to see the correlation of the Life Expectancy (Year) column with other columns.
  ```php
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
sns.scatterplot(ax=axes[0, 0], data=df, x='CO2 emissions (Billion tons)', y='Life Expectancy (Year)', s=200, alpha=0.7,color='red')
axes[0,0].grid(True)
sns.scatterplot(ax=axes[0, 1], data=df, x='GDP per capita ($)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='green')
axes[0,1].grid(True)
sns.scatterplot(ax=axes[0, 2], data=df, x='Rate of using basic drinking water (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='black')
axes[0,2].grid(True)
sns.scatterplot(ax=axes[1, 0], data=df, x='Obesity among adults (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='pink')
axes[1,0].grid(True)
sns.scatterplot(ax=axes[1, 1], data=df, x='Beer consumption per capita (Liter)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='brown');
axes[1,1].grid(True)
```
![image](https://user-images.githubusercontent.com/110837675/202599885-02d4a5f2-bd12-4660-b409-b985656d59d3.png)

```php
plt.figure(figsize=(20,10))
plt.title('CORRELATION ')
corr1= df.corr()
sns.heatmap(corr1, square= True, annot= True, fmt= '.2f', annot_kws= {'size':16}, cmap='Blues', linecolor='white', linewidths=0.5);
```
![image](https://user-images.githubusercontent.com/110837675/202419617-d2e6541c-452e-4abf-9d05-edb7a0e2ec02.png)

### III. Data Processing

I will drop Country columns. Because this columns contains word.

```php
data= df.drop(['Country'], axis='columns')
df1= data.values
```
Next, to make sure that the model have a good result, we must scaler data about the same range of values 0 and 1. I will use MinmaxScaler for data.
```php
from sklearn.preprocessing import StandardScaler
st= StandardScaler()
x= st.fit_transform(df1)
data_scaled= pd.DataFrame(x, columns=data.columns)
data_scaled
```
I will find the number of best match groups using elbow method and Silhouette_score.
 - ELBOW
 ```php
 k_war= {
'init':'random',
'n_init': 10,
'max_iter': 300,
'random_state': 42
}

sse= []
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, **k_war)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE");
```
![image](https://user-images.githubusercontent.com/110837675/202424684-b82b278c-51ad-499f-9e23-ee4431dbb0f3.png)

The slope of the downward curve is very uniform, making it very difficult to choose the number of groups. I will draw Silhouette_score method.
- Silhouette_score.
```php
silhouette= []
for i in range(2,11):
    kmeans= KMeans(n_clusters=i, **k_war)
    kmeans.fit(data_scaled)
    silhouette_sc= silhouette_score(data_scaled, kmeans.labels_)
    silhouette.append(silhouette_sc)

plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
plt.plot(range(2,11), silhouette)
plt.xticks(range(2,11))
plt.xlabel("Number of Clusters")
plt.ylabel("silhouette_score");
```
![image](https://user-images.githubusercontent.com/110837675/202426019-43c3fdad-b33e-4762-8165-aaecb2d19318.png)

Theoretically, K=2 should be chosen because it has the highest value. But from k=2 to k=3, the slope is very large, so I choose k=3 and compare the elbow, k=3 is also in the downward slope of the elbow.

Besides, I wil use KneeLocator to determine elbow.

![image](https://user-images.githubusercontent.com/110837675/202427410-75094ca4-4caa-432c-ad19-cdde83aba1ec.png)

The results is k=3.

### IV. Build Model.

I will use K-MEANS, Hierarchical Clustering, Probabilistic Clustering to compare with each other.

  #### A. K-MEANS.
  
  ```php
  kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(data_scaled)
  
plt.figure(figsize=(12,6))
plt.scatter(data_scaled[y_kmeans==0,0], data_scaled[y_kmeans==0,1], s=100, c='yellow', label='cluster 1')
plt.scatter(data_scaled[y_kmeans==1,0], data_scaled[y_kmeans==1,1], s=100, c='green', label='cluster2')
plt.scatter(data_scaled[y_kmeans==2,0], data_scaled[y_kmeans==2,1], s=100, c='purple', label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,2], s=300, c='black', label='centroid')
plt.title('Clusters Country (K-MEANS)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202428409-7f4614d6-2d63-4130-afe9-7a9f34705492.png)

This is k=3, now I draw k=4 to compare with k=3.

```php
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_scaled)
  
plt.figure(figsize=(12,6))
plt.scatter(data_scaled[y_kmeans==0,0], data_scaled[y_kmeans==0,1], s=100, c='yellow', label='cluster 1')
plt.scatter(data_scaled[y_kmeans==1,0], data_scaled[y_kmeans==1,1], s=100, c='green', label='cluster2')
plt.scatter(data_scaled[y_kmeans==2,0], data_scaled[y_kmeans==2,1], s=100, c='purple', label='cluster3')
plt.scatter(data_scaled[y_kmeans==3,0], data_scaled[y_kmeans==3,1], s=100, c='blue', label='cluster4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,2], s=300, c='black', label='centroid')
plt.title('Clusters Country (K-MEANS)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202428919-8c3b0daf-8d6c-4107-87be-a8b606a24f1a.png)

Comparing 2 drawings, k=3 is more reasonable.

I will add the clustering column to the data and plot them on the map.
```php
df['class']= kmeans.labels_

import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='class', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(K-MEANS)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202429948-77531a26-0d0b-40c3-b2e1-b6172c0d252b.png)

  #### B.Hierarchical Clustering.
  
  I will plot dendrogram to determine cluster.
  
  ```php
  plt.figure(figsize = (20,10))
mergings = sch.linkage(data_scaled, method="complete", metric='euclidean')
sch.dendrogram(mergings)
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202430470-ff68c8cc-7515-4419-9c0b-9733d137cbb1.png)

The dendrogram is divided into 3 clusters.

```
agg= AgglomerativeClustering(n_clusters=3, linkage='ward',affinity='euclidean')
agg.fit_predict(data_scaled)

plt.figure(figsize=(12,6))
plt.scatter(data_scaled['Life Expectancy (Year)'], data_scaled['GDP per capita ($)'], c=agg.labels_, s=200);
plt.title('Clusters Country (Hierarchical Clustering)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202430948-6ea9199e-e39b-4fe3-8ed3-f5aee6844aac.png)

```php
df['class']= agg.labels_

import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='class', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(Hierarchical Clustering)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431141-ddeeae38-d48e-4e9a-aec6-244516470d67.png)

  #### C.Probabilistic Clustering
```php
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(data_scaled)

#predictions from gmm
labels = gmm.predict(data_scaled)
df['cluster'] = labels

plt.figure(figsize=(12,6))
plt.scatter(data_scaled['Life Expectancy (Year)'], data_scaled['GDP per capita ($)'], c=df.cluster, s=200);
plt.title('Clusters Country (Probabilistic Clustering)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431471-d73f56c6-41e8-4d55-ae3b-99138d4f5a23.png)

```php
import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='cluster', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(Probabilistic Clustering)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431596-53e8f5c4-9fa5-4a56-89ee-d3c9be57c033.png)

### V. Compare model.

![image](https://user-images.githubusercontent.com/110837675/202431821-225c239f-1a7d-4ffa-b2df-2ceb20d68206.png)

Grouping with K-means is the most reasonable.

![image](https://user-images.githubusercontent.com/110837675/202432313-372981fa-bebb-4d02-ae3f-473a24394dbf.png)

### VI. Analysis after grouped Country follow GDP.
```php
plt.figure(figsize=(12,6))
plt.grid(True)
sns.violinplot(data=df, x='class', y='Life Expectancy (Year)');
```
![image](https://user-images.githubusercontent.com/110837675/202433370-8724c666-20a5-4eb9-8fa4-9cfe55272e21.png)

Group 2 had the highest life expectancy at about 83 years. Group 1 accounts for a large amount but the average life expectancy is very low.

```php
import plotly.express as px
fig = px.scatter(df, x = 'CO2 emissions (Billion tons)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between CO2 emissions and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202435585-fbbc764a-8522-4892-809c-7e261bff89af.png)

Group 2 high-income countries (yellow) have a long life expectancy and high CO2 emissions. Poor countries have a lower life expectancy than rich countries, and emit less CO2.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Rate of using basic drinking water (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Rate of using basic drinking water (%) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436088-71800f4e-8165-496b-aead-67f564b2c2b7.png)

High-income countries have very high access to clean drinking water and a high life expectancy. Poor countries suffer from very low access to safe drinking water and lower life expectancy than rich countries.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Obesity among adults (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Obesity among adults (%) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436667-11336f8c-eb9b-4998-894b-c82a25e62115.png)

High and relative income countries (yellow and blue) have very high rates of adult obesity. Because high-income countries have a lot of excess food, the rate of obesity is very high. Compared to low-income countries, obesity rates are very low. Because low-income countries do not have much food and lack of food, obesity rates are lower, but in terms of life expectancy, Poor countries still have a lower average age than high-income countries.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Beer consumption per capita (Liter)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Beer consumption per capita (Liter) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436905-8425d0c8-c3d5-4566-9ecb-9d2c32b39e49.png)


In terms of beer consumption per capita, Group 2 and Group 0 countries (yellow and blue) have a very high beer consumption rate. As for the low-income Group 1 (purple) countries, the amount of beer consumed per capita is lower than in the high-income countries.


Rich countries should share the economy and poorer countries, so that CO2 emissions into the environment are reduced. And poor countries, if they have economic conditions, their quality of life will be higher, the average life expectancy will be higher.
