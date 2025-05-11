import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

df = pd.read_csv("Airports2.csv")
df = df[['Origin_airport', 'Destination_airport', 'Passengers', 'Seats', 'Flights', 'Distance', 'Fly_date']]

df_2006 = df[(df['Fly_date'] >= '2006-01-01') & (df['Fly_date'] < '2007-01-01')]

df_2006 = df_2006.copy()

# Calculate the total distance worth of flights (32 flights * 100 miles = 3200 miles)
df_2006['Total_Distance'] = df_2006['Distance'] * df_2006['Flights']

# Sum by origin
origin_stats = df_2006.groupby('Origin_airport')[['Passengers', 'Seats', 'Flights', 'Total_Distance']].sum()

# Sum by destination
dest_stats = df_2006.groupby('Destination_airport')[['Passengers', 'Seats', 'Flights', 'Total_Distance']].sum()

# Combine the two to get total traffic per airport
airport_df = origin_stats.add(dest_stats, fill_value=0).reset_index()
airport_df.rename(columns={'index': 'Airport'}, inplace=True)

# Calculate the distance per flight average, and the fill ratio per flight
airport_df['Dist per Flight'] = airport_df['Total_Distance'].astype(float) / airport_df['Flights']
airport_df['Fill Ratio'] = airport_df['Passengers'].astype(float) / airport_df['Seats']
airport_df['Fill Ratio'] = airport_df['Fill Ratio'].fillna(0)
airport_df['Dist per Flight'] = airport_df['Dist per Flight'].fillna(0)

# Yeo-Johnson transformation to correct for skew/to normalize
pt = PowerTransformer(method='yeo-johnson')
airport_df[['YJ_Passengers', 'YJ_Seats', 'YJ_Flights', 'YJ_Distance']] = pt.fit_transform(
    airport_df[['Passengers', 'Seats', 'Flights', 'Total_Distance']]
)

# Set up test variable
X = airport_df[['YJ_Passengers', 'YJ_Seats', 'YJ_Flights', 'YJ_Distance']]
features = ['YJ_Passengers', 'YJ_Seats', 'YJ_Flights', 'YJ_Distance']

# Step 3: Use the Elbow Method to determine optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.grid(True)
plt.show()

# Selecting k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
airport_df['Cluster'] = kmeans.fit_predict(X)

# PCA Visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X)

# PCA Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=airport_df['Cluster'], palette='Set1')
plt.title('Airport Route Clusters (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()


## Radar Chart
features = ['YJ_Passengers', 'YJ_Seats', 'YJ_Flights', 'YJ_Distance', 'Dist per Flight', 'Fill Ratio']

# Compute average feature values for each cluster
cluster_means = airport_df.groupby('Cluster')[features].mean()

# Normalize the data to [0, 1] for fair visual comparison
cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()

# Setup radar plot structure
labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the circle

# Create the radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for idx, row in cluster_means_norm.iterrows():
    values = row.tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, label=f'Cluster {idx}')
    ax.fill(angles, values, alpha=0.25)

# 5. Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Add legend and formatting
ax.set_title('Cluster Feature Profiles (Normalized)', size=14, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()


# Dictionary to label airports from their 3-digit codes
airport_names = {
    'HUF': 'Terre Haute Regional Airport',
    'ARB': 'Ann Arbor Airport',
    'STJ': 'Rosecrans Memorial Airport',
    'SKY': 'Griffing Sandusky Airport',
    'DFW': 'Dallas Fort Worth International Airport',
    'PIT': 'Pittsburgh International Airport',
    'CAE': 'Columbia Metropolitan Airport',
    'COS': 'Colorado Springs Airport',
    'TUL': 'Tulsa International Airport',
    'ELD': 'South Arkansas Regional Airport at Goodwin Field',
    'VCT': 'Victoria Regional Airport',
    'CDC': 'Cedar City Regional Airport',
    'WDG': 'Enid Woodring Regional Airport',
    'HYS': 'Hays Regional Airport',
    'DQU': 'Neets Bay'
}

# Sample 5 random flights
sampled_flights = (
    airport_df
    .groupby('Cluster', group_keys=True)
    .apply(lambda x: x.sample(n=5, random_state=41), include_groups=False)
)

# Add the Airport Name column from the 3-digit code using the dictionary above
sampled_flights['Airport Name'] = sampled_flights['Airport'].map(airport_names)
# Group by 'Cluster', sample 5 random flights from each cluster, and reset the index
sampled_flights_grouped = sampled_flights.groupby('Cluster', as_index=False).apply(lambda x: x.sample(n=5, random_state=42))

# Drop the unnecessary columns
sampled_flights_grouped = sampled_flights_grouped.drop(columns=['YJ_Passengers','YJ_Seats','YJ_Flights','YJ_Distance'])

# Move the 'Airport Name' column to the second column
cols = ['Airport Name', 'Airport', 'Passengers', 'Seats', 'Flights', 'Total_Distance', 'Dist per Flight', 'Fill Ratio']
sampled_flights_grouped = sampled_flights_grouped[cols]

# Format numerical columns to avoid scientific notation
sampled_flights_grouped['Passengers'] = sampled_flights_grouped['Passengers'].apply(lambda x: '{:,.0f}'.format(x))
sampled_flights_grouped['Seats'] = sampled_flights_grouped['Seats'].apply(lambda x: '{:,.0f}'.format(x))
sampled_flights_grouped['Flights'] = sampled_flights_grouped['Flights'].apply(lambda x: '{:,.0f}'.format(x))
sampled_flights_grouped['Total_Distance'] = sampled_flights_grouped['Total_Distance'].apply(lambda x: '{:,.0f}'.format(x))

# Display the tables grouped by Cluster
for cluster_label, group in sampled_flights_grouped.groupby('Cluster'):
    print(f"\n### Cluster {cluster_label} ###\n")
    print(tabulate(group, headers='keys', tablefmt='github', showindex=False))
    
# Visual space for table
print()