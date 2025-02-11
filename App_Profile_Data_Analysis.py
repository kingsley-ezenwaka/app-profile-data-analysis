# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data import
playstore = pd.read_csv('googleplaystore.csv')

# Info on dataset shape (rows/columns) and data types
print(playstore.info())

# Inspect for columns with readily analyzable data
playstore.describe()

# Check total null entries per column
playstore.isna().sum()

# Preview of first 3 rows
playstore.head(3)

# --
# Data Cleaning
# --

# Filter out the 
mask = playstore.Installs == 'Free'
playstore = playstore[~mask]

# 'Installs' column object to int64 dtype
playstore['Installs'] = playstore['Installs'].str.replace('+', '').str.replace(',','').astype('int64')

# Pre-inspected with plot of `value_counts` (`dropna` set to False to include nulls)

# 'Price' column object to float64 dtype
playstore['Price'] = playstore['Price'].str.replace('$', '').astype('float64')

# Extract size multiplier (i.e.'M' or 'k') to integer values
extract = playstore.Size.str[-1]
mmap = {'M':1000, 'k':1}
SizeMultiplier = np.array([mmap[x] if x in mmap.keys() else np.nan for x in extract])

# Extract the 'Size' integer portion and broadcast to new column 'SizeBytes' float dtype column
playstore['SizeKBytes'] = pd.to_numeric(playstore.Size.str[:-1], errors='coerce') * SizeMultiplier

# Convert 'Reviews' to numeric type
playstore['Reviews'] = pd.to_numeric(playstore['Reviews'])

# Use a boolean mask to filter out 19.0 rating
mask = playstore['Rating'] == 19.0
playstore = playstore[~mask]

# Drop all rows with `Null` data
playstore = playstore.dropna()

# Check for 'App' duplicates
playstore[playstore['App'].duplicated()].shape

# Sort duplicates by 'App' and inspect the last 4th and 3rd rows
playstore[playstore['App'].duplicated()].sort_values('App').iloc[-4:-2]

# Sort the dataset by 'App' ascending and 'Reviews' descending
playstore.sort_values(by=['App','Reviews'],ascending=[True,False])

# Make a mask of the duplicates (from the sorted dataset)
mask = playstore['App'].duplicated()

# Filter out duplicates 
playstore = playstore[~mask]

# nenglish function -> returns true for non-English language strings 
def nenglish(string):
    count = 0
    for char in string:
        if ord(char) > 127: 
            count += 1
            if count > 3: return True
    return False

# Run a list of App names through the `nenglish()` function to get a boolean mask of the non-English apps 
non_english_mask = np.array([nenglish(app) for app in playstore['App'].tolist()])

# Apply mask to dataset to get the non-English apps
non_english_apps = playstore[non_english_mask]
non_english_apps.tail(2)

# Filter out non_English apps
playstore = playstore[~non_english_mask]

# Check 'Type' value_counts plot --> 2 values: 'Free' and 'Paid'
playstore.Type.value_counts()

# Use boolean mask to filter out 'Paid' apps
mask = playstore['Type'] == 'Paid'
playstore = playstore[~mask]

# Export cleaning results to CSV file
playstore.to_csv('googleplaystore_rev.csv', index=False)

# --
# Data Analysis
# --

## Build the Category Frequency Table (table0)

# Get the number of apps per category, sorted by category name
table0 = playstore.Category.value_counts().reset_index()
table0.columns = ['Category', 'NumOfApp']
table0 = table0.sort_values(by='Category', ascending=True)

# Get Total and Average Downloads per Category (in millions)
df = (playstore.Installs.groupby(playstore.Category).sum()).reset_index()
df = df.sort_values(by='Category', ascending=True)
table0['TotInstalls'] = df['Installs'].values / 1000000
table0['AvgInstalls'] = round(table0.TotInstalls / table0.NumOfApp, 2)

# Get Total and Average Review counts per Category (in hundred thousands) 
df = playstore.Reviews.groupby(playstore.Category).sum().reset_index()
table0['TotReviews'] = df['Reviews'].values / 100000
table0['AvgReviews'] = round(table0.TotReviews / table0.NumOfApp, 2)

table0.head(3) # View first 3 rows

## Broadcast Installs and Reviews to reduced units per 1 million and 100k respectively
playstore.loc[:,'Installs_rd'] = playstore.Installs / 1000000
playstore.loc[:,'Reviews_rd'] = playstore.Reviews / 100000
playstore.loc[:,'Ratio'] = round(playstore.Reviews_rd / playstore.Installs_rd, 4)

## Broadcast Analysis indices to Category freq table
table0.loc[:,'v2'] = round(table0.TotReviews / table0.NumOfApp, 1)              # Measure of Engagements
table0.loc[:,'v3'] = round(1 / (table0.AvgInstalls / table0.NumOfApp), 1)       # Measure of Saturation

## Combined Bar plot of Average Downloads and Average Reviews

# Set plot theme and initialize figure
sns.set(style='darkgrid',font_scale=0.6)
fig, axs = plt.subplots(figsize=(10,4))

# Plot the Average Downloads
sns.set_color_codes("pastel")
ax1 = sns.barplot(x='AvgInstalls', y='Category', data=table0, label='Average App Installations', color='b')

# Plot the Average Reviews
sns.set_color_codes("muted")
ax2  = sns.barplot(x='AvgReviews', y='Category', data=table0, label='Average App Reviews', color='b')

# Add a legend and informative labels
axs.legend(ncol=2, loc="lower right", frameon=True)
axs.set(xlim=[0,16], ylabel="", xlabel="Average Installations (in millions) & Average Reviews (in hundred thousands)")
sns.despine(left=True, bottom=True)

plt.show()

# Extract the category table for most popular app categories
list1 = table0[table0.AvgInstalls > 6].Category.tolist() # Categories with avg. downloads > 6 million
list2 = table0[table0.AvgReviews > 1.5].Category.tolist() # Categories with avg. reviews > 150,000
table1 = table0[table0.Category.isin(list1) | table0.Category.isin(list2)]

# Create temporary long-form dataframe for the bar plot
df = table1.rename(columns={'AvgInstalls':'Average Downloads','AvgReviews':'Average Reviews'})
df = df.melt(id_vars=['Category'],value_vars=['Average Downloads','Average Reviews']) # Un-pivot the table to long-form

# Bar plot of the most popular categories
sns.set(font_scale=.6)
fig, axs = plt.subplots(figsize=[8.5,3])
axs = sns.barplot(x='Category', y='value', hue='variable', data=df)
axs.set(ylim=[0,18], ylabel='', xlabel='', title='Category Average Downloads (in millions) & Average Reviews (in 100,000s)')
axs.legend(ncol=1, loc="upper right", frameon=True)
for i in range(0,len(axs.containers)):
    axs.bar_label(axs.containers[i], fontsize=7)
sns.despine(fig)

plt.show()

# Box plot of Ratings across Categories
sns.set(font_scale=0.6)
fig, axs = plt.subplots(1, 7, figsize=[8,3])
cat = table1.Category.tolist() # List of the most popular categories
i = 0
for a in axs:
    a = sns.boxplot(y=playstore.Rating[(playstore.Category==cat[i])], width=0.15, ax=axs[i])
    a.set(xlabel=cat[i], ylabel='Rating')
    i += 1

plt.suptitle('Boxplot of Ratings across Most Popular Categories')
plt.tight_layout()
plt.show()

# Compute the Popularity Index (P-Index)
table1.loc[:,'pindex'] = round((table1.AvgInstalls**2 + table1.AvgReviews**2)**0.5, 1)
table1.loc[:,'pindex_text'] = table1.Category + " (" + table1.pindex.astype(str) + ")"

# Scatter plot of the P-indices
sns.set()
fig, axs = plt.subplots(figsize=[5,4])

sns.scatterplot(x='AvgInstalls', y='AvgReviews', hue='Category', data=table1, legend=False)
axs.set(xlabel="'Average Downloads (in millions)'", ylabel="'Average Reviews (in 100,000s)'", title='Popularity Index')

# Annotate the points with corresponding P-index value
for i, row in table1.iterrows():  
    axs.text(row['AvgInstalls'], row['AvgReviews'], row["pindex_text"], 
            fontsize=7, ha='left', va='bottom', color='black', alpha=0.4)

plt.show()

# Computing the 'Saturation Index' for each category
table0.loc[:,'Sat'] = table0.Category + " (" + table0.v3.astype(str) + ")"
df = table0[table0.Category.isin(cat)]

# Scatter plot of top 5 MP ranks
sns.set(font_scale=0.8)
fig, axs = plt.subplots(figsize=[4,3])
sns.scatterplot(x='NumOfApp', y='v3', hue='Category', legend=False, data=df)

axs.set(xlabel='Number of Apps', 
        ylabel='Saturation Index', title='Empirical Estimation of Category Saturation')

# Annotate select points in scatterplot
for i, row in df.iterrows():  
    axs.text(row['NumOfApp'], row["v3"], row["Sat"], 
            fontsize=8, ha='left', va='bottom', color='black', alpha=0.4)
plt.show()

# --
# Summary
# --

