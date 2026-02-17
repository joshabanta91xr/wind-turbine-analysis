import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import numpy as np
import seaborn as sns 

df = pd.read_csv(r"D:\PROJECTS\Wind turbine\data wt\Wind_Turbine_2025.csv")
df
df.dtype
pd.set_option('display.max_columns', None)
df.head(10)
df.columns.tolist()
# Analyze errored
df.shape
df.dtypes

df.info()
df.describe()


#Missing value count 
df.isnull().sum()

# there are ~500 null values are there in each column
# we fill each numaric column  with mean or median 
# becouse sensore usualy fluctuate arounf avarage

for col in df.select_dtypes(include = 'float'):
    df[col] = df[col].fillna(df[col].mean())
    
df.isnull().sum()
#check missing values

df.duplicated().sum()


#Fixes column naming, missing values, and prepares the label column for modeling.

df['date'] = pd.to_datetime(df['date'], dayfirst=True)

df['Failure_status'] = df['Failure_status'].str.strip().map({'Failure': 1, 'No Failure': 0})
df.dtypes


# Statistical Metrics
numeric_cols = df.select_dtypes(include=np.number).columns
print(df[numeric_cols].mean())     # Mean for all numerics
print(df[numeric_cols].median())   # Median for all numerics
print(df[numeric_cols].mode().iloc[0])  # Mode for all numerics
print(df[numeric_cols].std())      # Standard deviation
print(df[numeric_cols].min())      # Minimum of each variable
print(df[numeric_cols].max())      # Maximum of each variable

# Full descriptive statistics
desc = df[numeric_cols].describe()
desc

# now we moving to the EDA part:
 
# Plot distributions for key numerical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()

df[num_cols].hist(figsize=(18, 12), bins=30)
plt.tight_layout()
plt.show()
# Shows summary statistics and plots feature distributions.

   
# 1.Failure count
plt.figure()
df["Failure_status"].value_counts().plot(kind='bar',color = "orange",edgecolor = "black",alpha = 0.9)
plt.title("Failure vs Non-Failure count")
plt.xlabel("Failure Status (0 = No Failure,1 = Failure)")
plt.ylabel("Count")
plt.show()


# this is very important insight:
# we have
# most of the records are NON-FAILURE
# failures are very few compared to normal running time



#*Failure Analysis
# Compare distributions by failure status
for col in num_cols:
    if col != 'Failure_status':
        plt.figure(figsize=(7, 4))
        sns.boxplot(x='Failure_status', y=col, data=df)
        plt.title(f"{col} by Failure Status")
        plt.show()

# Failure rate
failure_rate = df['Failure_status'].mean()
print(f"Failure Rate: {failure_rate:.2%}")

# Now we explore sensor differences
# The MOST IMPORTANT sensor known from real wind turbine research is:
# Gear Oil Temperature

# compare - Avg Gear Oil Temperature between Failure vs No Failure

avg_vals = df.groupby('Failure_status')['Gear_oil_temperature'].mean()

plt.figure()
avg_vals.plot(kind='bar', color = "orange",edgecolor = "black",alpha = 0.9)
plt.title("Avg Gear Oil Temperature: Failure vs No Failure")
plt.xlabel("Failure Flag (0= No Failure, 1= Failure)")
plt.ylabel("Average Gear Oil Temperature")
plt.show()

#This will show if average temperature is higher in failure periods.

# this is a VERY important result.
# we can clearly see:
# Status	          Avg Gear Oil Temperature
# No Failure (0) ---	~15 °C
# Failure (1) ---	   ~19.6 °C

# this means:
# Gear Oil Temperature increases before failure
# this is a real–world insight used in predictive maintenance.



# Avg Generator Bearing Temperature vs Failure Status

avg_vals = df.groupby('Failure_status')['Generator_bearing_temperature'].mean()

plt.figure()
avg_vals.plot(kind='bar', color = "orange",edgecolor = "black",alpha = 0.9)
plt.title("Avg Generator Bearing Temperature: Failure vs No Failure")
plt.xlabel("Failure Flag (0=No Failure, 1=Failure)")
plt.ylabel("Average Generator Bearing Temperature")
plt.show()


# Status	          Avg Bearing Temp
# No Failure (0) ---   ~17.6°C
# Failure (1)    ---   	~22.2°C



# Histogram for seeing feature distribution
df['Wind_speed'].hist(bins=30)
plt.title('Distribution of Windspeed')
plt.xlabel('Wind_speed')
plt.ylabel('Frequency')
plt.show()

# bin wind speed (nearest integer)

df['Wind_bin'] = df['Wind_speed'].round()

# group & average power output by binned wind

curve = df.groupby('Wind_bin')['Power'].mean().reset_index()

plt.figure()
plt.plot(curve['Wind_bin'], curve['Power'], color = "orange",alpha=0.8)
plt.title("Wind Speed vs Avg Power Output (Binned Curve)")
plt.xlabel("Wind Speed (rounded)")
plt.ylabel("Avg Power Output")
plt.show()


# this is the signature shape of a real wind turbine:
# Wind Speed Zone	Behaviour
# 0–8 m/s	            very low power
# 8–20 m/s	        power increases rapidly
# around ~23 m/s	    Maximum power output
# above ~25 m/s	    power drops (turbine protects itself / shut down)



#*Correlation and Feature Insights
plt.figure(figsize=(14, 12))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()




#Visualizes which operational parameters are abnormal during failures, and calculates the overall failure rate.

# Maximize: power, rotor_speed, operating_temperature in optimal range (not extremes), system uptime.
# Minimize: High generator_bearing_temperature, high gear_oil_temperature, ambient_temperature extremes, failure_status.

# Find top predictors of failure (correlation)
failure_corr = df[num_cols].corr()['Failure_status'].sort_values(ascending=False)
print("Top features correlated with failure:\n", failure_corr.head())








