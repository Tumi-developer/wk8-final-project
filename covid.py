import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime


#2. Data Loading & Exploration
df = pd.read_csv('owid-covid-data.csv')
print("Column names:")
print(df.columns)
print("\n")


# Preview the first 5 rows of the dataframe
print("Preview of the COVID-19 dataset:")
df.head()


# Check for missing values in each column
print("Missing values count per column:")
missing_values = df.isnull().sum()

# Display the count of missing values
print(missing_values)


#3. Data Cleaning

# Filter for countries of interest: Kenya, USA, India
countries_of_interest = ['Kenya', 'United States', 'India']
filtered_df = df[df['location'].isin(countries_of_interest)]

# Display basic information about the filtered dataset
print(f"Dataset shape after filtering: {filtered_df.shape}")
print("\nSample of filtered data:")
print(filtered_df.head())

# Check available columns to identify interesting metrics
print("\nAvailable columns for analysis:")
print(filtered_df.columns.tolist())


# Create a summary of key COVID metrics for these countries
# total cases, total deaths, and total vaccinations
summary = filtered_df.groupby('location').agg(
    latest_date=('date', 'max'),
    total_cases=('total_cases', 'last'),
    total_deaths=('total_deaths', 'last'),
    total_vaccinations=('total_vaccinations', 'last'),
    population=('population', 'first')
).reset_index()

print("\nSummary of COVID-19 data for countries of interest:")
print(summary)


# Create a visualization of daily new cases over time
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country)

plt.title('Daily New COVID-19 Cases (7-day smoothed)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Check the original date format
print("Original date column type:", df['date'].dtype)
print("First few dates before conversion:")
print(df['date'].head())


# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Verify the conversion
print("\nDate column type after conversion:", df['date'].dtype)
print("First few dates after conversion:")
print(df['date'].head())


#3. Handling Missing Numeric Values with interpolate()

df = pd.read_csv('owid-covid-data.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"\nMissing values before interpolation:")
print(df.isna().sum().sort_values(ascending=False).head(10))


# Group by location to ensure interpolation happens within each country's data
# This prevents interpolating between different countries
grouped = df.groupby('location')

# Create a new dataframe to store the interpolated results
interpolated_df = pd.DataFrame()

for name, group in grouped:
    # Sort by date to ensure proper time-series interpolation
    group = group.sort_values('date')
    
    # Select only numeric columns for interpolation
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    
    # Apply interpolation to numeric columns
    group[numeric_cols] = group[numeric_cols].interpolate(method='linear')
    
    # Append to the result dataframe
    interpolated_df = pd.concat([interpolated_df, group])


# Check missing values after interpolation
print(f"\nMissing values after interpolation:")
print(interpolated_df.isna().sum().sort_values(ascending=False).head(10))


# Calculate the percentage of missing values filled
before_missing = df.isna().sum().sum()
after_missing = interpolated_df.isna().sum().sum()
if before_missing > 0:
    percent_filled = ((before_missing - after_missing) / before_missing) * 100
    print(f"\nPercentage of missing values filled: {percent_filled:.2f}%")


# Visualize an example of interpolation for a specific country and metric
country = 'India'
metric = 'new_cases'

# Get original and interpolated data for the selected country and metric
country_orig = df[df['location'] == country][['date', metric]]
country_interp = interpolated_df[interpolated_df['location'] == country][['date', metric]]

# Plot the comparison
plt.figure(figsize=(12, 6))
plt.plot(country_orig['date'], country_orig[metric], 'o-', label='Original Data', alpha=0.5)
plt.plot(country_interp['date'], country_interp[metric], 'r.-', label='Interpolated Data')
plt.title(f'{metric} in {country}: Original vs Interpolated')
plt.xlabel('Date')
plt.ylabel(metric)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Load the COVID-19 dataset
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])


# Select countries to visualize
selected_countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'France', 'Germany', 'Japan']

# Filter data for selected countries and sort by date
filtered_df = df[df['location'].isin(selected_countries)].sort_values('date')

# Create the plot
plt.figure(figsize=(14, 8))

# Plot each country's total cases
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], linewidth=2, label=country)


# Format the plot
plt.title('COVID-19 Total Cases Over Time by Country', fontsize=18, pad=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Cases (log scale)', fontsize=14)
plt.yscale('log')  # Using log scale to better visualize differences
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Countries', fontsize=12, title_fontsize=14)

# Format the x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Add annotations for the latest data points
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country]
    latest = country_data.iloc[-1]
    plt.annotate(f"{latest['total_cases']:,.0f}",
                 xy=(latest['date'], latest['total_cases']),
                 xytext=(10, 0),
                 textcoords='offset points',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


# Add a note about the data source
plt.figtext(0.5, 0.01, "Data source: Our World in Data (OWID)", 
            ha="center", fontsize=10, style='italic')

# Add grid lines for better readability
plt.grid(True, which="both", ls="-", alpha=0.2)

# Adjust layout and display
plt.tight_layout()
plt.show()

# a second plot showing daily new cases (7-day rolling average)
plt.figure(figsize=(14, 8))

# Calculate and plot 7-day rolling average of new cases for each country
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    country_data['rolling_new_cases'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['rolling_new_cases'], linewidth=2, label=country)

# Format the second plot
plt.title('COVID-19 Daily New Cases (7-Day Rolling Average)', fontsize=18, pad=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases (7-Day Average)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Countries', fontsize=12, title_fontsize=14)

# Format the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

# Load the COVID-19 dataset
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Select countries to visualize (including some of the most affected countries)
selected_countries = ['United States', 'Brazil', 'India', 'United Kingdom', 
                     'Italy', 'France', 'Germany', 'Russia', 'Mexico', 'Peru']

# Filter data for selected countries and sort by date
filtered_df = df[df['location'].isin(selected_countries)].sort_values('date')

# Create the main plot for total deaths
fig, ax1 = plt.subplots(figsize=(16, 10))


# Plot each country's total deaths
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country]
    ax1.plot(country_data['date'], country_data['total_deaths'], 
             linewidth=2.5, label=country, alpha=0.9)

# Format the plot
ax1.set_title('COVID-19 Total Deaths Over Time by Country', 
              fontsize=22, pad=20, fontweight='bold')
ax1.set_xlabel('Date', fontsize=16)
ax1.set_ylabel('Total Deaths', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Format the x-axis to show dates nicely
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Add legend with custom positioning
legend = ax1.legend(title='Countries', fontsize=14, title_fontsize=16, 
                   loc='upper left', bbox_to_anchor=(1, 1))


# Add annotations for the latest data points
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and pd.notna(country_data['total_deaths'].iloc[-1]):
        latest = country_data.iloc[-1]
        ax1.annotate(f"{country}: {int(latest['total_deaths']):,}",
                    xy=(latest['date'], latest['total_deaths']),
                    xytext=(10, 0),
                    textcoords='offset points',
                    fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Add a second subplot for deaths per million (normalized view)
ax2 = fig.add_subplot(212)

# Plot each country's deaths per million
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country]
    ax2.plot(country_data['date'], country_data['total_deaths_per_million'], 
             linewidth=2.5, label=country, alpha=0.9)


# Format the second plot
ax2.set_title('COVID-19 Deaths per Million Population', fontsize=18, pad=15)
ax2.set_xlabel('Date', fontsize=16)
ax2.set_ylabel('Deaths per Million', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Format the x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Add legend to the second plot
ax2.legend(title='Countries', fontsize=14, title_fontsize=16, 
          loc='upper left', bbox_to_anchor=(1, 1))

# Add a note about the data source
plt.figtext(0.5, 0.01, "Data source: Our World in Data (OWID)", 
            ha="center", fontsize=12, style='italic')


# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, right=0.85)

# Show the plot
plt.show()

# Optional: Create a third visualization showing the mortality rate over time
plt.figure(figsize=(16, 8))

# Calculate and plot case fatality rate (total_deaths/total_cases)
for country in selected_countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    # Calculate case fatality rate (as percentage)
    country_data['case_fatality_rate'] = (country_data['total_deaths'] / 
                                         country_data['total_cases']) * 100
    plt.plot(country_data['date'], country_data['case_fatality_rate'], 
             linewidth=2, label=country)



# Format the plot
plt.title('COVID-19 Case Fatality Rate Over Time', fontsize=20, pad=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Case Fatality Rate (%)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Countries', fontsize=12, title_fontsize=14, loc='best')

# Format the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# Load the COVID-19 dataset
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Select countries to compare
countries = ['United States', 'India', 'Brazil', 'United Kingdom', 
            'France', 'Germany', 'Japan', 'South Korea', 'Italy', 'Spain']

# Filter data for selected countries
filtered_df = df[df['location'].isin(countries)]

# Create a figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 24), sharex=True)



# 1. First plot: Daily new cases with 7-day rolling average
ax1 = axes[0]
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    
    # Calculate 7-day rolling average
    country_data['new_cases_smooth'] = country_data['new_cases'].rolling(window=7).mean()
    
    # Plot the smoothed data
    ax1.plot(country_data['date'], country_data['new_cases_smooth'], 
             linewidth=2.5, label=country, alpha=0.9)

# Format the first plot
ax1.set_title('Daily New COVID-19 Cases (7-Day Rolling Average)', 
              fontsize=20, pad=15, fontweight='bold')
ax1.set_ylabel('New Cases', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(title='Countries', fontsize=12, title_fontsize=14, loc='upper left')

# Add thousands separator to y-axis
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

# 2. Second plot: Daily new cases per million population (normalized by population size)
ax2 = axes[1]
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    
    # Calculate 7-day rolling average for new cases per million
    country_data['new_cases_per_million_smooth'] = country_data['new_cases_per_million'].rolling(window=7).mean()
    
    # Plot the smoothed data
    ax2.plot(country_data['date'], country_data['new_cases_per_million_smooth'], 
             linewidth=2.5, label=country, alpha=0.9)

# Format the second plot
ax2.set_title('Daily New COVID-19 Cases per Million Population (7-Day Rolling Average)', 
              fontsize=20, pad=15, fontweight='bold')
ax2.set_ylabel('New Cases per Million', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(title='Countries', fontsize=12, title_fontsize=14, loc='upper left')

# 3. Third plot: Heatmap-style comparison of case intensity over time
ax3 = axes[2]


# Create a pivot table for the heatmap data
# First, get the smoothed data for all countries
heatmap_data = []
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    country_data['new_cases_per_million_smooth'] = country_data['new_cases_per_million'].rolling(window=7).mean()
    country_data = country_data[['date', 'location', 'new_cases_per_million_smooth']]
    heatmap_data.append(country_data)

heatmap_df = pd.concat(heatmap_data)

# Pivot the data for the heatmap
pivot_df = heatmap_df.pivot(index='location', columns='date', values='new_cases_per_million_smooth')

# Resample to weekly data to make the heatmap more readable
pivot_df = pivot_df.resample('W', axis=1).mean()

# Plot the heatmap
im = ax3.imshow(pivot_df.values, aspect='auto', cmap='YlOrRd')


# Configure the heatmap
ax3.set_title('Weekly Average New Cases per Million by Country', 
              fontsize=20, pad=15, fontweight='bold')
ax3.set_yticks(np.arange(len(countries)))
ax3.set_yticklabels(pivot_df.index)
ax3.tick_params(axis='y', which='major', labelsize=12)

# Add x-axis labels (dates) at regular intervals
date_indices = np.linspace(0, pivot_df.shape[1]-1, 10, dtype=int)
date_labels = [pivot_df.columns[i].strftime('%b %Y') for i in date_indices]
ax3.set_xticks(date_indices)
ax3.set_xticklabels(date_labels, rotation=45)

# Add a colorbar
cbar = fig.colorbar(im, ax=ax3)
cbar.set_label('New Cases per Million (7-Day Avg)', fontsize=14)

# Format the x-axis for the first two plots
for ax in axes[:2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


# Add a note about the data source
plt.figtext(0.5, 0.01, "Data source: Our World in Data (OWID)", 
            ha="center", fontsize=12, style='italic')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()

# Create an additional plot showing case growth rates
plt.figure(figsize=(16, 10))

# Calculate and plot percentage change in weekly cases
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country].copy()
    
    # Calculate 7-day rolling sum of new cases
    country_data['weekly_cases'] = country_data['new_cases'].rolling(window=7).sum()


 # Calculate week-over-week percentage change
    country_data['pct_change'] = country_data['weekly_cases'].pct_change(periods=7) * 100
    
    # Plot the percentage change
    plt.plot(country_data['date'], country_data['pct_change'], 
             linewidth=2, label=country, alpha=0.8)

# Format the plot
plt.title('Week-over-Week Percentage Change in COVID-19 Cases', fontsize=20, pad=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Percentage Change (%)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Countries', fontsize=12, title_fontsize=14, loc='best')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Format the x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Added import for numpy

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Check the columns to understand what data we have
print("Available columns:", df.columns.tolist())

# Calculate death rate (deaths per confirmed cases) for each country's latest data
# First, let's get the latest data for each country
latest_data = df.sort_values('date').groupby('location').last().reset_index()

# Calculate death rate (deaths/cases)
latest_data['death_rate'] = latest_data['total_deaths'] / latest_data['total_cases'] * 100

# Remove rows with NaN or infinite values
# Changed to use np.isfinite() instead of Series.isfinite()
latest_data = latest_data[latest_data['death_rate'].notna() & np.isfinite(latest_data['death_rate'])]

# Display top 15 countries by death rate
top_countries = latest_data.sort_values('death_rate', ascending=False).head(15)
print("\nTop 15 countries by death rate (deaths/cases):")
print(top_countries[['location', 'total_cases', 'total_deaths', 'death_rate']])

# Calculate global death rate
global_death_rate = df['total_deaths'].sum() / df['total_cases'].sum() * 100
print(f"\nGlobal death rate: {global_death_rate:.2f}%")

# Visualize death rates for top 10 countries
plt.figure(figsize=(12, 8))
sns.barplot(x='death_rate', y='location', data=top_countries.head(10), palette='viridis')
plt.title('COVID-19 Death Rate by Country (Top 10)')
plt.xlabel('Death Rate (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()



# Visualizing Vaccination Progress
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Select a few countries to plot (you can modify this list)
countries = ['United States', 'United Kingdom', 'Israel', 'Canada', 'Germany', 'India', 'Brazil']

# Filter the data for selected countries
filtered_df = df[df['location'].isin(countries)]

# Create a figure with a larger size
plt.figure(figsize=(14, 8))

# Plot cumulative vaccinations over time for each country
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]
    
    # Check if total_vaccinations column exists and has data for this country
    if 'total_vaccinations' in country_data.columns and country_data['total_vaccinations'].notna().any():
        # Plot only rows where total_vaccinations is not null
        valid_data = country_data[country_data['total_vaccinations'].notna()]
        plt.plot(valid_data['date'], valid_data['total_vaccinations'], label=country, linewidth=2)

# Add labels and title
plt.title('Cumulative COVID-19 Vaccinations Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Vaccinations', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Format y-axis to show numbers in millions
plt.ticklabel_format(style='plain', axis='y')
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x/1000000)}M" if x >= 1000000 else f"{int(x/1000)}K"))

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Select a few countries to plot (you can modify this list)
countries = ['United States', 'United Kingdom', 'Israel', 'Canada', 'Germany', 'India', 'Brazil']

# Filter the data for selected countries
filtered_df = df[df['location'].isin(countries)]

# Create a figure with a larger size
plt.figure(figsize=(14, 8))

# Plot percentage of population fully vaccinated over time for each country
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]
    
    # Check if people_fully_vaccinated_per_hundred column exists and has data for this country
    if 'people_fully_vaccinated_per_hundred' in country_data.columns and country_data['people_fully_vaccinated_per_hundred'].notna().any():
        # Plot only rows where people_fully_vaccinated_per_hundred is not null
        valid_data = country_data[country_data['people_fully_vaccinated_per_hundred'].notna()]
        plt.plot(valid_data['date'], valid_data['people_fully_vaccinated_per_hundred'], label=country, linewidth=2)

# Add labels and title
plt.title('Percentage of Population Fully Vaccinated Against COVID-19', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Percentage of Population (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Set y-axis to show percentage from 0 to 100
plt.ylim(0, 100)

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# For the latest vaccination rates, let's create a bar chart
# Get the latest data for each country
latest_data = filtered_df.sort_values('date').groupby('location').last().reset_index()

# Create a bar chart for the latest vaccination percentages
plt.figure(figsize=(12, 6))
latest_data = latest_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False)

sns.barplot(x='location', y='people_fully_vaccinated_per_hundred', data=latest_data, palette='viridis')
plt.title('Latest Percentage of Population Fully Vaccinated by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Percentage of Population (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add value labels on top of each bar
for i, v in enumerate(latest_data['people_fully_vaccinated_per_hundred']):
    if pd.notna(v):  # Only add label if value is not NaN
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Added import for numpy

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Check the columns to understand what data we have
print("Available columns:", df.columns.tolist())

# Calculate death rate (deaths per confirmed cases) for each country's latest data
# First, let's get the latest data for each country
latest_data = df.sort_values('date').groupby('location').last().reset_index()

# Calculate death rate (deaths/cases)
latest_data['death_rate'] = latest_data['total_deaths'] / latest_data['total_cases'] * 100

# Remove rows with NaN or infinite values
# Changed to use np.isfinite() instead of Series.isfinite()
latest_data = latest_data[latest_data['death_rate'].notna() & np.isfinite(latest_data['death_rate'])]

# Display top 15 countries by death rate
top_countries = latest_data.sort_values('death_rate', ascending=False).head(15)
print("\nTop 15 countries by death rate (deaths/cases):")
print(top_countries[['location', 'total_cases', 'total_deaths', 'death_rate']])

# Calculate global death rate
global_death_rate = df['total_deaths'].sum() / df['total_cases'].sum() * 100
print(f"\nGlobal death rate: {global_death_rate:.2f}%")

# Visualize death rates for top 10 countries
plt.figure(figsize=(12, 8))
sns.barplot(x='death_rate', y='location', data=top_countries.head(10), palette='viridis')
plt.title('COVID-19 Death Rate by Country (Top 10)')
plt.xlabel('Death Rate (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()



# Visualizing Vaccination Progress
# Plot cumulative vaccinations over time for selected countries.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Select a few countries to plot (you can modify this list)
countries = ['United States', 'United Kingdom', 'Israel', 'Canada', 'Germany', 'India', 'Brazil']

# Filter the data for selected countries
filtered_df = df[df['location'].isin(countries)]

# Create a figure with a larger size
plt.figure(figsize=(14, 8))

# Plot cumulative vaccinations over time for each country
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]
    
    # Check if total_vaccinations column exists and has data for this country
    if 'total_vaccinations' in country_data.columns and country_data['total_vaccinations'].notna().any():
        # Plot only rows where total_vaccinations is not null
        valid_data = country_data[country_data['total_vaccinations'].notna()]
        plt.plot(valid_data['date'], valid_data['total_vaccinations'], label=country, linewidth=2)

# Add labels and title
plt.title('Cumulative COVID-19 Vaccinations Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Vaccinations', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)


# Format y-axis to show numbers in millions
plt.ticklabel_format(style='plain', axis='y')
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x/1000000)}M" if x >= 1000000 else f"{int(x/1000)}K"))

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the COVID-19 data
df = pd.read_csv('owid-covid-data.csv')

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Select a few countries to plot (you can modify this list)
countries = ['United States', 'United Kingdom', 'Israel', 'Canada', 'Germany', 'India', 'Brazil']

# Filter the data for selected countries
filtered_df = df[df['location'].isin(countries)]

# Create a figure with a larger size
plt.figure(figsize=(14, 8))

# Plot percentage of population fully vaccinated over time for each country
for country in countries:
    country_data = filtered_df[filtered_df['location'] == country]

     # Check if people_fully_vaccinated_per_hundred column exists and has data for this country
    if 'people_fully_vaccinated_per_hundred' in country_data.columns and country_data['people_fully_vaccinated_per_hundred'].notna().any():
        # Plot only rows where people_fully_vaccinated_per_hundred is not null
        valid_data = country_data[country_data['people_fully_vaccinated_per_hundred'].notna()]
        plt.plot(valid_data['date'], valid_data['people_fully_vaccinated_per_hundred'], label=country, linewidth=2)

# Add labels and title
plt.title('Percentage of Population Fully Vaccinated Against COVID-19', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Percentage of Population (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Set y-axis to show percentage from 0 to 100
plt.ylim(0, 100)

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# For the latest vaccination rates, let's create a bar chart
# Get the latest data for each country
latest_data = filtered_df.sort_values('date').groupby('location').last().reset_index()

# Create a bar chart for the latest vaccination percentages
plt.figure(figsize=(12, 6))
latest_data = latest_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False)

sns.barplot(x='location', y='people_fully_vaccinated_per_hundred', data=latest_data, palette='viridis')
plt.title('Latest Percentage of Population Fully Vaccinated by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Percentage of Population (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add value labels on top of each bar
for i, v in enumerate(latest_data['people_fully_vaccinated_per_hundred']):
    if pd.notna(v):  # Only add label if value is not NaN
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# Insights & Reporting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('owid-covid-data.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Display basic information
print(f"Dataset timeframe: {df['date'].min()} to {df['date'].max()}")
print(f"Number of countries/regions: {df['location'].nunique()}")
print(f"Total columns/metrics: {df.shape[1]}")

# Key Insight 1: Global COVID-19 progression
# Aggregate data by date for global metrics
global_daily = df[df['location'] == 'World'].set_index('date')
plt.figure(figsize=(12, 6))
plt.plot(global_daily.index, global_daily['new_cases_smoothed'], label='New Cases (7-day avg)')
plt.plot(global_daily.index, global_daily['new_deaths_smoothed'], label='New Deaths (7-day avg)')
plt.title('Global COVID-19 Progression')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Key Insight 2: Top 10 countries by total cases per million
top_countries_cases = df[df['date'] == df['date'].max()].sort_values(by='total_cases_per_million', ascending=False).head(10)
print("\nTop 10 Countries by Total Cases per Million:")
print(top_countries_cases[['location', 'total_cases_per_million']].reset_index(drop=True))

# Key Insight 3: Vaccination progress across continents
latest_date = df['date'].max()
vax_by_continent = df[(df['date'] == latest_date) & (df['continent'].notna())].groupby('continent')[
    ['people_fully_vaccinated_per_hundred']].mean().sort_values(by='people_fully_vaccinated_per_hundred', ascending=False)
print("\nVaccination Progress by Continent (people fully vaccinated per 100):")
print(vax_by_continent)

# Key Insight 4: Correlation between GDP per capita and COVID-19 metrics
# Using the latest data for each country
latest_by_country = df.sort_values('date').groupby('location').last()
correlation_metrics = ['gdp_per_capita', 'total_cases_per_million', 'total_deaths_per_million', 
                       'people_fully_vaccinated_per_hundred']
correlation_data = latest_by_country[correlation_metrics].corr()
print("\nCorrelation between GDP per capita and COVID-19 metrics:")
print(correlation_data['gdp_per_capita'].drop('gdp_per_capita'))

# Key Insight 5: Case fatality rate comparison
latest_by_country['case_fatality_rate'] = latest_by_country['total_deaths'] / latest_by_country['total_cases'] * 100
top_cfr = latest_by_country.sort_values('case_fatality_rate', ascending=False).head(10)
print("\nTop 10 Countries by Case Fatality Rate (%):")
print(top_cfr[['case_fatality_rate']].round(2))