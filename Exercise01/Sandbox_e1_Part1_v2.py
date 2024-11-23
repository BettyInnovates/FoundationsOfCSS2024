import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================================
# World Bank Data Processing
# ==========================================================

# Define the World Bank indicators to fetch
world_bank_indicators = ['NY.GDP.PCAP.PP.KD', 'IT.NET.USER.ZS', 'SP.POP.TOTL']

# Fetch data for the defined indicators for the year 2015
world_bank_data = wb.data.DataFrame(
    world_bank_indicators,
    time=2015,
    skipAggs=True,
    labels=True
).reset_index()

# Show intermediate result: raw data
print(f"Raw data fetched from World Bank API (shape: {world_bank_data.shape}):")
print(world_bank_data.head())

# Drop rows with missing values to ensure clean data
world_bank_data = world_bank_data.dropna()

# Show intermediate result: after dropping NaN values
print(f"\nData after dropping rows with missing values (shape: {world_bank_data.shape}):")
print(world_bank_data.head())

# Calculate the total number of internet users (population * internet penetration rate)
world_bank_data = world_bank_data.assign(
    internet_user=world_bank_data['SP.POP.TOTL'] * world_bank_data['IT.NET.USER.ZS'] / 100
)

# Show intermediate result: after calculating internet users
print(f"\nData after calculating total internet users (shape: {world_bank_data.shape}):")
print(world_bank_data.head())

# Filter rows to keep only countries with at least 5 million internet users
world_bank_data = world_bank_data[world_bank_data['internet_user'] >= 5e+06]

# Show intermediate result: after filtering rows with at least 5 million internet users
print(f"\nData after filtering rows with at least 5 million internet users (shape: {world_bank_data.shape}):")
print(world_bank_data.head())

# Export the processed data to a CSV file
world_bank_data.to_csv(path_or_buf='world_bank_data.csv')
print("\nProcessed data has been saved to 'world_bank_data.csv'")

# ==========================================================
# Google Trends Data Processing
# ==========================================================

# Load the .csv file into a pandas DataFrame with proper column names
google_trends_data = pd.read_csv(
    'geoMap.csv',
    skiprows=3,
    names=['Country', 'G2013', 'G2015']
)

# Show intermediate result: raw data from CSV
print(f"Raw data loaded from 'geoMap.csv' (shape: {google_trends_data.shape}):")
print(google_trends_data.head())

# Remove all rows containing NaN values to ensure clean data
google_trends_data = google_trends_data.dropna()

# Show intermediate result: after dropping rows with missing values
print(f"\nData after dropping rows with missing values (shape: {google_trends_data.shape}):")
print(google_trends_data.head())

# Remove '%' symbols and convert columns G2013 and G2015 to integers
google_trends_data['G2013'] = google_trends_data['G2013'].str.replace('%', '').astype(int)
google_trends_data['G2015'] = google_trends_data['G2015'].str.replace('%', '').astype(int)

# Show intermediate result: after cleaning percentage symbols and converting to integers
print(f"\nData after cleaning percentage symbols and converting to integers (shape: {google_trends_data.shape}):")
print(google_trends_data.head())

# Calculate the Future Orientation Index (foi) as the ratio of G2015 to G2013
google_trends_data = google_trends_data.assign(foi=google_trends_data['G2015'] / google_trends_data['G2013'])

# Show intermediate result: after calculating factor of increase (foi)
print(f"\nData after calculating the factor of increase (foi) (shape: {google_trends_data.shape}):")
print(google_trends_data.head())

# Export the processed DataFrame to a new CSV file
google_trends_data.to_csv(path_or_buf='google_trends_data.csv', index=False)
print("\nProcessed data has been saved to 'google_trends_data.csv'")

# ==========================================================
# Combine DataFrames and Visualize
# ==========================================================

# Merge the World Bank and Google Trends data on the "Country" column
combined_data = world_bank_data.merge(
    right=google_trends_data,
    on='Country'
)

# Show intermediate result: combined data
print(f"\nCombined data after merging (shape: {combined_data.shape}):")
print(combined_data.head())

# Scatter plot: Future Orientation Index vs GDP per capita
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['foi'], combined_data['NY.GDP.PCAP.PP.KD'], alpha=0.7)
plt.title("GDP per Capita vs Future Orientation Index", fontsize=16)
plt.xlabel("Future Orientation Index (foi)", fontsize=12)
plt.ylabel("GDP per Capita (NY.GDP.PCAP.PP.KD)", fontsize=12)
plt.grid(True)
plt.show()

# ==========================================================
# Pearson Correlation and Permutation Test
# ==========================================================

# Calculate Pearson correlation for the original data
pearson_original = stats.pearsonr(combined_data['foi'], combined_data['NY.GDP.PCAP.PP.KD'])
print(f"Pearson correlation (original data): {pearson_original}")
print(f'The Pearson correlation coefficient is {pearson_original.statistic:.2f}, indicating a moderate positive relationship,\nand the p-value of {pearson_original.pvalue:.2e} suggests this result is statistically significant.')

# Shuffle the 'foi' column and create a new DataFrame
data_shuffled = combined_data.copy()
data_shuffled['foi'] = combined_data['foi'].sample(frac=1).values

# Plot the shuffled data
plt.figure(figsize=(10, 6))
plt.scatter(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'], alpha=0.7)
plt.title('GDP per Capita vs Shuffled Future Orientation Index', fontsize=16)
plt.xlabel('Shuffled Future Orientation Index (foi)', fontsize=12)
plt.ylabel('GDP per Capita (NY.GDP.PCAP.PP.KD)', fontsize=12)
plt.grid(True)
plt.show()

# Calculate Pearson correlation for shuffled data
pearson_shuffled = stats.pearsonr(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'])
print(f"Pearson correlation (shuffled data): {pearson_shuffled}")
print(f'The Pearson correlation coefficient is {pearson_shuffled.statistic:.2f}, indicating a very weak positive relationship,\nand the p-value of {pearson_shuffled.pvalue:.2e} suggests the result is not statistically significant.')
print('Therefore the original correlation is very likely not random.')

# ==========================================================
# Permutation Test
# ==========================================================

# Initialize lists to store statistics and p-values
pearson_statistics = []
pearson_pvalues = []

# Perform 1000 permutations
for _ in range(1000):
    data_shuffled['foi'] = combined_data['foi'].sample(frac=1).values
    pearson_test = stats.pearsonr(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'])
    pearson_statistics.append(pearson_test.statistic)
    pearson_pvalues.append(pearson_test.pvalue)

# ==========================================================
# Visualization of Permutation Test Results
# ==========================================================

# Histogram of permuted correlation coefficients
plt.figure(figsize=(10, 6))
plt.hist(pearson_statistics, bins=30, color='skyblue', edgecolor='black')
plt.axvline(pearson_original.statistic, color='red', linestyle='dotted', linewidth=3)
plt.title('Permutation Test: GDP per Capita vs Future Orientation Index', fontsize=16)
plt.xlabel('Permuted Correlation Coefficients', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# Interpretation
print("The permutation test shows a Gaussian distribution of correlation coefficients ranging approximately from -0.4 to 0.4.")
print(f"The coefficient above {pearson_original.statistic:.2f} lies far outside this range, further supporting the assumption that it is not a random correlation.")

# Histogram of permuted p-values
plt.figure(figsize=(10, 6))
plt.hist(pearson_pvalues, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0.05, color='red', linestyle='dotted', linewidth=2)
plt.title('Permutation Test: p-Values', fontsize=16)
plt.xlabel('Permuted p-Values', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# Count the number of significant p-values (<= 0.05)
significant_count = sum(1 for p in pearson_pvalues if p <= 0.05)
significant_percentage = (significant_count / len(pearson_pvalues)) * 100

# Interpretation
print(f"The p-values of the permutation test are approximately uniformly distributed between 0 and 1.")
print(f"Only {significant_count} values, or {significant_percentage:.2f}%, fall below the threshold of 0.05. This proportion is typical for random correlations.")
print("This result further supports the assumption that the observed correlation is not due to chance.")

