import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pytrends.request import TrendReq
import requests
import datetime
import time


# ==========================================================
# Setup: Google Trends API Connection
# ==========================================================

# Initializing session to retrieve cookie
session = requests.Session()
session.get('https://trends.google.com')
cookies_map = session.cookies.get_dict()
nid_cookie = cookies_map['NID']

# Creating an instance of the TrendReq class
pytrends = TrendReq(hl='en-US', tz=-60, retries=3,
                    requests_args={'headers': {'Cookie': f'NID={nid_cookie}'}})

# ==========================================================
# Google Trends Data: Retrieve Topics
# ==========================================================

# Retrieve topics related to "influenza"
topics = pytrends.suggestions(keyword='influenza')
print("\nAvailable topics for 'influenza':")
print(topics)

# Extract the topic ID (assuming it's the first result)
flu_query = topics[0]['mid']  # Should be /m/0cycc for influenza
print(f"\nSelected topic ID for 'influenza': {flu_query}")
"""
# ==========================================================
# Google Trends Data: Interest Over Time
# ==========================================================

# Build the payload for the specified topic and timeframe
time.sleep(5)  # To avoid being blocked due to rapid requests
pytrends.build_payload(kw_list=[flu_query], geo='US', timeframe='2014-01-01 2018-12-31')

# Retrieve interest over time
time.sleep(5)
flu_interest = pytrends.interest_over_time()

# Filter data to ensure it covers the desired time range
flu_interest_filtered = flu_interest.loc['2014-01-01':'2018-12-31'].reset_index(drop=True)

# Display the filtered data
print("\nFiltered Google Trends data:")
print(flu_interest_filtered)

# Save the filtered data to a CSV file
flu_interest_filtered.to_csv(path_or_buf='flu_interest_filtered.csv', index=False)
"""
flu_interest_filtered = pd.read_csv('flu_interest_filtered.csv')

# ==========================================================
# Load and Process US National Flu Data
# ==========================================================

# Load the CSV file and select columns of interest
flu_data = pd.read_csv(
    'ILINet.csv',
    skiprows=1,
    usecols=['YEAR', 'WEEK', '% WEIGHTED ILI']
)

# Show intermediate result: raw data
print(f"Raw data loaded from 'ILINet.csv' (shape: {flu_data.shape}):")
print(flu_data.head())

# Filter data to keep only observations from 2014 to 2018
flu_data = flu_data.loc[(flu_data['YEAR'] >= 2014) & (flu_data['YEAR'] <= 2018)].reset_index(drop=True)

# Show intermediate result: after filtering for the timeframe 2014-2018
print(f"\nData after filtering rows outside 2014-2018 (shape: {flu_data.shape}):")
print(flu_data.head())

# Export the processed data to a CSV file
flu_data.to_csv('flu_data_filtered.csv', index=False)

# ==========================================================
# Plot Flu Interest vs. US National Data
# ==========================================================

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(flu_interest_filtered[flu_query], flu_data['% WEIGHTED ILI'], alpha=0.7)
plt.title('Flu Interest vs. US National Data (2014 - 2018)', fontsize=16)
plt.xlabel('Search Interest for "Influenza" (0 = low, 50 = half, 100 = peak)', fontsize=12)
plt.ylabel('% Weighted ILI (Influenza-Like Illness) US', fontsize=12)
plt.grid(True)
plt.show()

# ==========================================================
# Pearson Correlation Coefficient
# ==========================================================

# Calculate the Pearson correlation coefficient
pearson_flu = stats.pearsonr(flu_interest_filtered[flu_query], flu_data['% WEIGHTED ILI'])

# Display the result
print("\nPearson correlation coefficient and p-value for Flu Interest vs. US National Data:")
print(f"Pearson correlation coefficient: {pearson_flu.statistic:.2f}")
print(f"P-value: {pearson_flu.pvalue:.2e}")

print("\nInterpretation:")
print(f"The Pearson correlation coefficient is {pearson_flu.statistic:.2f}, indicating a strong positive relationship between search interest for 'influenza' and the percentage of weighted ILI cases.")
print(f"The p-value is extremely low ({pearson_flu.pvalue:.2e}), meaning the result is statistically significant. This strongly supports the hypothesis that search trends for 'influenza' are closely related to actual influenza-like illness activity in the US during the years 2014 to 2018.")

# ==========================================================
# Shuffle Flu Interest Data and Recalculate Correlation
# ==========================================================

# Shuffle the flu interest data
flu_interest_shuffled = flu_interest_filtered.copy()  # Ensure the original data remains intact
flu_interest_shuffled[flu_query] = flu_interest_shuffled[flu_query].sample(frac=1).values

# Scatter plot for shuffled data
plt.figure(figsize=(10, 6))
plt.scatter(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'], alpha=0.7)
plt.title('Shuffled Flu Interest vs. US National Data (2014 - 2018)', fontsize=16)
plt.xlabel('Search Interest for "Influenza" (Shuffled)', fontsize=12)
plt.ylabel('% Weighted ILI (Influenza-Like Illness) US', fontsize=12)
plt.grid(True)
plt.show()

# Calculate the Pearson correlation coefficient for shuffled data
pearson_flu_shuffled = stats.pearsonr(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'])

# Display the result
print("\nPearson correlation coefficient and p-value for Shuffled Flu Interest vs. US National Data:")
print(f"Pearson correlation coefficient: {pearson_flu_shuffled.statistic:.2f}")
print(f"P-value: {pearson_flu_shuffled.pvalue:.2e}")

print("\nInterpretation:")
print(f"The Pearson correlation coefficient is approximately {pearson_flu_shuffled.statistic:.2f}, indicating a very weak positive relationship.")
print(f"The p-value is {pearson_flu_shuffled.pvalue:.2e}, which is far above the significance threshold of 0.05. This means the correlation is not statistically significant, strongly suggesting that the original observed correlation was not due to random chance.")

# ==========================================================
# Permutation Test: Flu Interest vs. US National Data
# ==========================================================

# Prepare lists to store permutation results
pearson_flu_statistics = []
pearson_flu_pvalues = []

# Perform 1000 permutations
for _ in range(1000):
    # Shuffle the flu interest data
    flu_interest_shuffled = flu_interest_filtered.copy()
    flu_interest_shuffled[flu_query] = flu_interest_shuffled[flu_query].sample(frac=1).values

    # Calculate Pearson correlation for shuffled data
    pearson_flu_shuffled = stats.pearsonr(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'])
    pearson_flu_statistics.append(pearson_flu_shuffled.statistic)
    pearson_flu_pvalues.append(pearson_flu_shuffled.pvalue)

# ==========================================================
# Visualize Permutation Test Results
# ==========================================================

# Histogram of permuted correlation coefficients
plt.figure(figsize=(10, 6))
plt.hist(pearson_flu_statistics, bins=30, color='skyblue', edgecolor='black')
plt.axvline(pearson_flu.statistic, color='red', linestyle='dotted', linewidth=2)
plt.title('Permutation Test:\nInfluenza Interest vs. US National Data (2014 - 2018)', fontsize=16)
plt.xlabel('Permuted Correlation Coefficients', fontsize=12)
plt.ylabel('Count of Permuted Values', fontsize=12)
plt.grid(True)
plt.show()

print("\nThe permutation test shows nearly a Gaussian distribution of correlation coefficients ranging approximately from -0.2 to 0.2.")
print(f"The observed correlation coefficient of {pearson_flu.statistic:.2f} lies very far outside this range, further supporting the assumption that it is not a random correlation.")

# Histogram of permuted p-values
plt.figure(figsize=(10, 6))
plt.hist(pearson_flu_pvalues, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0.05, color='red', linestyle='dotted', linewidth=2)
plt.title('Permutation Test:\nInfluenza Interest vs. US National Data (2014 - 2018)', fontsize=16)
plt.xlabel('Permuted p-Values', fontsize=12)
plt.ylabel('Count of Permuted Values', fontsize=12)
plt.grid(True)
plt.show()

print("\nThe analysis of the p-values shows that they are approximately uniformly distributed between 0 and 1,")
print("with fewer than 5% being significant (p-value <= 0.05). This supports the assumption that the observed correlation is not due to chance.")

# ==========================================================
# Count Significant Results
# ==========================================================

# Count the number of significant results (p-value <= 0.05)
significant_permutations_flu = sum(1 for p in pearson_flu_pvalues if p <= 0.05)

# Display the result
print(f"\nNumber of significant permutations (p-value <= 0.05): {significant_permutations_flu}")
print(f"Percentage of significant permutations: {(significant_permutations_flu / len(pearson_flu_pvalues)) * 100:.2f}%")