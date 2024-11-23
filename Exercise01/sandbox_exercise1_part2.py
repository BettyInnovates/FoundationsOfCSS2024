import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pytrends.request import TrendReq
import requests
import datetime
import time


# getting cookie
session = requests.Session()
session.get('https://trends.google.com')
cookies_map = session.cookies.get_dict()
nid_cookie = cookies_map['NID']

# creating an instance of the TrendReq class
pytrends = TrendReq(hl='en-US', tz=-60, retries=3,
                    requests_args={'headers': {'Cookie': f'NID={nid_cookie}'}})

topics = pytrends.suggestions(keyword='influenza')
print(topics)

flu_query = topics[0]['mid']  # should be /m/0cycc

# building payload
time.sleep(5)
pytrends.build_payload(kw_list=[flu_query], geo='US',
                       timeframe=['2014-01-01 2018-12-31'])

# flu interest over time
time.sleep(5)
flu_interest = pytrends.interest_over_time()
# we also need to drop first observation since that week started in 2014
flu_interest_filtered = flu_interest[flu_interest.index >= datetime.datetime(2014, 1, 1)]
print(flu_interest_filtered)
# flu_interest_filtered.to_csv(path_or_buf='flu_interest_filtered.csv')

flu_interest_filtered_alternative = flu_interest.loc['2014-1-1':'2018-12-31'].reset_index(drop=True)
print(flu_interest_filtered_alternative)
# flu_interest_filtered_alternative.to_csv(path_or_buf='flu_interest_filtered_alternative.csv')


# store columns of interest as a pandas dataframe
flu_data = pd.read_csv('ILINet.csv', skiprows=1, usecols=['YEAR', 'WEEK', '% WEIGHTED ILI'])
# drop the rows which store observations from before 2014, or later than 2018
flu_data  = flu_data.loc[(flu_data['YEAR'] >= 2014) & (flu_data['YEAR'] <= 2018)].reset_index(drop=True)  # dropping rows outside the timeframe
print(flu_data)

flu_interest = pd.read_csv('flu_interest_filtered_alternative.csv')

plt.scatter(flu_interest[flu_query], flu_data['% WEIGHTED ILI'])
plt.title('Flu interest vs. US National data  (2014 - 2018)')
plt.xlabel('Search Interest for \'Influenza\' (0 = low, 50 = half, 100 = peak)')
plt.ylabel('% Weighted ILI (Influenza-Like Illness) US')
plt.show()

pearson_flu= stats.pearsonr(flu_interest[flu_query], flu_data['% WEIGHTED ILI'])
print(pearson_flu)



flu_interest_shuffled = flu_interest
flu_interest_shuffled[flu_query] = flu_interest_shuffled[flu_query].sample(frac=1).values

plt.scatter(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'])
plt.title('Shuffled flu interest vs. US National data  (2014 - 2018)')
plt.xlabel('Search Interest for \'Influenza\' (0 = low, 50 = half, 100 = peak)')
plt.ylabel('% Weighted ILI (Influenza-Like Illness) US')
plt.show()

pearson_flu_shuffled = stats.pearsonr(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'])
print(pearson_flu_shuffled)

# Permutation Test

flu_interest_shuffled = flu_interest
pearson_flu_statistics = []
pearson_flu_pvalues = []


for i in range(0, 999):
    flu_interest_shuffled[flu_query] = flu_interest_shuffled[flu_query].sample(frac=1).values
    pearson_flu_shuffled = stats.pearsonr(flu_interest_shuffled[flu_query], flu_data['% WEIGHTED ILI'])
    pearson_flu_statistics.append(pearson_flu_shuffled.statistic)
    pearson_flu_pvalues.append(pearson_flu_shuffled.pvalue)
    i += 1

# correlation coefficients
print(pearson_flu_statistics)
plt.hist(pearson_flu_statistics)
plt.axvline(pearson_flu.statistic, color='r', linestyle = 'dotted', linewidth=2)
plt.title('Permutation Test:\n Influenza Interest vs. US National data  (2014 - 2018)')
plt.xlabel('Permuted Correlation Coefficients')
plt.ylabel('Count of Permuted Values')
plt.show()


# p-Values
print(pearson_flu_pvalues)
plt.hist(pearson_flu_pvalues)
plt.axvline(0.05, color='r', linestyle='dotted', linewidth=2)
plt.title('Permutation Test:\n Influenza Interest vs. US National data  (2014 - 2018)')
plt.xlabel('Permuted p-Values')
plt.ylabel('Count of Permuted Values')
plt.show()

# count number of significant Results (p-Value <= 0.05) in Permutation Test
significant_permutations_flu = 0

for j in range(0, 999):
    if pearson_flu_pvalues[j] <= 0.05:
        significant_permutations_flu += 1

print(significant_permutations_flu)