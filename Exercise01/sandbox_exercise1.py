import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


wb_data = wb.data.DataFrame(['NY.GDP.PCAP.PP.KD', 'IT.NET.USER.ZS', 'SP.POP.TOTL'], time=2015, skipAggs=True, labels=True).reset_index()
wb_data = wb_data.dropna()
# calculate total number of internet users
wb_data = wb_data.assign(internet_user=wb_data['SP.POP.TOTL'] * wb_data['IT.NET.USER.ZS'] / 100)
# keep rows where there are at least 5 Million internet users
wb_data = wb_data[wb_data['internet_user'] >= 5e+06]
wb_data.to_csv(path_or_buf='wb_data.csv')

# Load the .csv file in a pandas data frame
gt_data = pd.read_csv('geoMap.csv', skiprows=3, names=['Country', 'G2013', 'G2015'])
# Remove all rows containing NaN
gt_data = gt_data.dropna()
# Remove % symbol and convert value to int
gt_data['G2013'] = gt_data['G2013'].str.replace('%', '')
gt_data['G2015'] = gt_data['G2015'].str.replace('%', '')
gt_data['G2013'] = gt_data['G2013'].astype(int)
gt_data['G2015'] = gt_data['G2015'].astype(int)

# calculate foi
gt_data = gt_data.assign(foi=gt_data['G2015'] / gt_data['G2013'])
gt_data.to_csv(path_or_buf='gt_data.csv')

print(gt_data)
print(wb_data)


wb_data = pd.read_csv('wb_data.csv')
gt_data = pd.read_csv('gt_data.csv')

data = wb_data.merge(right=gt_data, on='Country')
# print(data)

#plt.scatter(data['foi'], data['NY.GDP.PCAP.PP.KD'])
plt.title('GDP per capita vs. Future Orientation Index')
plt.xlabel('Future Orientation Index')
plt.ylabel('GDP per capita')
#plt.show()

pearson_2014 = stats.pearsonr(data['foi'], data['NY.GDP.PCAP.PP.KD'])
print(pearson_2014)

# Data shuffled
data_shuffled = data
data_shuffled['foi'] = data_shuffled['foi'].sample(frac=1).values

plt.scatter(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'])
plt.title('GDP per capita vs. Shuffled Future Orientation Index')
plt.xlabel('Future Orientation Index')
plt.ylabel('GDP per capita')
plt.show()

pearson_shuffled = stats.pearsonr(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'])

# Permutation Test
data_shuffled = data
pearson_statistics = []
pearson_pvalues = []

for i in range(0, 999):
    data_shuffled['foi'] = data_shuffled['foi'].sample(frac=1).values
    pearson_shuffled = stats.pearsonr(data_shuffled['foi'], data_shuffled['NY.GDP.PCAP.PP.KD'])
    pearson_statistics.append(pearson_shuffled.statistic)
    pearson_pvalues.append(pearson_shuffled.pvalue)
    i += 1

#print(pearson_statistics)
#print(pearson_pvalues)

plt.hist(pearson_statistics)
plt.axvline(pearson_2014.statistic, color='r', linestyle = 'dotted', linewidth=3)
plt.title('Permutation Test:\n GDP per capita vs. Future Orientation Index')
plt.xlabel('Permuted Correlation Coefficients')
plt.ylabel('Count of Permuted Values')
plt.show()

# p-Values
plt.hist(pearson_pvalues)
plt.axvline(0.05, color='r', linestyle = 'dotted', linewidth=2)
plt.title('Permutation Test:\n GDP per capita vs. Future Orientation Index')
plt.xlabel('Permuted p-Values')
plt.ylabel('Count of Permuted Values')
plt.show()

significant = 0

for j in range (0, 999):
    if pearson_pvalues[j] <= 0.05:
        significant += 1

print (significant)