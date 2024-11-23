import wbgapi as wb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

GoogleDF = pd.read_csv("geoMap.csv", skiprows=3, names=["Country", "G2013", "G2015"])

GoogleDF.dropna(axis=0, how='any', inplace=True)

GoogleDF['G2013'] = GoogleDF['G2013'].str.replace('\%','', regex=True).astype(int)

GoogleDF['G2015'] = GoogleDF['G2015'].str.replace('\%','', regex=True).astype(int)

GoogleDF.head()

print(GoogleDF)

