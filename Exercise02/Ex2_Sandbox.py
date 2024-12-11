import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import zipfile # added to open zip-file

# ==========================================================
# 1 Construct the timelines of Twitter users
# ==========================================================

# Getting a list of users
# Forbes top Twitter influencers
# https://twitter.com/i/lists/100791150

# Get list of users from csv
users_list = pd.read_csv("users.csv")
# users list

# Get column names
column_names_users = users_list.columns
# print(column_names_users)

# public_metrics.followers_count >= 100
# public_metrics.tweet_count >= 100
users_filtered = users_list.loc[(users_list["public_metrics.followers_count"] >= 100) & \
  (users_list["public_metrics.tweet_count"] >= 100)]
# print(users_filtered)

# sample 500 users
users_sampled = users_filtered.sample(n=500)
# print(users_sampled)

# Loading timelines
zip_file = zipfile.ZipFile('timeline.zip')
time_lines =pd.read_csv(zip_file.open("timeline.csv"))
# print(time_lines)

# Get column names
column_names_time_lines = time_lines.columns
#print(column_names_time_lines)


# Aggregating and arranging data
# author_id
# public_metrics.retweet_count

# Select specific columns to display
time_lines_selected_columns = time_lines[["author_id", "public_metrics.retweet_count"]]

# Group by author
time_lines_grouped = time_lines_selected_columns.groupby(['author_id']).mean()
# print(time_lines_grouped) # nur 100!

# Merge with user data
merged_data = users_sampled.merge(time_lines_grouped, left_on="id", right_on="author_id")
# print(merged_data) # 88 left
# print(merged_data.columns)

# author ID, name, the follower count and the mean retweet count
dataset = merged_data[["id", "username", "public_metrics.followers_count", "public_metrics.retweet_count"]]
# print(dataset)

# ==========================================================
# 2 Visualize distributions and scatter plots
# ==========================================================

# Distribution of the number of followers
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
fig1.suptitle('Distribution of the Number of Followers')
# Original Followers Count
ax1.hist(dataset["public_metrics.followers_count"])
ax1.set_title("Followers Count")
ax1.set_xlabel("Number of Followers")
ax1.set_ylabel("Number of Authors by Number of Followers")
# Followers Count Logarithmic
ax2.hist(dataset["public_metrics.followers_count"])
ax2.set_title("Followers Count Logarithmic")
ax2.set_xlabel("Number of Followers")
ax2.set_ylabel("Number of Authors by Number of Followers (Log Scale)")
ax2.set_yscale("log")
plt.show()
fig1.savefig("histo_follower.png")


# Distribution of Average Number of Retweets (Social Impact)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
fig2.suptitle('Distribution of Social Impact')
# Original Social Impact
ax1.hist(dataset["public_metrics.retweet_count"])
ax1.set_title("Social Impact")
ax1.set_xlabel("Average Number of Retweets (Social Impact)")
ax1.set_ylabel("Number of Authors by Social Impact")
# Social Impact Logarithmic
ax2.hist(dataset["public_metrics.retweet_count"])
ax2.set_title("Social Impact Logarithmic")
ax2.set_xlabel("Average Number of Retweets (Social Impact)")
ax2.set_ylabel("Number of Authors by Social Impact (Log Scale)")
ax2.set_yscale("log")
plt.show()
fig2.savefig("histo_impact.png")

# Number of followers vs social impact
plt.figure(figsize=(10, 6))
plt.scatter(dataset["public_metrics.followers_count"], dataset["public_metrics.retweet_count"], alpha=0.7)
plt.title("Number of Followers vs Social Impact", fontsize=16)
plt.xlabel("Followers Count Logarithmic", fontsize=12)
plt.ylabel("Social Impact Logarithmic", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()

# ==========================================================
# 4. Fit and visualize a regression model
# ==========================================================

# Add new Columns with logarithmic values
dataset = dataset.assign(SI=np.log(dataset["public_metrics.retweet_count"]))
dataset = dataset.assign(FC=np.log(dataset["public_metrics.followers_count"]))

# target = dataset["SI"].values
# features = dataset["FC"].values

# Calculate Linear Regression
target = dataset["SI"].values
features = dataset["FC"].values

target = target.reshape(len(target), 1)
features = features.reshape(len(features), 1)

reg = LinearRegression().fit(features, target)
# reg = LinearRegression().fit(features, target)
reg_score = reg.score(features, target)
reg_coef = reg.coef_[0][0]
reg_intercept = reg.intercept_[0]
print(f"R^2 Score: {reg_score:.3f}")
print(f"Coefficient: {reg_coef:.3f}")
print(f"Intercept: {reg_intercept:.3f}")

# This linear regression model has an R^2 score of 0.204, meaning it explains 20.4% of the variance in the target variable.
# The coefficient (0.414) indicates a moderate positive relationship between the feature and the target, while the intercept (9.163) represents the target value when the feature is zero.
# The model's performance is relatively low and could be improved by adding more features or using a non-linear approach.

"""
# Initialize layout
fig3, ax = plt.subplots(figsize=(10, 6))
fig3.suptitle("Linear Regression: Number of Followers vs Social Impact", fontsize=16)
# Add scatterplot
ax.scatter(dataset["public_metrics.followers_count"], dataset["public_metrics.retweet_count"], alpha=0.7, edgecolors="k")
# ToDo ax.x_label("Number of Followers", fontsize=12)
# ToDo ax.y_label("Social Impact", fontsize=12)

# ax.grid(True)

# Fit linear regression via least squares with numpy.polyfit
# It returns an slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
# b, a = np.polyfit(x, y, deg=1)

# y = slope * x + intercept
## Create sequence of 100 numbers from 0 to 100

#xseq = np.linspace(1000, 10000000, num=100)
#xseq = xseq.tolist()
## xseq = xseq.reshape(1, len(xseq))
#yseq = [reg_coef[0] * x for x in xseq]
#yseq = [x + reg_intercept for x in yseq]

# xseq auf logarithmischer Skala
xseq = np.logspace(3, 7, num=100)  # Von 10^3 (1000) bis 10^7 (10,000,000)

# Berechnung der y-Werte basierend auf der logarithmischen Regressionsformel
yseq = [(reg_coef * x) + np.log(reg_intercept) for x in xseq]


# Plot regression line
#ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
# ax.plot(xseq, reg_intercept + reg_coef * xseq, color="r", lw=5)
ax.plot(xseq, yseq)

ax.set_xscale("log")
ax.set_yscale("log")
fig3.show()
fig3.savefig("linear_regression.png")
"""
"""
# X-Werte für die Regressionslinie
log_xseq = np.logspace(3, 7, num=100)  # Von 10^3 bis 10^7 (in der Originalskala)
xseq = np.exp(log_xseq)
# Berechnung der Regressionslinie
log_yseq = reg_coef * log_xseq + reg_intercept  # Logarithmierte Vorhersage
yseq = np.exp(log_yseq)  # Exponentiere zurück in die Originalskala
"""
"""
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(dataset["public_metrics.followers_count"], dataset["public_metrics.retweet_count"], alpha=0.7)
plt.title("Linear Regression: Number of Followers vs Social Impact", fontsize=16)
plt.xlabel("Number of Followers", fontsize=12)
plt.ylabel("Social Impact", fontsize=12)
plt.plot(xseq, yseq, color="red", label="Regression line", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()
"""

# Calculate values for regression line
xseq = np.linspace(dataset["FC"].min(), dataset["FC"].max(), num=100).reshape(-1, 1)
yseq = reg_coef * xseq.flatten() + reg_intercept  # logarithmic prediction

##################################################################
# Plot Linear Regression - Log-Transformed Values on Linear Scales
##################################################################

plt.figure(figsize=(10, 6))

# Scatter points
plt.scatter(dataset["FC"], dataset["SI"], alpha=0.7)

#sns.regplot(
 #   x="FC", y="SI", data=dataset,
  #  scatter_kws={"alpha": 0.7},
 #   line_kws={"color": "green", "linewidth": 4, "label": "Seaborn Regression Line"},
  #  ci=None
#)

# Manually calculated regression line
plt.plot(xseq, yseq, color="red", label="Manually Calculated Regression Line", linewidth=2)

plt.title("Linear Regression: Number of Followers vs Social Impact\n(Log-Transformed Values on Linear Scales)", fontsize=16)
plt.xlabel("Log(Number of Followers)", fontsize=12)
plt.ylabel("Log(Social Impact)", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Calculate quality of fit
dataset = dataset.assign(SI_pred=reg.predict(dataset[["FC"]]))

# Calculate predicted values (y_pred) using the model
SI_pred = reg.predict(dataset[["FC"]])  # Predict SI based on log-transformed FC
dataset = dataset.assign(residual=(dataset["SI"] - dataset["SI_pred"]))