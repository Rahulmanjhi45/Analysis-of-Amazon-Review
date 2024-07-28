import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/Reviews.csv'
data = pd.read_csv(file_path)

# Data Cleaning
# Check for missing values
missing_values = data.isnull().sum()

# Convert 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'], unit='s')

# Hypothesis 1: Distribution of review scores
plt.figure(figsize=(10, 6))
sns.countplot(data['Score'])
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Count')
plt.show()

# Hypothesis 2: Helpfulness comparison
helpfulness_ratio = data['HelpfulnessNumerator'] / data['HelpfulnessDenominator']
helpfulness_ratio = helpfulness_ratio.fillna(0)

plt.figure(figsize=(10, 6))
sns.histplot(helpfulness_ratio, bins=30, kde=True)
plt.title('Distribution of Helpfulness Ratio')
plt.xlabel('Helpfulness Ratio')
plt.ylabel('Count')
plt.show()

# Hypothesis 3: Average review scores for different products
product_scores = data.groupby('ProductId')['Score'].mean().sort_values(ascending=False)
top_10_products = product_scores.head(10)

plt.figure(figsize=(10, 6))
top_10_products.plot(kind='bar')
plt.title('Top 10 Products by Average Review Score')
plt.xlabel('Product ID')
plt.ylabel('Average Review Score')
plt.show()

# Hypothesis 4: Correlation between review text length and review score
data['ReviewLength'] = data['Text'].apply(len)
correlation = data[['Score', 'ReviewLength']].corr().iloc[0, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='ReviewLength', y='Score', data=data)
plt.title(f'Review Length vs. Review Score (Correlation: {correlation:.2f})')
plt.xlabel('Review Length')
plt.ylabel('Review Score')
plt.show()

# Hypothesis 5: Top 10 most active reviewers
top_10_reviewers = data['UserId'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_10_reviewers.plot(kind='bar')
plt.title('Top 10 Most Active Reviewers')
plt.xlabel('User ID')
plt.ylabel('Number of Reviews')
plt.show()

# Hypothesis 6: Average length of reviews over time
data['YearMonth'] = data['Time'].dt.to_period('M')
average_review_length_over_time = data.groupby('YearMonth')['ReviewLength'].mean()

plt.figure(figsize=(10, 6))
average_review_length_over_time.plot()
plt.title('Average Length of Reviews Over Time')
plt.xlabel('Time')
plt.ylabel('Average Review Length')
plt.show()

# Hypothesis 7: Relationship between review scores and helpfulness votes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='HelpfulnessNumerator', data=data)
plt.title('Review Score vs. Helpfulness Votes')
plt.xlabel('Review Score')
plt.ylabel('Helpfulness Votes')
plt.show()

# Hypothesis 8: Relationship between number of reviews and average score
product_review_counts = data.groupby('ProductId').size()
product_average_scores = data.groupby('ProductId')['Score'].mean()
correlation_reviews_scores = product_review_counts.corr(product_average_scores)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=product_review_counts, y=product_average_scores)
plt.title(f'Number of Reviews vs. Average Score (Correlation: {correlation_reviews_scores:.2f})')
plt.xlabel('Number of Reviews')
plt.ylabel('Average Score')
plt.show()

# Hypothesis 9: Comparison of negative and positive review lengths
negative_reviews = data[data['Score'] < 3]
positive_reviews = data[data['Score'] > 3]

plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='ReviewLength', data=data)
plt.title('Review Score vs. Review Length')
plt.xlabel('Review Score')
plt.ylabel('Review Length')
plt.show()

# Hypothesis 10: Average review scores and helpfulness ratios for top 10 products
top_10_products_scores = data[data['ProductId'].isin(top_10_products.index)]
top_10_products_helpfulness = top_10_products_scores.groupby('ProductId').apply(lambda x: (x['HelpfulnessNumerator'] / x['HelpfulnessDenominator']).mean())

fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()
top_10_products_scores.groupby('ProductId')['Score'].mean().plot(kind='bar', ax=ax1, color='g', position=0, width=0.4)
top_10_products_helpfulness.plot(kind='bar', ax=ax2, color='b', position=1, width=0.4)

ax1.set_ylabel('Average Review Score')
ax2.set_ylabel('Average Helpfulness Ratio')
ax1.set_xlabel('Product ID')
plt.title('Top 10 Products by Average Review Score and Helpfulness Ratio')
plt.show()
