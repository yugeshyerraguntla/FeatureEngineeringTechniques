# FeatureEngineeringTechniques

Feature engineering has two goals primarily:

1. Preparing the proper input dataset, compatible with the machine learning algorithm requirements
2. Improving the performance of machine learning models

###List of Techniques
1.Imputation
2.Handling Outliers
3.Binning
4.Log Transform
5.One-Hot Encoding
6.Grouping Operations
7.Feature Split
8.Scaling
9.Extracting Date


__1.Imputation: Handling Missing Values__
Most simple solution to the missing values is to drop the rows or the entire column. There is not an optimum threshold for dropping but you can use 70% as an example value and try to drop the rows and columns which have missing values with higher than this threshold.

threshold = 0.7
######Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

######Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]

_-Numerical Imputation_
######Filling all missing values with 0
data = data.fillna(0)
######Filling missing values with medians of the columns
data = data.fillna(data.median())

_-Categorical Imputation:_ Replacing the missing values with the maximum occurred value
_-Random Sample Imputation:_ Taking random observation from the dataset and we use this observation to replace the NaN values


__2.Handling Outliers__
  1. Naive Bayes        -  Not Sensitive
  2. SVM                -  Not Sensitive
  3. Linear Reg         -  Sensitive
  4. Logistic Reg       -  Sensitive
  5. DT Reg/Class       -  Not Sensitive
  6. Ensemble(RF,XG,GB) -  Not Sensitive
  7. KNN                -  Not Sensitive
  8. Kmeans             -  Sensitive
  9. Hierarchial        -  Sensitive
  10. PCA               -  Sensitive
  11. Neural Networks   -  Sensitive

1. Vizualize with graphs and box plot
2. Outliers in terms of SD
factor = 3
upper_lim = data['column'].mean () + data['column'].std () * factor
lower_lim = data['column'].mean () - data['column'].std () * factor

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
3. Outliers in Percentile
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)



__3.Binning__
Creating Bins to data. Can be for both categorical and numerical.



__4. Log Transform__
-helps to handle skewed data and after transformation, the distribution becomes more approximate to normal
-decreases the effect of the outliers due to the normalization of magnitude differences and the model become more robust
-must have only positive values, otherwise you receive an error

data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})
data['log+1'] = (data['value']+1).transform(np.log)



__5. OneHot Encoding - (get_dummies function of Pandas)__
-This method spreads the values in a column to multiple flag columns and assigns 0 or 1 to them. These binary values express the relationship between grouped and encoded column.
-changes your categorical data, which is challenging to understand for algorithms, to a numerical format and enables you to group your categorical data without losing any information.
get_dummies function of Pandas
encoded_columns = pd.get_dummies(data['column'])
data = data.join(encoded_columns).drop('column', axis=1)



__6. Grouping Operations__
Categorical Column Grouping

-The first option is to select the label with the highest frequency. In other words, this is the max operation for categorical columns, but ordinary max functions generally do not return this value, you need to use a lambda function for this purpose.
data.groupby('id').agg(lambda x: x.value_counts().index[0])

-Second option is to make a pivot table. 

__7. Feature Split__
data.name
0  Luther N. Gonzalez
1    Charles M. Young

#Extracting first names
data.name.str.split(" ").map(lambda x: x[0])
0     Luther
1    Charles

#Extracting last names
data.name.str.split(" ").map(lambda x: x[-1])
0    Gonzalez
1       Young

__8. Scaling__
In most cases, the numerical features of the dataset do not have a certain range and they differ from each other. In order for a symmetric dataset, scaling is required.

Normalization
Normalization (or min-max normalization) scales all values in a fixed range between 0 and 1. This transformation does not change the distribution of the feature and due to the decreased standard deviations, the effects of the outliers increases. Therefore, before normalization, it is recommended to handle the outliers

data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})

data['normalized'] = (data['value'] - data['value'].min()) / (data['value'].max() - data['value'].min())

Standardization
Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features is different, their range also would differ from each other. This reduces the effect of the outliers in the features.

data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})

data['standardized'] = (data['value'] - data['value'].mean()) / data['value'].std()


__9. Date Splitting__
#Transform string to date
data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

#Extracting Year
data['year'] = data['date'].dt.year

#Extracting Month
data['month'] = data['date'].dt.month

#Extracting passed years since the date
data['passed_years'] = date.today().year - data['date'].dt.year

#Extracting passed months since the date
data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

#Extracting the weekday name of the date
data['day_name'] = data['date'].dt.day_name()


https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

https://www.analyticsvidhya.com/blog/2020/10/7-feature-engineering-techniques-machine-learning/


