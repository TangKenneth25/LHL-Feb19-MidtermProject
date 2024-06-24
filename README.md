# Data Science Midterm Project

Data Scientist Flex February 2024 Cohort <br>
Group Members: Bimpe Adeleke, Ken Wall, Kenneth Tang

## Project/Goals
### To use a supervised machine learning model to predict the selling price of a house, based on a wide variety of features.
<p>Predicting home prices has significant business implications across a variety of sectors, including:</p>

1. Real estate investing

2. Government resource allocation

3. Mortgages and lending

4. Construction and development companies

5. Economic analysis


## Process
### 1. Expolaratory data analysis - Import and clean the dataset. Handle missing values and outliers
<p>The dataset was provided to our group as part of the project as numerous JSON files by city containing a wide variety of features related to the sale of houses, including the:

1. sale price
2. number of bedrooms
3. number of bathrooms
4. City
5. etc.

We parsed the JSON data and looped throut the numerous files using a self-defined function keep_keys() that had every variable we wanted to keep.</P>

```
directory = r'../raw_data'

keepKeys = keep_keys()
housingData = pd.DataFrame(columns=keepKeys)

for file in os.listdir(directory):
    if file != '.gitkeep':
        result = json.load(open(rf'{directory}/{file}'))

        df = parse_json_data(result['data']['results'],keepKeys)
        housingData = pd.concat([housingData,df],ignore_index=True)
```
<p>Using .info() and .describe() we discovered that the tags column is made up a list of various tags like that needs to be dealt with using our self-defined one-hot-encoding function later in the EDA: 

1. central_air
2. dishwasher
3. basement
4. etc.



There were also numerous columns that had many null values that would need to be filled some how or dropped. We then went feature by feature and determine how to deal with the null values in each column. In total we have 41 columns to start with.</p>

![Raw Null Values](/images/housingDataInfoRaw.png "Info Raw")

<p>There were many columns that had no value in them so we dropped them, which resulted in 10 columns being dropped for a revised total of 31 columns.</p>

```
# Getting columns that are completely empty
columnCount = housingData.describe().loc['count']
emptyColumns = columnCount[columnCount == 0].index

# Dropping columns
housingData.drop(columns=emptyColumns,inplace=True)
```

<p>We then performed exploratory data analysis on the numerical and categorical columns, looking at the non-null values. We looked at the unique items and the value counts of each feature. Based on the analysis there were additional 4 columns that didn't add or were provided redundant values so they were dropped as well resulting in a dataframe of 27 columns.</p>

<p>The next step was to fill the null values or drop the null rows if necessary. We evaluated each column on its own. For example: replacing the longitude and latitude with the average for the city and replacing the null values in the baths, stories, garage, etc with zero. After all our cleaning and null replacements we are left with 25 columns and 6,598 rows</p>

```
#Get the average longitude and latitude by city
avgLongitude = housingData.groupby('location.address.city')['location.address.coordinate.lon'].transform('mean')

avgLatitude = housingData.groupby('location.address.city')['location.address.coordinate.lat'].transform('mean')

housingData['location.address.coordinate.lon'] = housingData['location.address.coordinate.lon'].fillna(avgLongitude) # fill in Nan values with avg longitude of the city

housingData['location.address.coordinate.lat'] = housingData['location.address.coordinate.lat'].fillna(avgLatitude) # fill in Nan values with avg latitude of the city

#Some cities do not have a lat long so we cannot get the mean, we will drop these rows (22) for now. In the future we can get this lat long data from an online sourse.
housingData = housingData.dropna(subset=['location.address.coordinate.lat'])

#creating a subset for columns with null values that can be replaced with 0
description_null_cols = ['description.baths_3qtr', 'description.baths_full', 'description.baths_half', 'description.baths', 'description.stories', 'description.beds', 'description.garage']

#To fill all null values with 0
def fill_nulls(df, column_subset):
    for column in column_subset:
        df[column].fillna(0, inplace=True)

fill_nulls(housingData, description_null_cols)
```

#### Dealing with tags
1. First we created a function to determint the unique tags and their frequency
```
def count_word_frequencies(df, column_name):
    word_freq = defaultdict(int)
    
    #Loop through each row in the specified column
    for row in df[column_name]:
        if isinstance(row, list):
            words = row
        elif isinstance(row, str):
            # Clean the string and split by comma
            words = row.strip("[]").replace("'", "").split(',')
        else:
            continue  # Skip NaN or unexpected types
        
        for word in words:
            clean_word = word.strip()
            word_freq[clean_word] += 1  # Count frequencies
    
    return dict(word_freq)


#Calculate word frequencies in the 'tags' column
word_freq = count_word_frequencies(housingData, 'tags')
```

2. We then sorted the tags in ascending order for the 16 most frequent values based on desired feature tags.

```
sorted_tags = sorted(word_freq, key=word_freq.get, reverse=True)[0:16]
```

3. Manually perform OHE based on the sorted tags from step 2. We created a user defined function to help with this.

```
def one_hot_encode_tags(tags, specific_tags):
    """ 
    Encodes tags from housing data
    Params:
        tags (str): tags for one row in housing data
        specific_tags (list): specific tags to encode
    Returns:
        DataFrame of encoded tags
    """
    ohe_dict = {tag: False for tag in specific_tags}
    ohe_dict['other'] = False

    if len(tags) == 0:
        tags = []
    else: 
        tags = eval(tags)

    for tag in tags:
        if tag in specific_tags:
            ohe_dict[tag] = True
        else:
            ohe_dict['other'] = True
    return pd.Series(ohe_dict)
```
<p>We the  created a list of OHE tags and with that list created an encoded dataframe using the data and then concatenated that dataframe with the cleaned housing dataframe from above.</p>

```
encodedTagsList = []
for row in housingData['tags']:
    encodedTagsList.append(one_hot_encode_tags(row,sorted_tags))

encodedTagsDF = pd.DataFrame(encodedTagsList)

housingData = pd.concat([housingData,encodedTagsDF],axis=1)
housingData.drop(columns=['tags'],inplace=True)
```

#### Visualization

<p>Our first step to view the cleaned data is still to first use .info() and .describe() to look for any possible outliers. After the initial screen we looked at a correlation heatmap of all the numerical data including the target variable, which is below:</p>

![Numerical Heatmap](/images/housingDataNumericalHeatmap.png "Numerical Heatmap")

<p>List price is the highest correlated feature with ourt target variableat 0.99 followed by price reductions at 0.5, square footage at 0.39 and bathrooms full at 0.37.</p>
</br>

<p>We looked at a histogram of sold prices and noticed that the results are extremely skewed. See figure below. When you look at .describe() the mean is about $400k, but the max price is $27M, which severely skews the data as can be seen in the scatterplots to be analyzed below.</p>

![Sold Prices Histogram](/images/housingDataHistSoldPrices.png "Sold Prices Histogram")

</br>
<p>When you plot the sold price vs the list price you see a strong linear relationship which makes sense, but in the scatter plot you see one extreme outlier at about $100k list price and $27M sold price.</p>

![Scatter Plot: Sold vs. List Price](/images/housingDataScatterSoldList.png "Scatter Plot: Sold vs. List Price")

<p>The outliers were removed and a more linear relationship below is observed</p>

![Scatter Plot: Sold vs. List Price](/images/housingDataScatterSoldList2.png "Scatter Plot: Sold vs. List Price")

</br>
<p>We then did scatter plots for the highest correlated variables to see any trends. The outputs of scatter plots can be seen below. It is difficult to see detail due to the scale of the plot as the result of the outlier which force the data to scale down. It appears that anywhere between 2 - 6 bedrooms and baths will have a higher sales price. Square footage, which you would expect to have a strong linear relationship shows slight gradual incline but is difficult to tell because of the scale.


![Scatter Plot: Sold vs. Bathrooms](/images/housingDataScatterSoldBaths.png "Scatter Plot: Sold vs. Bathrooms")
![Scatter Plot: Sold vs. Bedrooms](/images/housingDataScatterSoldBeds.png "Scatter Plot: Sold vs. Bedrooms")
![Scatter Plot: Sold vs. Lot square footage](/images/housingDataScatterSoldLotSqft.png "Scatter Plot: Sold vs. Lot square footage")
![Scatter Plot: Sold vs. Square footage](/images/housingDataScatterSoldSqft.png "Scatter Plot: Sold vs. Square footage")
![Scatter Plot: Sold vs. Price reduced](/images/housingDataScatterSoldPriceReduced.png "Scatter Plot: Sold vs. Price reduced")
![Scatter Plot: Sold vs. Year built](/images/housingDataScatterSoldYearBuilt.png "Scatter Plot: Sold vs. Year build")

</br>
<p>Next we wanted to determine the correlation between numerical variables(less target variable) to detect highly correlated features that we may want to drop. Baths and baths_full have a correlation of 0.86 and both may not be useful to the model given such a high correlation. It was decided to drop baths_full</p>

![Heatmap NonTarget](/images/housingDataNonTargetHeatMap.png "Heatmap NonTarget")


### 2. Model Selection - evaluate multiple models and compare their performance
<p>After preprocessing, scaling, splitting into train and test we were ready to fit the various models outlined below</p>

#### Model 1 - Linear Regression
```
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_LR = model.predict(X_test)

mse_LR = mean_squared_error(y_test, y_pred_LR)
r2_LR = r2_score(y_test[100:200], y_pred_LR[100:200])
```

- Mean squared error = 4325054394
- R squared = 0.98094

#### Model 2 - Decision Tree
```
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
```

- Mean squared error = 58515925
- R squared = 0.99948

#### Model 3 - Random Forest
```
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)
```

- Mean squared error = 76713112
- R squared = 0.999319

#### Model 4 - Gradient Boost
```
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
```

- Mean squared error = 660330741
- R squared = 0.99414

#### Model 5 - Support Vector Regression
```
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

y_pred_svr = svr_model.predict(X_test)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test[100:200], y_pred_svr[100:200])
```

- Mean squared error = 118806954502
- R squared = -0.041469

#### Model Conclusion
<p>
Our overall model performance was very well accross all models, with R Squared above 98% for all models, except SVR. However, based on the model testwork performed above the decision tree model performed the best having the best R Squared of 0.9995. The decision tree model also had the lowest mean squared error of 58515925. Random forest also performed well with an R Squared 0.9993.
</p>

### 3. Tuning and pipelining - creating a pipeline for easier retraining


## Challenges 
1. Data leakage
2. Dealing with tags took extra time
3. 

## Future Goals
(what would you do if you had more time?)
