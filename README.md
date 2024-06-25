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
LR_model = LinearRegression()
LR_model.fit(X_train_scaled, y_train)

y_train_LR = LR_model.predict(X_train_scaled)
y_test_LR = LR_model.predict(X_test_scaled)

train_mse_LR = mean_squared_error(y_train, y_train_LR)
train_r2_LR = r2_score(y_train, y_train_LR)

mse_LR = mean_squared_error(y_test, y_test_LR)
r2_LR = r2_score(y_test, y_test_LR)

LR_metrics = {'MSE':mse_LR, 'R2':r2_LR}
```

- Train MSE: 	 3479895584.8480883
- Test MSE: 	 4328005232.184329
- Train R2: 	 0.965973443727785
- Test R2: 	    0.961626506100333

<p>The linear regression model performed really well with over 0.96 R squared for both train and test data. If the test R-squared is significantly lower than the train R-squared, it can be an indication of overfitting. If the Mean Squared Error (MSE) is higher on the test set compared to the training set, it typically indicates that the model is overfitting the training data. However, R squared is the proportion of the variance in the dependent variable that is predictable from the independent variables and was determined to be a better measure of our model success. The difference between the train and test set R squared is so minimal that we can say the model does a good job of predicting the results of the sale price based on the features selected in our EDA process. When we complete our model tuning in the next steps we can help reduce the MSE using regularization and cross-validation.</p>

#### Model 2 - Decision Tree
```
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train_scaled, y_train)

y_train_tree = tree_model.predict(X_train_scaled)
y_test_tree = tree_model.predict(X_test_scaled)

train_mse_tree = mean_squared_error(y_train, y_train_tree)
train_r2_tree = r2_score(y_train, y_train_tree)

mse_tree = mean_squared_error(y_test, y_test_tree)
r2_tree = r2_score(y_test, y_test_tree)

DTree_metrics = {'MSE':mse_tree, 'R2':r2_tree}
```

- Train MSE: 	 0.0
- Test MSE: 	 77004555.55555555
- Train R2: 	 1.0
- Test R2: 	     0.9993172527101206

<p>As with linear regression above R squared is the measure we are more concerned with and is basically 1.0 for both train and test, so this model performs almost perfectly at determining the variance in the dependent variable that is predictable from the independent variables.</p>

#### Model 3 - Random Forest
```
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train_scaled, y_train)

y_train_forest = forest_model.predict(X_train_scaled)
y_test_forest = forest_model.predict(X_test_scaled)

train_mse_forest = mean_squared_error(y_train, y_train_forest)
train_r2_forest = r2_score(y_train, y_train_forest)

mse_forest = mean_squared_error(y_test, y_test_forest)
r2_forest = r2_score(y_test, y_test_forest)

Forest_metrics = {'MSE':mse_forest, 'R2':r2_forest}
```

- Train MSE: 	 7239159.801686489
- Test MSE: 	 75306351.56509332
- Train R2: 	 0.9999292152099539
- Test R2: 	     0.9993323095358342

<p>The random forest model, like decision tree, performs almost indentically between train and test sets with the R square metric of 0.999. There is quite a big difference in MSE, but that could most likely be fixed with hyper parameter tuning.</p>

#### Model 4 - XG Boost
```
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

y_train_xgb = xgb_model.predict(X_train_scaled)
y_test_xgb = xgb_model.predict(X_test_scaled)

train_mse_xgb = mean_squared_error(y_train, y_train_xgb)
train_r2_xgb = r2_score(y_train, y_train_xgb)

mse_xgb = mean_squared_error(y_test, y_test_xgb)
r2_xgb = r2_score(y_test, y_test_xgb)

XGB_metrics = {'MSE':mse_xgb, 'R2':r2_xgb}
```

- Train MSE: 	 4386703.237700574
- Test MSE: 	 29075644.353282813
- Train R2: 	 0.9999571066427346
- Test R2: 	     0.999742205935214

<p>XG Boost performs similarly to random forest in R squared, with a measure of 0.999. However, its MSE is much less, which could possibly indicate a better model and less overfitting.</p>

#### Model 5 - Support Vector Regression
```
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

y_train_svr = svr_model.predict(X_train_scaled)
y_test_svr = svr_model.predict(X_test_scaled)

train_mse_svr = mean_squared_error(y_train, y_train_svr)
train_r2_svr = r2_score(y_train, y_train_tree)

mse_svr = mean_squared_error(y_test, y_test_svr)
r2_svr = r2_score(y_test, y_test_svr)

SVR_metrics = {'MSE':mse_svr, 'R2':r2_svr}
```

- Train MSE: 	 106808398328.74374
- Test MSE: 	 118773156180.84033
- Train R2: 	 1.0
- Test R2: 	     -0.05308120938878913

<p>Model results are not as expected and do not show the model performs well. The other models so far showed strong performance, so this model will be excluded from further analysis.</p>

#### Model Conclusion
<p>
Our overall model performance was very well accross all models, with R Squared above 96% for all models, except SVR. However, based on the model testwork performed above the decision tree, random forest and XG boost for comparative purposed had similar R Squared of basically 1.0. The XG Boost model also had the lowest mean squared error of 29,075,644.35 in the test set. Random forest and the decision tree both had MSE of 70M in the test set, which could indicate XG Boost is a slightly better model. We will choose:

- Random Forest 
- Gradient Boost
- Decision tree
- SVR
</p>

### 3. Tuning and pipelining - creating a pipeline for easier retraining
<p>First we import our processed train, test split data, which has 50 features with 6,597 rows of data.</p>

<p>As suggested in the requirements we developed our own custom function to create n_splits sets of trainin and validatin folds. The code is outlined below:</p>

```
    kfold = KFold(n_splits=n_splits)
    training_folds = []
    validation_folds = []

    for train_index, val_index in kfold.split(train_df):
      train_fold = train_df.iloc[train_index].copy()
      val_fold = train_df.iloc[val_index].copy()

      # Compute city means in the training fold
      city_means = train_fold.groupby(city_column)[target_column].mean().reset_index()
      city_means.columns = [city_column, f'{target_column}_city_mean']

      # Merge these means into the validation fold based on the city column
      val_fold = val_fold.merge(city_means, on=city_column, how='left')

      training_folds.append(train_fold)
      validation_folds.append(val_fold)

    return training_folds, validation_folds
```

<p>We originally used GridsearchCV to do the hyper parameter tuning, but we were finding that with the number of parameter we were trying to tune it was taking to long to process, so we switched to RandomsearchCV.</p>

<p>Initially a custom function was built to perform the grid search:</p>

```
def perform_grid_search_cv(model, param_grid, X, y, cv=5):

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error')
    
    # Fit the GridSearchCV on the housingData
    grid_search.fit(X_train, y_train)

    # Return the best estimator and the best parameters
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
```

#### Random Forest 
```
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train_scaled, y_train)
y_pred_forest = forest_model.predict(X_test_scaled)

mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)
```

```
# RandomForestRegressor
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


grid_search = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

grid_search.fit(X_train_scaled, y_train)
```

#### XG Boost
```
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
```

#### Decision Tree
```

```
#### Best Model
<p>
The best model based on the hyperparameter tuning was _____ with the following parameters:</p>

- parm 1 - 
- parm 2 - 
- parm 3 - 





## Challenges 
1. Data leakage
2. Tags - Most challenging aspect of EDA (Sorting, Ranking JSON data and selecting the top tags and then OHE)
3. 

## Future Goals
1. Stretch activities:
    - EDA - Importing and joining new data to the data set
    - Model Selection - Feature Selection to improve model performance
    - Tuning Pipeline - Implementing a prediction pipeline
2. Model Selection - Trying out additional models on the dataset
3. Tuning Pipeline - Further research on the best paramenters for each model

