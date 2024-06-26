# Libraries
import pandas as pd
from collections import defaultdict
from pandas import json_normalize


# Functions used to parse through files and perform EDA


# Keys we decided to keep during group meeting. Keys only gathered from first file, maybe there are more keys in other files that we may want? Spot check a couple. 
def keep_keys():
    """
    Returns list of keys determined in group meeting to be kept for dataframe
    ['tags','status','list_date','open_houses','description.year_built',
    'description.baths_3qtr', 'description.sold_date',
    'description.sold_price', 'description.baths_full', 'description.name',
    'description.baths_half', 'description.lot_sqft', 'description.sqft',
    'description.baths', 'description.sub_type', 'description.baths_1qtr',
    'description.garage', 'description.stories', 'description.beds',
    'description.type','list_price','property_id','flags.is_new_construction', 
    'flags.is_for_rent', 'flags.is_subdivision', 'flags.is_contingent', 
    'flags.is_price_reduced', 'flags.is_pending', 'flags.is_foreclosure', 
    'flags.is_plan', 'flags.is_coming_soon', 'flags.is_new_listing',
    'listing_id','price_reduced_amount','location.address.postal_code',
    'location.address.state', 'location.address.coordinate.lon',
    'location.address.coordinate.lat', 'location.address.city',
    'location.address.state_code', 'location.address.line']
    """
    keys = ['tags','status','list_date','open_houses','description.year_built',
              'description.baths_3qtr', 'description.sold_date',
              'description.sold_price', 'description.baths_full', 'description.name',
              'description.baths_half', 'description.lot_sqft', 'description.sqft',
              'description.baths', 'description.sub_type', 'description.baths_1qtr',
              'description.garage', 'description.stories', 'description.beds',
              'description.type','list_price','property_id','flags.is_new_construction', 
              'flags.is_for_rent', 'flags.is_subdivision', 'flags.is_contingent', 
              'flags.is_price_reduced', 'flags.is_pending', 'flags.is_foreclosure', 
              'flags.is_plan', 'flags.is_coming_soon', 'flags.is_new_listing',
              'listing_id','price_reduced_amount','location.address.postal_code',
              'location.address.state', 'location.address.coordinate.lon',
              'location.address.coordinate.lat', 'location.address.city',
              'location.address.state_code', 'location.address.line']
    
    return keys


# Get list of keys not intended to be used for model analysis
def find_drop_keys(jsonColumns, keepKeys):
    """ 
    Find keys to drop from list of result columns in file
    Parameters:
        jsonColumns: List of columns in file results
        keepKeys: List of keys to be kept
    Returns a list of keys to be dropped from a file for data frame
    """
    dropKeys = []

    for keys in jsonColumns:
        if keys not in keepKeys:
            dropKeys.append(keys)

    return dropKeys


# Convert json list dictionary to dataframe with intended columns
def parse_json_data(jsonResults, keepKeys):
    """ 
    Parses through provided LHL midterm project data file
    Parameters:
        jsonResults: List Dictionary of results from file
        dropKeys: List of keys to be dropped
    Returns a dataframe of preset column keys
    """
    df = json_normalize(jsonResults)

    dropKeys = find_drop_keys(df.columns, keepKeys)
    df.drop(columns=dropKeys,inplace=True)

    return df


# Encodes tags
def one_hot_encode_tags(tags, specific_tags):
    """ 
    Encodes tags from housing data
    Params:
        tags (str): tags for one row in housing data, string of list
        specific_tags (list): specific tags to encode
    Returns:
        DataFrame of encoded tags
    """
    ohe_dict = {tag: False for tag in specific_tags}
    ohe_dict['other_tags'] = False

    if len(tags) == 0:
        tags = []
    else: 
        tags = eval(tags)

    for tag in tags:
        if tag in specific_tags:
            ohe_dict[tag] = True
        else:
            ohe_dict['other_tags'] = True
    return pd.Series(ohe_dict)



#creating a function todetermine the unique tags and their frequency
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