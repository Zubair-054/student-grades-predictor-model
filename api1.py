import os
import random
import datetime
import requests
import matplotlib.pyplot as plt

def getAndMergeDataset():
    # API endpoint
    url = 'https://api.ers.usda.gov/data/arms/surveydata?api_key=KgVOmvKBRkV3GkrT726ucs6Qi3rQmjInebRkqnKY&year=2011,2012,2013,2014,2015,2016&variable=vprodtot&category=operator+age,economic+class'
    url2 = 'https://api.ers.usda.gov/data/arms/surveydata?api_key=KgVOmvKBRkV3GkrT726ucs6Qi3rQmjInebRkqnKY&year=2011,2012,2013,2014,2015,2016&variable=tacres&category=operator+age,economic+class'
    # Send the GET request
    response = requests.get(url)
    response2 = requests.get(url2)
    # Check if the request was successful
    if response.status_code == 200 and response2.status_code == 200:
        # Print the response data
        data = response.json()['data']
        data2 = response2.json()['data']
        df = pd.json_normalize(data)
        df2 = pd.json_normalize(data2)
        dropColumns = ['variable_id', 'variable_group', 'variable_group_id', 'median']
        df = df.drop(dropColumns, axis=1)
        df2 = df2.drop(dropColumns, axis=1)
        merged_df = pd.merge(df, df2,
                             on=['year', 'state', 'farmtype', 'report', 'category', 'category_value', 'category2',
                                 'category2_value', 'variable_level', 'variable_is_invalid', 'statistic',
                                 'unreliable_estimate', 'decimal_display'])
        merged_df = merged_df.drop_duplicates()
        return merged_df

    else:
        # Print an error message
        print(f"Request One's status code {response.status_code}")
        print(f"Request Two's status code {response2.status_code}")
        return None
import pandas as pd

def equal_frequency_binning(df, column1, num_bins):
    """
    Perform equal frequency binning on a given column of a DataFrame.

    Args:
        df (DataFrame): Input DataFrame.
        column1 (str): Name of the column to perform binning on.
        num_bins (int): Number of bins to create.

    Returns:
        DataFrame: New DataFrame with an additional column for the bins.
    """
    # Sort the DataFrame by the column1
    df_sorted = df.sort_values(column1)

    # Create a new column for bin labels
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df_sorted['bins'] = pd.qcut(df_sorted[column1], num_bins, labels=labels)

    return df_sorted

def add_random_date(df, new_field_name):
    """
    Add a new field with a random date between "2000-01-01" and "2000-12-31" (without time) in a DataFrame.

    Args:
        df (DataFrame): Input DataFrame.
        new_field_name (str): Name of the new field to be created.

    Returns:
        DataFrame: DataFrame with the new field containing random dates (without time).
    """
    start_date = pd.to_datetime("2000-01-01")
    end_date = pd.to_datetime("2000-12-31")

    random_dates = pd.to_datetime([start_date + pd.DateOffset(days=random.randint(0, (end_date - start_date).days)) for _ in range(len(df))]).date
    df[new_field_name] = random_dates

    return df


def overwrite_year_with_random_date(df):
    """
    Overwrite the year in the 'RandomDate' column with the year from the 'year' column in a DataFrame.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with the year in 'RandomDate' column overwritten.
    """
    df['RandomDate'] = pd.to_datetime(df['RandomDate'])
    df['RandomDate'] = df['RandomDate'].dt.strftime('%Y-%m-%d')
    df['RandomDate'] = df['year'].astype(str)+ df['RandomDate'].str[4:]

    return df

def datestdtojd(stddate):
    fmt = '%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_year * 1000 + sdtdate.tm_yday
    return jdate


# Function to convert date to Julian and append last two digits of the year
def date_to_julian_with_year(date):
    year_digits = str(date.year)[-2:]
    julian_date = date.toordinal() - datetime.datetime(year=date.year, month=1, day=1).toordinal() + 1
    julian_with_year = int(year_digits+str(julian_date) )
    return julian_with_year

def convert_to_julian(df):
    """
    Convert the dates in a DataFrame column to Julian format and insert the values into a new column.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        date_column (str): Name of the column containing dates in the format "YYYY-MM-DD".
        julian_column (str): Name of the new column to store the Julian date values.

    Returns:
        pandas.DataFrame: Updated DataFrame with the new Julian date column.
    """
    df["JulianDate"] = pd.to_datetime(df["RandomDate"])
    df['JulianDate'] = df['JulianDate'].apply(date_to_julian_with_year)

    return df





finalDF = getAndMergeDataset()
if finalDF is not None:
    finalDF = equal_frequency_binning(finalDF, 'estimate_x', 5)
    finalDF = add_random_date(finalDF,'RandomDate')
    finalDF = overwrite_year_with_random_date(finalDF)
    finalDF = convert_to_julian(finalDF)
    print(finalDF)

    df = finalDF
    # Group the data by year and calculate the sum of total value of production
    df_yearly = df.groupby('year')['estimate_x'].sum()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(df_yearly.index, df_yearly.values, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Total Value of Production')
    plt.title('Total Value of Production vs. Year')
    plt.grid(True)
    plt.show()

    # Group the data by year and calculate the sum of total value of production
    df_yearly = df.groupby('year')['estimate_y'].sum()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(df_yearly.index, df_yearly.values, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Total Acres Operated')
    plt.title('Total Acres Operated over Years')
    plt.grid(True)
    plt.show()

    file_path = "FinalDataFrame.csv"

    try:
        # Check if the file already exists
        if os.path.exists(file_path):
            # Check if the file is writable
            if os.access(file_path, os.W_OK):
                # Save the DataFrame to a CSV file
                df.to_csv(file_path, index=False)
                print("CSV file saved successfully.")
            else:
                print("File is not writable. Check file permissions.")
        else:
            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
            print("CSV file saved successfully.")
    except IOError:
        print("An error occurred while saving the CSV file.")

