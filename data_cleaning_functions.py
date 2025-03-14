import pandas as pd
import numpy as np
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# formats the PCE data
def pce_tables_clean(df):
    # put rows 2 and 3 together to get a real date row and make that the set of column names instead
    new_row = df.iloc[2:4].astype(str).apply(''.join, axis=0)

    # replace column names with the concatenated row
    df.columns = new_row

    # drop the empty rows
    df = df.iloc[4:]

    # reset the index
    df = df.reset_index(drop=True)
    
    # assorted data cleaning stuff
    df = df.drop('LineLine', axis=1)
    df = df.rename(columns={'nannan': 'product'})

    # get rid of the weird aggregates that we dont need
    index_to_remove = df.index[df['product'] == 'Additional aggregates:']
    df = df.iloc[:index_to_remove[0]]
    
    # wide to long
    df_long = pd.melt(df, id_vars=['product'], var_name='date', value_name='index')

    # convert to numeric
    df_long['index'] = pd.to_numeric(df_long['index'], errors='coerce')
    
    # convert to datetime
    df_long['date'] = pd.to_datetime(df_long['date'], format='mixed')
    df_long['date'] = df_long['date'] + pd.offsets.MonthEnd(0)

    # deal with nonprofit stuff

    # remove anything with "less" in front of it
    # remove anything with "to households" in the name since these are sales from nonprofits
    df_long = df_long[~(df_long['product'].str.contains('to households'))]
    df_long = df_long[~(df_long['product'].str.contains('Less'))]

    # remove foreigner expenditures
    df_long = df_long[~(df_long['product'].str.contains('Foreign travel in the United States'))]
    df_long = df_long[~(df_long['product'].str.contains('Medical expenditures of foreigners'))]
    df_long = df_long[~(df_long['product'].str.contains('Expenditures of foreign students in the United States'))]

    # clean product names 
    # remove numbers between parentheses that follow some of the column names so that i can use the provided concordance
    df_long['product'] = df_long['product'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))

    # remove leading and trailing spaces
    df_long['product'] = df_long['product'].str.strip()

    return df_long

def requirements_clean(df, wide=False):
    # temporarily join rows 3 and 4 (convenient for merging)
    new_row = df.iloc[2:4].astype(str).apply('-- '.join, axis=0)

    # replace column names with the concatenated row
    df.columns = new_row

    # drop the empty rows
    df = df.iloc[4:]

    # Drop the sum row
    df = df.iloc[:-1]

    # reset index
    df = df.reset_index(drop=True)

    # assorted data cleaning stuff
    df = df.rename(columns={'Industry / Industry-- Code': 'NAICS_I', 'nan-- Industry Description': 'desc_I'})
    
    # wide to long
    df_long = pd.melt(df, id_vars=['NAICS_I', 'desc_I'], var_name='NAICS_desc_O', value_name='value')

    # split the NAICS code and descriptions back up
    df_long[['desc_O', 'NAICS_O']] = df_long['NAICS_desc_O'].str.split('-- ', expand=True)

    
    # reorder columns
    df_long = df_long[['NAICS_I', 'desc_I', 'NAICS_O', 'desc_O', 'value']]

    # removing rows with no naics
    # specify the value column since i dont want to get rid of nans there
    exclude_columns = ['value']
    df_long = df_long.dropna(subset=df_long.columns.difference(exclude_columns))

    df_long.loc[df_long['desc_O'] == "Drugs and druggists' sundries", 'desc_O'] = "Drugs and druggists sundries"
    df_long.loc[df_long['desc_I'] == "Drugs and druggists' sundries", 'desc_I'] = "Drugs and druggists sundries"

    df_long.loc[df_long['desc_O'] == 'Automotive repair and maintenance (including car washes)', 'desc_O'] = 'automotive repair and maintenance'
    df_long.loc[df_long['desc_I'] == 'Automotive repair and maintenance (including car washes)', 'desc_I'] = 'automotive repair and maintenance'

    df_long["desc_O"] = df_long["desc_O"].str.lower()
    df_long["desc_I"] = df_long["desc_I"].str.lower()

    if wide == False:
        return df_long
    else:
        df_wide = df_long[['desc_I', 'desc_O', 'value']].pivot_table(index='desc_I', columns='desc_O', values='value', aggfunc='mean')
        return df_wide


def concordance_PCE_clean(pce_bridge): 
    """Cleans the PCE Concordance Table to Return only the PCE products bridged to industries"""
    pce_bridge = pce_bridge.iloc[4:]
    pce_bridge = pce_bridge.iloc[:, [1,3]]
    pce_bridge.rename(columns={'Unnamed: 1': 'PCE Bridge Products' , 'Unnamed: 3': 'PCE Bridge Industries'}, inplace=True)

    pce_bridge.loc[pce_bridge['PCE Bridge Industries'] == 'Insurance Carriers, except Direct Life Insurance', 'PCE Bridge Industries'] = 'Insurance carriers, except direct life'
    pce_bridge.loc[pce_bridge['PCE Bridge Industries'] == 'Tobacco product manufacturing', 'PCE Bridge Industries'] = 'Tobacco manufacturing'

    pce_bridge["PCE Bridge Products"] = pce_bridge["PCE Bridge Products"].str.lower()
    pce_bridge["PCE Bridge Industries"] = pce_bridge["PCE Bridge Industries"].str.lower()
    return pce_bridge


def concordance_PCQ_clean(peq_bridge): 
    """Cleans the PEQ Concordance Table to Return only the investment products bridged to industries"""
    peq_bridge = peq_bridge.iloc[4:]
    peq_bridge = peq_bridge.iloc[:, [1,3]]
    peq_bridge.rename(columns={'Unnamed: 1': 'PEQ Investment Products' , 'Unnamed: 3': 'Industries'}, inplace=True)
    peq_bridge["PEQ Investment Products"] = peq_bridge["PEQ Investment Products"].str.lower()
    peq_bridge["Industries"] = peq_bridge["Industries"].str.lower()
    return peq_bridge


def find_intermediate_industries(use_table):
    """Takes BEA Use table as input, returns industires with zero PCE expenditures"""
    use_table = use_table.iloc[4:-11]
    use_table = use_table.loc[:, use_table.iloc[0].isin(['Commodity Description', 'F01000'])]
    use_table = use_table.iloc[1:]
    use_table.rename(columns={'Unnamed: 1': 'Industry' , 'Unnamed: 405': 'PCE Expenditure'}, inplace=True)
    use_table.loc[use_table['Industry'] == 'Drugs and druggists’ sundries', 'Industry'] = 'Drugs and druggists sundries'
    use_table.loc[use_table['Industry'] == 'Insurance Carriers, except Direct Life Insurance', 'Industry'] = 'Insurance carriers, except direct life'
    use_table.loc[use_table['Industry'] == 'Tobacco product manufacturing', 'Industry'] = 'Tobacco manufacturing'
    use_table.loc[use_table['Industry'] == 'Scenic and sightseeing transportation and support activities for transportatio', 'Industry'] = 'scenic and sightseeing transportation and support activities'
    use_table.loc[use_table['Industry'] == 'Community food, housing, and other relief services, including rehabilitation services', 'Industry'] = 'community food, housing, and other relief services, including vocational rehabilitation services'
    use_table["Industry"] = use_table["Industry"].str.lower()
    use_table["Industry"] = use_table["Industry"].str.strip()
    use_table = use_table.dropna(subset=['Industry'])
    use_table = use_table[use_table['PCE Expenditure'].isna()]
    return use_table 


def get_sales_from_make_matrix(make_matrix):
    """Cleans and gets sales in dollars from BEA Make Matrix ie sums the columns of make matrix"""
    make_matrix = make_matrix.iloc[3:,1:]
    make_matrix = make_matrix.drop(make_matrix.index[1])
    make_matrix.columns = make_matrix.iloc[0]
    make_matrix = make_matrix.drop(make_matrix.index[0])
    make_matrix.reset_index(drop = True, inplace=True)
    make_matrix = make_matrix[[make_matrix.columns[0], 'Total Commodity Output']]
    make_matrix.rename(columns={make_matrix.columns[0]: 'Industries', 'Total Commodity Output': "Sales"}, inplace=True)
    make_matrix.loc[make_matrix['Industries'] == 'Drugs and druggists’ sundries', 'Industries'] = 'Drugs and druggists sundries'
    make_matrix.loc[make_matrix['Industries'] == 'Insurance Carriers, except Direct Life Insurance', 'Industries'] = 'Insurance carriers, except direct life'
    make_matrix.loc[make_matrix['Industries'] == 'Tobacco product manufacturing', 'Industries'] = 'Tobacco manufacturing'
    make_matrix.loc[make_matrix['Industries'] == 'Automotive repair and maintenance (including car washes)', 'Industries'] = 'automotive repair and maintenance'
    make_matrix.loc[make_matrix['Industries'] == 'Scenic and sightseeing transportation and support activities for transportation', 'Industries'] = 'scenic and sightseeing transportation and support activities'
    make_matrix.loc[make_matrix['Industries'] == 'Community food, housing, and other relief services, including rehabilitation services', 'Industries'] = 'community food, housing, and other relief services, including vocational rehabilitation services'
    make_matrix["Industries"] = make_matrix["Industries"].str.lower()
    make_matrix['Industries'] = make_matrix['Industries'].str.strip()
    with pd.option_context("future.no_silent_downcasting", True):
        make_matrix = make_matrix.fillna(0).infer_objects(copy=False)
    
    sales = make_matrix.iloc[:-3]
    return sales


def clean_make_matrix(make_matrix):
    """Cleans BEA Make Matrix"""
    make_matrix = make_matrix.iloc[3:,1:]
    make_matrix = make_matrix.drop(make_matrix.index[1])
    make_matrix.columns = make_matrix.iloc[0]
    make_matrix = make_matrix.drop(make_matrix.index[0])
    make_matrix.reset_index(drop = True, inplace=True)
    make_matrix = make_matrix.iloc[:-3,:-12]
    make_matrix_rows = make_matrix[[make_matrix.columns[0]]]
    make_matrix_columns = pd.DataFrame(make_matrix.columns).iloc[1:]
    make_matrix_columns.columns= ["Column Industries"]
    make_matrix_rows.columns= ["Row Industries"]
    make_matrix.set_index(make_matrix.columns[0], inplace=True)

    industries_not_in_make_matrix_row = ["Secondary smelting and alloying of aluminum", "Federal electric utilities",\
                        "State and local government passenger transit", "State and local government electric utilities"]
                        
    industries_not_in_make_columns = ["scrap", "used and secondhand goods", "rest of the world adjustment", "noncomparable imports"]

    make_matrix = pd.concat([make_matrix, pd.DataFrame(0, index=make_matrix.index, columns=industries_not_in_make_columns)], axis=1)

    # Add new rows by concatenating with a DataFrame of zeros
    make_matrix = pd.concat([make_matrix, pd.DataFrame(0, index=industries_not_in_make_matrix_row, columns=make_matrix.columns)])


    make_matrix.rename(index={"Drugs and druggists’ sundries": 'Drugs and druggists sundries'}, inplace=True)
    make_matrix.rename(columns={"Drugs and druggists’ sundries": 'Drugs and druggists sundries'}, inplace=True)

    make_matrix.rename(index={'Insurance Carriers, except Direct Life Insurance': 'Insurance carriers, except direct life'}, inplace=True)
    make_matrix.rename(columns={'Insurance Carriers, except Direct Life Insurance': 'Insurance carriers, except direct life'}, inplace=True)

    make_matrix.rename(index={'Tobacco product manufacturing': 'Tobacco manufacturing'}, inplace=True)
    make_matrix.rename(columns={'Tobacco product manufacturing': 'Tobacco manufacturing'}, inplace=True)

    make_matrix.rename(index={'Scenic and sightseeing transportation and support activities for transportation': \
                              'scenic and sightseeing transportation and support activities'}, inplace=True)
    make_matrix.rename(columns={'Scenic and sightseeing transportation and support activities for transportation': \
                                'scenic and sightseeing transportation and support activities'}, inplace=True)

    make_matrix.rename(index={'Community food, housing, and other relief services, including rehabilitation services': \
                            'community food, housing, and other relief services, including vocational rehabilitation services'}, inplace=True)
    make_matrix.rename(columns={'Community food, housing, and other relief services, including rehabilitation services': \
                            'community food, housing, and other relief services, including vocational rehabilitation services'}, inplace=True)


    make_matrix.rename(index={'Automotive repair and maintenance (including car washes)': \
                            'automotive repair and maintenance'}, inplace=True)
    make_matrix.rename(columns={'Automotive repair and maintenance (including car washes)': \
                            'automotive repair and maintenance'}, inplace=True)

    make_matrix.index = make_matrix.index.str.lower()
    make_matrix.columns = make_matrix.columns.str.lower()
    make_matrix.columns = make_matrix.columns.str.strip()
    make_matrix.index = make_matrix.index.str.strip()

    with pd.option_context("future.no_silent_downcasting", True):
        make_matrix = make_matrix.fillna(0).infer_objects(copy=False)

    return make_matrix

