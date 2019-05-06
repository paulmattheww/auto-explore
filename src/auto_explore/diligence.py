'''
dil·i·gence    /ˈdiləjəns/
noun
careful and persistent work or effort.
"few party members challenge his diligence as an MP"
synonyms:	conscientiousness, assiduousness, assiduity, industriousness, rigor,
    rigorousness, punctiliousness, meticulousness, carefulness, thoroughness,
    sedulousness, attentiveness, heedfulness, earnestness, intentness, studiousness
'''


def get_df_columns_dtypes(df):
    '''Analyzes the datatypes of a pd.DataFrame and returns the inferred
    datatype of each column as a {'col': 'dtype'}.  Type value is returned as
    a str.
    '''
    return {k: str(v) for k,v in df.dtypes.items()}

def get_numeric_columns(df):
    '''Filters dtype dict (where the values are of type str) such that only
    numeric columns are returned as a list.
    '''
    dtypes = get_df_columns_dtypes(df)
    num_dtypes = ['float64', 'int64', 'float32', 'int32', 'float', 'int',
                'long', 'complex', 'decimal']
    only_numbers = lambda x: dtypes[x] in num_dtypes
    return list(filter(only_numbers, dtypes))

def get_str_or_object_columns(df):
    '''Filters dtype dict (where the values are of type str) such that only
    str or object dtype columns are returned as a list.
    '''
    dtypes = get_df_columns_dtypes(df)
    obj_str_dtypes = ['object', 'str']
    only_numbers = lambda x: dtypes[x] in obj_str_dtypes
    return list(filter(only_numbers, dtypes))
