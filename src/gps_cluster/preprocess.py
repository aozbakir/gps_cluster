from scipy.stats import zscore
'''
    TBD
'''
def remove_outliers(df, std=2):
    """
    Remove, if the data is N standard deviations above or below the mean
    """
    #df[df[['Velocity (E)', 'Velocity (N)']]
    condition = (df[['Ve', 'Vn']].apply(zscore).abs() > std).any(axis=1)
    return df[~condition]

def remove_fixed(df):
    #cond = (df['Velocity (E)'] == 0) & (df['Velocity (N)'] == 0)
    cond = (df['Ve'] == 0) & (df['Vn'] == 0)
    return df[~cond]

def remove_uncertain(df):
    #condition = (df['Sigma (E)'] > .6) | (df['Sigma (N)'] > .6)
    condition = (df['Se'] >= .6) | (df['Sn'] >= .6)
    return df[~condition]
