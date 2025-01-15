import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Function that changes all plays to going in the same direction, and also converts floats to ints to save memory
def standardize_df(df):
    
    #flip the x and y coordinates for plays, so they all go left to right
    df['x_std'] = np.where(
        df["playDirection"] == "left",
        (120 - df["x"].to_numpy()) * 100,
        df["x"].to_numpy() * 100
    ).astype(int)
    df['y_std'] = np.where(
        df["playDirection"] == "left",
        (120 - df["y"].to_numpy()) * 100,
        df["y"].to_numpy() * 100
    ).astype(int)

    #standardize and clean the 'o' column to always go left to right
    df["o_std"] = (-(df["o"] - 90)) % 360
    df["o_std"] = np.where(
        df["playDirection"] == "left",
        (180 - df["o_std"]) % 360,
        df["o_std"]
    )
    df["o_std"] = (df["o_std"] * 100).fillna(0).astype(int) #convert to int
    
    #standardize and clean the 'dir' column to always go left to right
    df["dir_std"] = (-(df["dir"] - 90)) % 360
    df["dir_std"] = np.where(
        df["playDirection"] == "left",
        (180 - df["dir_std"]) % 360, 
        df["dir_std"]
    )
    df["dir_std"] = (df["dir_std"] * 100).fillna(0).astype(int)  #convert to int

    #convert other cells to int
    df['s_std'] = (df['s']*100).astype(int)
    df['a_std'] = (df['a']*100).astype(int)
    df['dis_std'] = (df['dis']*100).astype(int)

    #downcast integer columns to save memory
    df['x_std'] = pd.to_numeric(df['x_std'], downcast='integer')
    df['y_std'] = pd.to_numeric(df['y_std'], downcast='integer')
    df['o_std'] = pd.to_numeric(df['o_std'], downcast='integer')
    df['dis_std'] = pd.to_numeric(df['dis_std'], downcast='integer')
    df['dir_std'] = pd.to_numeric(df['dir_std'], downcast='integer')
    df['s_std'] = pd.to_numeric(df['s_std'], downcast='integer')
    df['a_std'] = pd.to_numeric(df['a_std'], downcast='integer')

#creates a column of the first frame where the passed event occurred on the specific play
def find_frame_of_event(df, event_name, event_suffix):
    
    #get frameId for every event occurrence per play
    frameIds = df[df['event'] == event_name][['gameId','playId','frameId']].drop_duplicates()

    #get frameId column by merging dataframes
    merged_df = df.merge(frameIds, on=['gameId', 'playId'], suffixes=('', event_suffix))

    return merged_df

#filters the tracking data passed to the frames between 'line_set' and 'ball_snap'
def filter_to_between_line_set_ball_snap(df):

    #filter out any plays where lineset was not marked (ended up removing 0.7% of frames: (1 - len(filtered_tracking_motion) / len(tracking_motion)))
    df_new = (
        df
        .groupby(['gameId', 'playId'])
        .filter(lambda x: (x['event'] == 'line_set').any())
        .reset_index(drop=True)
    )

    #remove frames after ball_snap
    df_new = df_new[df_new['frameType'] != 'AFTER_SNAP']
    
    #add column for frameId when lineset occurred
    filtered_df = find_frame_of_event(df_new, 'line_set', '_ls')
    
    #remove frames before line_snap
    filtered_df = filtered_df[filtered_df['frameId'] >= filtered_df['frameId_ls']]

    #add column for frameId when the ball was snapped
    filtered_df = find_frame_of_event(filtered_df, 'ball_snap', '_bs')

    return filtered_df

#this function adds a column from one DataFrame (other_df) to another DataFrame (df) based on specified merge keys. 
#it allows for renaming the added column in the resulting DataFrame.
def add_column_from_other_df(df, other_df, col, new_col_name, merge_keys=['gameId', 'playId', 'nflId']):

    #shrink other dataframe
    df_small = other_df[merge_keys + [col]]
    
    #put placeholder column name that won't be in df
    df_small.rename(columns={col: 'placeholder_column_x09ds623n'}, inplace=True)
    
    #merge the dataframes
    df = df.merge(df_small, on = merge_keys)

    #rename the new column
    df.rename(columns={'placeholder_column_x09ds623n': new_col_name}, inplace=True)

    return df