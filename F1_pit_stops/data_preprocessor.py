import pandas as pd
pd.set_option('display.max_columns', None)

'''
Year – Season of the race
Race – Grand Prix name
Driver – Driver code
LapNumber – Lap index within the race
Position – Driver’s position on that lap
LapTime (s) – Lap time in seconds
Stint – Tire stint number
TyreLife – Number of laps on current tire
Normalized_TyreLife – Tire life normalized within stint
Compound_Encoded – Tire compound
LapTime_Delta – Change in lap time from previous lap
Cumulative_Degradation – Accumulated tire performance drop
Position_Change – Position gain/loss compared to previous lap
RaceProgress – Fraction of race completed (0 → 1)
PitStop – Whether the driver pitted on that lap (0/1)
'''

# скользящие статистики
def add_LapTime_RollingAggs(df):
    df = df.copy()
    window = 5

    df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
    roll = df.groupby(['Driver', 'Race'])['LapTime (s)'].rolling(window, min_periods=1)

    mean = roll.mean().reset_index(level=[0,1], drop=True) # среднее
    std = roll.std().reset_index(level=[0,1], drop=True) # стан-е отклонение
    # min_time = roll.min().reset_index(level=[0,1], drop=True) # минимум
    # max_time = roll.max().reset_index(level=[0,1], drop=True) # максимум

    df['LapTime_RollingMean'] = mean
    df['LapTime_RollingStd'] = std
    df['LapTime_vs_Mean'] = df['LapTime (s)'] - mean
    # df['LapTime_RollingMin'] = min_time
    # df['LapTime_RollingMax'] = max_time

    return df

# кол-во пит-стопов
def add_PitStop_Count(df):
    df = df.copy()
    df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
    df['PitStop_Count'] = df.groupby(['Driver', 'Race'])['PitStop'].cumsum()

    return df

# пит-стоп на пред-м круге
def add_PitStop_Prev(df):
    df = df.copy()
    df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
    df['PitStop_Prev'] = df.groupby(['Driver', 'Race'])['PitStop'].shift(1).fillna(0)

    return df

# оставшееся кол-во кругов
def add_LapsRemaning(df):
    df = df.copy()
    total_laps = df.groupby('Race')['LapNumber'].max()
    df['TotalLaps'] = df['Race'].map(total_laps)
    df['LapsRemaning'] = df['TotalLaps'] - df['LapNumber']
    df.drop('TotalLaps', axis=1, inplace=True)

    return df

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # print(train.columns)
    # print(train.describe())

    train = add_LapTime_RollingAggs(train)
    test = add_LapTime_RollingAggs(test)

    train = add_PitStop_Count(train)
    test = add_PitStop_Count(test)

    train = add_PitStop_Prev(train)
    test = add_PitStop_Prev(test)

    train = add_LapsRemaning(train)
    test = add_LapsRemaning(test)

    train.to_csv('train_aug.csv', index=False)
    test.to_csv('test_aug.csv', index=False)




