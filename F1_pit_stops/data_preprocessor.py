import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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



class DataPreprocessor(BaseEstimator, TransformerMixin ):
    def __init__(self, window=5):
        self.window = window
        self.total_laps = None

    def fit(self, X, y=None):
        X_sorted = X.sort_values(['Race', 'Driver', 'LapNumber'])
        self.total_laps = X_sorted.groupby('Race')['LapNumber'].max().to_dict()
        return self

    def transform(self, X):
        X = self.add_LapTime_RollingAggs(X)
        # X = self.add_TyreLife_RollingAggs(X)
        X = self.add_PitStop_Count(X)
        X = self.add_PitStop_Prev(X)
        X = self.add_LapsRemaining(X)

        return X


    # скользящие статистики
    def add_LapTime_RollingAggs(self, df):
        df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
        roll = df.groupby(['Driver', 'Race'])['LapTime (s)'].rolling(self.window, min_periods=1)

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

    def add_TyreLife_RollingAggs(self, df):
        df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
        roll = df.groupby(['Driver', 'Race'])['TyreLife'].rolling(self.window, min_periods=1)

        mean = roll.mean().reset_index(level=[0,1], drop=True)
        std = roll.std().reset_index(level=[0,1], drop=True)

        df['TyreLife_RollingMean'] = mean
        df['TyreLife_RollingStd'] = std

        return df

    # кол-во пит-стопов
    def add_PitStop_Count(self, df):
        df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
        df['PitStop_Count'] = df.groupby(['Driver', 'Race'])['PitStop'].cumsum()

        return df

    # пит-стоп на пред-м круге
    def add_PitStop_Prev(self, df):
        df.sort_values(['Race', 'Driver', 'LapNumber'], inplace=True)
        df['PitStop_Prev'] = df.groupby(['Driver', 'Race'])['PitStop'].shift(1).fillna(0)

        return df

    # оставшееся кол-во кругов
    def add_LapsRemaining(self, df):
        df['TotalLaps'] = df['Race'].map(self.total_laps)
        df['LapsRemaning'] = df['TotalLaps'] - df['LapNumber']
        df.drop('TotalLaps', axis=1, inplace=True)

        return df

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # print(train.columns)
    # print(train.describe())

    preprocessor = DataPreprocessor()
    train = preprocessor.fit_transform(train)
    test = preprocessor.transform(test)

    train.to_csv('train_aug.csv', index=False)
    test.to_csv('test_aug.csv', index=False)




