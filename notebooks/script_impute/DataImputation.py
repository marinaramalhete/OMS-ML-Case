import math

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from tqdm import tqdm

# Construção de script separado do notebook,
# para tornar o mesmo mais enxuto

class DataImputation():

    def __init__(self, input_df):
        self.__batch_size = 10000 # incremento lote a lote
        self.__df = input_df.sample(frac=1).copy()
        self.__pre_proc_df() # função de pré processamento
        self.__n_neighbors = int(math.sqrt(self.__batch_size)) # definição dos vizinhos

    def impute(self):
        batches = self.__get_batches()
        # método escolhido: sklearn.impute.KNNImputer
        imputer = KNNImputer(n_neighbors=self.__n_neighbors, weights='distance')
        df = pd.DataFrame(columns=list(self.__df.columns))
        for batch in tqdm(batches):
            data = imputer.fit_transform(batch.drop(['name'], axis=1))
            batch.iloc[:, 1:] = data
            df = pd.concat([df, batch])
        self.__update_df(df)

    def get_data(self):
        self.__get_readable_df()
        for col in self.__df:
            if (self.__df[col].dtype != 'object' and col != 'IMC') or col == 'name':
                self.__df[col] = self.__df[col].astype('int32')
        return self.__df

    def __pre_proc_df(self):
        self.__norm_params = {}
        self.__str_and_num = {}
        self.__categorical_cols = []

        for col in self.__df.columns:
            if col != 'name':
                if self.__df[col].dtype == 'object':
                    self.__categorical_cols.append(col)
                    self.__categorical_to_num(col)
                self.__col_minmax_norm(col)

    def __col_minmax_norm(self, col):
        maxim = self.__df[col].max()
        minim = self.__df[col].min()
        self.__norm_params[f'max_{col}'] = maxim
        self.__norm_params[f'min_{col}'] = minim
        self.__df[col] = (self.__df[col] - minim)/(maxim - minim)

    def __categorical_to_num(self, col):
        # transformação categórica para uso
        # do método sklearn.impute.KNNImputer
        original_values = list(self.__df[col].unique())
        original_values.remove(np.nan)

        for idx, original_value in enumerate(original_values):
            self.__str_and_num[f'{col}_{original_value}'] = float(idx)
            self.__str_and_num[f'{col}_{idx}.0'] = original_value
        self.__str_and_num[f'{col}_{np.nan}'] = np.nan
        self.__str_and_num[f'{col}_{np.nan}'] = np.nan
        self.__df[col] = self.__df[col].apply(lambda x: self.__str_and_num[f'{col}_{x}'])

    def __num_to_categorical(self, col):
        self.__df[col] = self.__df[col].apply(lambda x: self.__str_and_num[f'{col}_{x}'])

    def __get_readable_df(self):
        for col in self.__df.columns:
            if col != 'name':
                if col not in self.__categorical_cols:
                    self.__denorm_data(col)
                else:
                    self.__denorm_data(col, True)
                    self.__num_to_categorical(col)

    def __denorm_data(self, col):
        maxim = self.__norm_params[f'max_{col}']
        minim = self.__norm_params[f'min_{col}']

        self.__df[col] = self.__df[col]*(maxim - minim) + minim

    def __get_batches(self):
        train_df = self.__df.dropna(axis=0).copy()
        missing_df = self.__df[self.__df.isna().any(axis=1)].copy()
        batches = []
        start = 0
        end = int(start + self.__batch_size/2)

        while start < missing_df.shape[0]:
            batches.append(pd.concat([train_df[start:end].copy(), missing_df[start:end].copy()]))
            start = end
            end += int(self.__batch_size/2)
        return batches

    def __update_df(self, df):
        # atualização dos dados
        present = list(df['name'])
        to_be_checked = list(self.__df['name'])
        to_be_added = np.setdiff1d(to_be_checked, present)
        to_add = self.__df[self.__df['name'].isin(to_be_added)]
        df = pd.concat([df, to_add])
        self.__df = df