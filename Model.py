import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

def build_model(dataset):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(dataset)]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

def columns(dataset):
    for column in dataset:
        print(column)
    print()

def main():

    df_pymes = pd.read_csv(
        r'https://raw.githubusercontent.com/juanrios1307/ResearchPymesModel/main/Pymes-Separados-Caratula.csv')
    df_EFE = pd.read_csv(
        r'https://raw.githubusercontent.com/juanrios1307/ResearchPymesModel/main/Pymes-Separados-EFE.csv')
    df_ERI = pd.read_csv(
        r'https://raw.githubusercontent.com/juanrios1307/ResearchPymesModel/main/Pymes-Separados-ERI.csv')
    df_ESF = pd.read_csv(
        r'https://raw.githubusercontent.com/juanrios1307/ResearchPymesModel/main/Pymes-Separados-ESF.csv')
    df_ORI = pd.read_csv(
        r'https://raw.githubusercontent.com/juanrios1307/ResearchPymesModel/main/Pymes-Separados-ORI.csv')


    #print(df_pymes.head())
    #print(df_EFE.head())
    #print(df_ERI.head())
    #print(df_ESF.head())
    #print(df_ORI.head())

    columns(df_pymes)
    columns(df_EFE)
    columns(df_ERI)
    columns(df_ESF)
    columns(df_ORI)

    print(df_pymes.iloc[:, 16].value_counts())

    model = build_model(df_pymes)
    print(model.summary())


if __name__ == '__main__':
    main()