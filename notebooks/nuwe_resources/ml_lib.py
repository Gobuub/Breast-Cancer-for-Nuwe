import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def print_scatter(x: pd.Series, y: pd.Series, clas: pd.Series)-> plt.show():
    
    
    x_name = x.name
    print(x_name)
    y_name = y.name
    print(y.name)
    plt.figure(figsize=(8,8))
    plt.title(f'{x_name} vs {y_name}')
    plt.xlabel(f'{x_name}')
    plt.ylabel(f'{y_name}')
    sns.scatterplot(x, y, hue=clas)
    
    plt.savefig(f'../images/{x_name}_vs_{y_name}_scatter.png')
    return plt.show();

def print_correlation(df : pd.DataFrame, plt_title: str) -> plt.show():    
    correlation = df.corr()

    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 15))
    
    plt.title(f'{plt_title} Correlation Matrix')
    
    cmap = sns.diverging_palette(180, 20, as_cmap=True)
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    plt.savefig(f'../images/{plt_title}_correlation_matrix.png')
    
    return plt.show()

