import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score,\
                            accuracy_score, precision_score, recall_score, make_scorer



def print_scatter(x: pd.Series, y: pd.Series, clas: pd.Series)-> plt.show():
    
    '''
        Function for plot a custom scatterplot.
        
        Params:
            - x: Data for x axis -> pd.Series
            - y: Data for y axis -> pd.Series
            - clas: Data info for print the labels on plot
        Return:
            - Scatterplot
    '''
    
    
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
    
    '''
        Function for print correlation matrix of an expecific dataframe
        
        Params:
            - DataFrame -> pd.DataFrame
            - Title for the plot -> str
        Returns:
            - Lower diagonal Correlation matrix plot
    
    '''
    
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

def train_levels_models(models: list, X_train: pd.DataFrame, y_train: pd.Series,\
                        X_validation: pd.DataFrame, y_validation: pd.Series, test: pd.DataFrame)\
                        -> ((pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame), tuple): 
    '''
        Function that train a list of models make predictions of train and test and returns an array with it
        
        Parameters.
            - Models : List with models to train
            - X_train : pd.DataFrame with train data
            - y_train : pd.Series with the target of train
            - X_test : pd.DataFrame with test data
            - y_test: pd.Series with the target of test
    
        Returns:
            - Tuple of (pd.DataFrames with the predictions of train, validation and test , y_train and y validation)
              and a custom classification report with f1_score, cohen kappa score, precision score, recall score and
              confusion matrix
    '''
    
    preds_train = pd.DataFrame()
    preds_validation = pd.DataFrame()
    preds_test = pd.DataFrame()
    
    for model in tqdm(models):
        name = str(model)[:14]
        if 'cat' in name:
            name = 'catboost'
        print(f'Training {name}')
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        f1_train= f1_score(y_train, pred_train, average='macro')

        if f1_train > 0.96:
            print(f'f1_score on train of {name}: {f1_train}\n')
            preds_train[f'{name}'] = pred_train
            pred_val = model.predict(X_validation)
            f1_test = f1_score(y_validation, pred_val, average='macro')
            kappa = cohen_kappa_score(y_validation, pred_val)
            prec = precision_score(y_validation, pred_val)
            recal = recall_score(y_validation, pred_val)
            cm = confusion_matrix(y_validation, pred_val)
            print(f'f1_score on validation of {name}: {f1_test}')
            preds_validation[f'{name}'] = pred_val
            pred_test = model.predict(test)
            preds_test[f'{name}'] = pred_test
    
    return ((preds_train, y_train, preds_validation, y_validation, preds_test), (f1_test, kappa, prec, recal, cm))

def train_stack_model(stack : list, X_train: pd.DataFrame, y_train: pd.Series,\
                      X_validation: pd.DataFrame, y_validation: pd.Series, test: pd.DataFrame)\
                     ->(pd.DataFrame, dict):
    
    '''
        Function for train a stacked model with n levels, receives a list of lists with the models on it, the last list must contain only
        one model
    '''
    
    cr= ()
    preds = ((X_train, y_train, X_validation, y_validation, test), cr)
    
    if len(stack[-1])>1:
        
        raise ValueError(f'Exception: the length last level of stack model must be 1 and received {len(stack[-1])}')
    
    for s in tqdm(stack):
        preds = train_levels_models(s, preds[0][0], preds[0][1], preds[0][2], preds[0][3], preds[0][4])
    
    return (preds[0][4], {'f1_score': preds[1][0], 'kappa': preds[1][1],
                         'Precision': preds[1][2], 'Recall': preds[1][3],
                         'Confusion matrix': preds[1][4]})
