# Breast Cancer Detection

    In this case we look for a model to predict the breats cancer.
    
<p align=”center”> <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO9_MwfWPArG6IdpgFJmtw6FVhIp_PtZYyhw&usqp=CAU'/> </p>


## Background | Problema a solucionar
    
    Breast's Cancer is one of most common cancer's types, if we can detect it in a initial stage, the probability of a good final is very high, so we work to make a machine learning model that can predict it in this stage.
    

## Resultados y anlásis 

First view of a feature correlation.

![image](https://github.com/Gobuub/Repte-4-for-Nuwe/blob/main/images/Breast%20Cancer%20Train_correlation_matrix.png)

We receive a dataset with 30 features, in a first view i try to make a simple classification with only 2 features, it gives to me a good results but talking about healthcare problems an 85% of accuracy is not enough.

With a simple plot we can make a classificator.

![image](https://github.com/Gobuub/Repte-4-for-Nuwe/blob/main/images/radius_mean_vs_symmetry_mean_scatter.png)

## Solución adoptada

For this case i decide to work with several model, i make an stack models architecture, in the base of the model i use boosting and bagging models, and then i use its predictions to train de final model (logistic regression).
With this architecture i have get and 90% of f1_score with only two features radius_mean and symetry_mean, i use this features based on my background knowledge, i work on healthcare for a 15 years.
Then using all the features of the dataset and apply the same architecture i have get a 98,81% of f_score, get a 100% of f1_score is unlikely, so with this results i'm very confortable.

![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwDBEoFKZ2lxkS70Hopc89M1j8Vhik6smOlQ&usqp=CAU)

## Installation

    In the folder notebook/nuwe_resources you can find an ml_lib.py, in this file you can find all the functions that i used to make the model and plots.

## Contact info | 

 + Github: [Gobuub](https://github.com/Gobuub)

 + Web: [WEB](https://enriquerevueltagarcia.com/)

 + Linkedin: [Enrique Revuelta](https://www.linkedin.com/in/kike-rev/)
