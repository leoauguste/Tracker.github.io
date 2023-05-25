import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import pandas as pd
from iminuit import Minuit
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import Tracker_library as Lbr
r.gStyle.SetOptStat(0)



####################################################################################
'''Regarde si une liste de nombre est croissant ou decroissant'''
####################################################################################
def croissante(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

def decroissante(lst):
    return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))


####################################################################################
'''Regarde si une liste de nombre est positif ou negatif'''
####################################################################################
def positif(lst):
    return all(element >= 0 for element in lst)

def negatif(lst):
    return all(element <= 0 for element in lst)


####################################################################################
'''Regarde si une liste de nombre a que des zero ou que des 1'''
####################################################################################
def ZeroBIB(lst):
    return all(element == 0 for element in lst)

def OneTop(lst):
    return all(element == 1 for element in lst)


####################################################################################
'''Compter le nombre de hit dans une list'''
####################################################################################
def Comptage(Liste,nom_liste):
    a=0
    for hit in Liste:
        a+=1
    print("Nombre de Hit total pour",nom_liste,a)


####################################################################################
'''Compter le nombre de trace dans une liste de liste'''
####################################################################################
def Comptage2(Liste,nom_liste):
    a = 0
    for event in Liste:
        for hit in event:
            a += 1
    print("Nombre de Trace total:",nom_liste, a)


####################################################################################
'''Compter le nombre de hit dans une liste de liste de liste'''
####################################################################################
def Comptage3(Liste,nom_liste):
    a = 0
    for event in Liste:
        for hit in event:
            for nb in hit:
                a += 1
    print("Nombre de hit total:",nom_liste, a)