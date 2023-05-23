import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
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
'''Compter le nombre de hit dans une list'''
####################################################################################
def Comptage(Liste,nom_liste):
    a=0
    for hit in Liste:
        a+=1
    print("Nombre de Hit total pour",nom_liste,a)


####################################################################################
'''Compter le nombre de hit dans une liste de liste'''
####################################################################################
def Comptage2(Liste,nom_liste):
    a = 0
    for event in Liste:
        for hit in event:
            a += 1
    print("Nombre de Hit total:",nom_liste, a)
