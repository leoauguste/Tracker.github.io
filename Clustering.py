import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import pandas as pd
import warnings
from iminuit import Minuit
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import gc
from tqdm import tqdm

import Clustering_library as Lbr
import Tools_library as Tools

warnings.filterwarnings("ignore", category=RuntimeWarning)
r.gStyle.SetOptStat(0)
####################################################################################
'''
LDL
Dans ce script on prendra tout les event du BIB et du top qu'on mixera event par 
event (avec plusieurs event BIB pour un event top), ensuite on fera le clustering 
puis l'analyse (sans faire de trie au préalable)
'''
####################################################################################

####################################################################################
#Importation des data
####################################################################################

#parametre = np.linspace(0.0005, 1, 20)



tfile_HGTD1 = r.TFile.Open("Global_s4038_BeamHalo_20MeV.HIT.root")
tree_HGTD1= tfile_HGTD1.Get("ntuples/SiHGTD")

tfile_HGTD2 = r.TFile.Open("Global_s4038_BeamGas_20MeV.HIT.root")
tree_HGTD2= tfile_HGTD2.Get("ntuples/SiHGTD")

tfile_HGTD3 = r.TFile.Open("Global_s4038_BeamGas_20GeV.HIT.root")
tree_HGTD3= tfile_HGTD3.Get("ntuples/SiHGTD")

tfile_top = r.TFile.Open("Global_s4038_ttbar_400.HIT.root")
tree_top = tfile_top.Get("ntuples/SiHGTD")

x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1 = Lbr.Lbr_ImportDataBIB(tree_HGTD1) #variable x,y,z,R pour le BIB
x_BIB2, y_BIB2, z_BIB2, t_BIB2, R_BIB2 = Lbr.Lbr_ImportDataBIB(tree_HGTD2) #variable x,y,z,R pour le BIB
x_BIB3, y_BIB3, z_BIB3, t_BIB3, R_BIB3 = Lbr.Lbr_ImportDataBIB(tree_HGTD3) #variable x,y,z,R pour le BIB

x_top, y_top, z_top, t_top, R_top = Lbr.Lbr_ImportDataTop(tree_top)  #variable x,y,z,R pour le top

x_BIB = x_BIB1 + x_BIB2 + x_BIB3 
y_BIB = y_BIB1 + y_BIB2 + y_BIB3 
z_BIB = z_BIB1 + z_BIB2 + z_BIB3 
t_BIB = t_BIB1 + t_BIB2 + t_BIB3 
R_BIB = R_BIB1 + R_BIB2 + R_BIB3 

####################################################################################
#On fais une liste pour reconnaitre le BIB et le top une fois les evenement ensemble
####################################################################################
TrueBIB = Lbr.Lbr_BIB(x_BIB)
Truetop = Lbr.Lbr_top(x_top)


####################################################################################
#On ajoute les evenement BIB et top
####################################################################################
x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix = Lbr.Lbr_Melange2(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB, x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop)


####################################################################################
#Creation de l'indexage en z des layer
####################################################################################
index1 = Lbr.Lbr_IndexLayerZLDL(z_mix)


####################################################################################
#Creation des cluster et indexage pour passer des cluster a nos liste d'event
####################################################################################
parametre = 0.008 #parametre epsilon pour cluster
MaxHit = 2 #Nombre de hit minimum pour faire un cluster
labels2, n_clusters2, n_noise2, Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2 = Lbr.Lbr_Clustering(parametre, R_mix, index1, t_mix, z_mix, BIBorTOP_mix)

####################################################################################
#Clean des clusters
####################################################################################
Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_true = Lbr.Lbr_CleanCluster(Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2)


####################################################################################
#Fit avec minuit et on place les coef "a" dans la liste Coef
####################################################################################
Coef=[]
for i in tqdm(range(len(Cluster))):
    A = Lbr.Lbr_MinuitFit(Cluster[i], Index_cluster[i], z_mix[i], t_mix[i])
    Coef.append(A)

####################################################################################
#Comptage des resultats
####################################################################################
G_B1, G_B2, G_B3, G_B4, G_T1, G_T2, G_T3, G_T4, G_TruePositif, G_FalsePositif, G_TrueNegative, G_FalseNegative, \
D_B1, D_B2, D_B3, D_B4, D_T1, D_T2, D_T3, D_T4,  D_TruePositif, D_FalsePositif, D_TrueNegative, D_FalseNegative  = Lbr.Lbr_AnalyseTrace3(Coef, Index_layer, BIB_true)

print("##################  HGTD Gauche  ##############################")
print("B1=",G_B1, "B2=",G_B2, "B3=",G_B3, "B4=",G_B4 )
print("T1=",G_T1, "T2=",G_T2, "T3=",G_T3, "T4=",G_T4 )
print("True Positif=",G_TruePositif)
print("False Positif=",G_FalsePositif)
print("True Negative=",G_TrueNegative)
print("False Negative=",G_FalseNegative)

print("##################  HGTD Droit  ##############################")
print("B1=",D_B1, "B2=",D_B2, "B3=",D_B3, "B4=",D_B4 )
print("T1=",D_T1, "T2=",D_T2, "T3=",D_T3, "T4=",D_T4 )
print("True Positif=",D_TruePositif)
print("False Positif=",D_FalsePositif)
print("True Negative=",D_TrueNegative)
print("False Negative=",D_FalseNegative)

####################################################################################
#Comment ça marche
####################################################################################
'''clustering
Utiliser les indices T et B pour y comprendre quelque chose. 
D pour HGTD Droit, G pour HGTD Gauche.
Le BIB va de Gauche a droite dans nos samples
+----------------------+-------------+
| 			 HGTD Gauche       	   |
+----------------------+-------------+
|  Sample     |  BIB  |   Top      |
+----------------------+-------------+
|  a > 0      |   B1  |   T1       |
+----------------------+-------------+
|  a < 0      |   B2  |   T2       |
+----------------------+-------------+
|  nan        |   B3  |   T3       |
+----------------------+-------------+
|  total      |   B4  |   T4       |
+----------------------+-------------+ 
######################################
#####################
+-------------------+
| True positive %   |
| 100 * (B2 / B4)   |
+-------------------+
| False positive %  |
| 100 * (T2 / T4)   |
+-------------------+
| True negative %   |
| 100 * (B1 / B4)   |
+-------------------+
| False negative %  |
| 100 * (T1 / T4)   |
+-------------------+

Pour faire le HGTD Droit il faut inverser les 1 et 2 (B1 -> B2, B2 -> B1 etc...)
'''
####################################################################################
