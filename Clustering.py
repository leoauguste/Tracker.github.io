import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import pandas as pd
import warnings
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import gc
from tqdm import tqdm
import sys
import Clustering_library as Lbr
import Tools_library as Tools
#sys.exit()
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



tfile_BIB1 = r.TFile.Open("Global_index_s4038_309680_BeamGas_20MeV.HIT.root")
tree_BIB1= tfile_BIB1.Get("ntuples/SiHGTD")

# tfile_HGTD2 = r.TFile.Open("Global_s4038_BeamGas_20MeV.HIT.root")
# tree_HGTD2= tfile_HGTD2.Get("ntuples/SiHGTD")

# tfile_HGTD3 = r.TFile.Open("Global_s4038_BeamGas_20GeV.HIT.root")
# tree_HGTD3= tfile_HGTD3.Get("ntuples/SiHGTD")

tfile_top = r.TFile.Open("Global_Index_s4038_ttbar_10.HIT.root")
tree_top = tfile_top.Get("ntuples/SiHGTD")

x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1, pdg_BIB1 = Lbr.Lbr_ImportDataBIB(tree_BIB1) #variable x,y,z,R pour le BIB
# x_BIB2, y_BIB2, z_BIB2, t_BIB2, R_BIB2 = Lbr.Lbr_ImportDataBIB(tree_HGTD2) #variable x,y,z,R pour le BIB
# x_BIB3, y_BIB3, z_BIB3, t_BIB3, R_BIB3 = Lbr.Lbr_ImportDataBIB(tree_HGTD3) #variable x,y,z,R pour le BIB

x_top, y_top, z_top, t_top, R_top, pdg_top = Lbr.Lbr_ImportDataTop(tree_top)  #variable x,y,z,R pour le top

x_BIB = x_BIB1 #+ x_BIB2 + x_BIB3 
y_BIB = y_BIB1 #+ y_BIB2 + y_BIB3 
z_BIB = z_BIB1 #+ z_BIB2 + z_BIB3 
t_BIB = t_BIB1 #+ t_BIB2 + t_BIB3 
R_BIB = R_BIB1 #+ R_BIB2 + R_BIB3 
pdg_BIB = pdg_BIB1

data = {
    "pdg": [round(num, 3) for num in pdg_BIB1[1][:10]],
    "x": [round(num, 3) for num in x_BIB1[1][:10]],
    "y": [round(num, 3) for num in y_BIB1[1][:10]],
    "z": [round(num, 3) for num in z_BIB1[1][:10]],
    "t": [round(num, 3) for num in t_BIB1[1][:10]]
}

df = pd.DataFrame(data)
#print(df)

#############################


def Verifcoor(NomTree):
    phi, eta, r = [], [], []
    phi2, eta2, r2 = [], [], []

    for event in NomTree:
        phi2.append(list(event.HGTD_phi))
        eta2.append(list(event.HGTD_eta))
        r2.append(list(event.HGTD_eta))
    #pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering
    for phi1,eta1, r1 in zip(phi2,eta2, r2):
        phi4,eta4,r4=[],[], []
        
        for i in range(len(phi1)): 
            eta4.append(eta1[i])
            phi4.append(phi1[i])
            r4.append(r1[i])
        if len(phi4)>2 and len(phi4)<700: #Il eta a des evenement top glisser dans mes samples BIB, ou du moins des evenement chelouphi. je les supprimes comme ça
            eta.append(eta4)
            phi.append(phi4)
            r.append(r4)

    return phi,eta, r

phi,eta, r = Verifcoor(tree_BIB1) #variable x,y,z,R pour le BIB

x_test = []
y_test = []
z_test = []
eta_test = []
phi_test = []
r_test = []
for i in range(4):
    x_test.append(x_BIB1[1][i])
    y_test.append(y_BIB1[1][i])
    z_test.append(z_BIB1[1][i])
    eta_test.append(eta[1][i])
    phi_test.append(phi[1][i])
    r_test.append(r[1][i])

theta_test = []
R_test = []
# data = {
#     'x': [round(x, 3) for x in x_test],
#     'y': [round(y, 3) for y in y_test],
#     'z': [round(z, 3) for z in z_test],
#     'r': [round(r, 3) for r in r_test],
#     'eta': [round(e, 3) for e in eta_test],
#     'phi': [round(p, 3) for p in phi_test]
# }

# # Créer un DataFrame pandas à partir du dictionnaire
# df = pd.DataFrame(data)

# # Afficher le DataFrame
# print(df)

x_new = []
y_new = []
z_new = []
for i in range(len(x_test)):
    R_test = np.sqrt( x_test[i]**2 + y_test[i]**2 + z_test[i] )
    theta2 = 2 * np.arctan(np.exp(-eta_test[i]))
    x = r_test[i] * np.tan(theta2) * np.cos(phi_test[i])
    y = r_test[i] * np.tan(theta2) * np.sin(phi_test[i])
    z = r_test[i] * np.sinh(eta_test[i]) 
    x_new.append(x)
    y_new.append(y)
    z_new.append(z)

import pandas as pd

# Créer un dictionnaire contenant les données
data_new = {
    'x_new': [round(x, 3) for x in x_new],
    'y_new': [round(y, 3) for y in y_new],
    'z_new': [round(z, 3) for z in z_new]
}

# Créer un DataFrame pandas à partir du dictionnaire
df_new = pd.DataFrame(data_new)

# Afficher le DataFrame
print(df_new)


sys.exit()

####################################################################################
#On fais une liste pour reconnaitre le BIB et le top une fois les evenement ensemble
####################################################################################
TrueBIB = Lbr.Lbr_BIB(x_BIB)
Truetop = Lbr.Lbr_top(x_top)


####################################################################################
#On ajoute les evenement BIB et top
####################################################################################
x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix, pdg_mix = Lbr.Lbr_Melange2(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB, x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop, pdg_BIB, pdg_top)

h=0
for i in range(len(x_mix)):
    if len(pdg_mix[i]) != len(x_mix[i]):
        print(len(pdg_mix))
        print(len(x_mix))
        h+=1


####################################################################################
#Creation de l'indexage en z des layer
####################################################################################
index1 = Lbr.Lbr_IndexLayerZLDL(z_mix)

####################################################################################
#On compte le nombre de cluster True
####################################################################################
def NombreClusterTrue(t_mix, pdg_mix, index1, R_mix):
    t, pdg, index, R = [], [], [], []
    for i in range(len(x_mix)):
        t2, pdg2, index2, R2 = [], [], [], []
        for j in range(len(x_mix[i])):
            if pdg_mix[i][j] != -9999.0:
                t2.append(t_mix[i][j])
                pdg2.append(pdg_mix[i][j])
                index2.append(index1[i][j])
                R2.append(R_mix[i][j])
        t.append(t2)
        pdg.append(pdg2)
        index.append(index2)
        R.append(R2)
    return t, pdg, index,R

t, pdg, index, R= NombreClusterTrue(t_mix, pdg_mix, index1, R_mix)

# Lbr.Lbr_GraphCluster2(R_mix[-1])




# def NombreClusterTrue(t, pdg, index, R):
#     for i in range(len(t)):
#         for j in range(len(t[i])):
#             indpos,indneg,tpos,tneg,pdgpos,pdgneg, = [], [], [], [], [], []
#             if index[i][j] > 0:
#                 indpos.append(index[i][j])
#                 tpos.append(t[i][j])
#                 pdgpos.append(pdg[i][j])
#             if index[i][j] < 0:
#                 indneg.append(index[i][j])
#                 tneg.append(t[i][j])
#                 pdgneg.append(pdg[i][j])

# print(index_true)

####################################################################################
#Creation des cluster et indexage pour passer des cluster a nos liste d'event
####################################################################################
parametre = 0.05 #parametre epsilon pour cluster
MaxHit = 2 #Nombre de hit minimum pour faire un cluster

labels2, n_clusters2, n_noise2, Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2, pdg2 = Lbr.Lbr_Clustering2(parametre,MaxHit, R_mix, index1, t_mix, z_mix, BIBorTOP_mix, pdg_mix)


####################################################################################
#Clean des clusters
####################################################################################
Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_true, pdg = Lbr.Lbr_CleanCluster(Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2, pdg2)

nbhit=0
v = 0


for cluster in pdg2[-1]:
    for hit in cluster:
        if hit != -9999.0:
            nbhit += 1
            v += 1
print(nbhit)


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
D_B1, D_B2, D_B3, D_B4, D_T1, D_T2, D_T3, D_T4,  D_TruePositif, D_FalsePositif, D_TrueNegative, D_FalseNegative, \
Nombre_Cluster, Nombre_Cluster_Bon  = Lbr.Lbr_AnalyseTrace3(Coef, Index_layer, BIB_true)




















print("Nombre Cluster =", Nombre_Cluster)
print("Nombre Cluster Bon =", Nombre_Cluster_Bon)
print("Taux Cluster Bon =", Nombre_Cluster_Bon/Nombre_Cluster)

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
