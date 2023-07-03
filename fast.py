import ROOT as r
import fast_library as Lbr
import Tools_library as Tools
from pyjet import ClusterSequenceArea
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from pyjet import cluster, DTYPE_PTEPM
import fastjet
import pytest
import math
import pprint
from tqdm import tqdm
vector = pytest.importorskip("vector")
import sys
#c

####################################################################################
#Importation des data
####################################################################################
tfile_BIB1 = r.TFile.Open("Global_index_s4038_309680_BeamGas_20MeV.HIT.root")
tree_BIB1= tfile_BIB1.Get("ntuples/SiHGTD")


tfile_top = r.TFile.Open("Global_Index_s4038_ttbar_400.HIT.root")
tree_top = tfile_top.Get("ntuples/SiHGTD")


x_BIB, y_BIB, z_BIB, t_BIB, RDB_BIB, RFJ_BIB,  pdg_BIB, E_BIB = Lbr.Lbr_ImportDataBIB(tree_BIB1) #variable x,y,z,R pour le BIB
x_BIB, y_BIB, z_BIB, t_BIB, RDB_BIB, RFJ_BIB,  pdg_BIB, E_BIB = x_BIB[:100], y_BIB[:100], z_BIB[:100], t_BIB[:100], RDB_BIB[:100], RFJ_BIB[:100],  pdg_BIB[:100], E_BIB[:100]
x_top, y_top, z_top, t_top, RDB_top, RFJ_top,  pdg_top, E_top = Lbr.Lbr_ImportDataTop(tree_top) #variable x,y,z,R pour le BIB
x_top, y_top, z_top, t_top, RDB_top, RFJ_top,  pdg_top, E_top = x_top[:30], y_top[:30], z_top[:30], t_top[:30], RDB_top[:30], RFJ_top[:30],  pdg_top[:30], E_top[:30]





####################################################################################
#On fais une liste pour reconnaitre le BIB et le top une fois les evenement ensemble
####################################################################################
TrueBIB = Lbr.Lbr_BIB(x_BIB)
Truetop = Lbr.Lbr_top(x_top)
#Truetop = Lbr.Lbr_top(x_top1)
####################################################################################
#On ajoute les evenement BIB et top et on met en format ak array
####################################################################################
x, y, z, t, RFJ_tot, BIBorTOP1, pdg = Lbr.Lbr_Melange2(x_BIB, y_BIB, z_BIB, t_BIB, RFJ_BIB, x_top, y_top, z_top, t_top, RFJ_top, TrueBIB, Truetop, pdg_BIB, pdg_top)
g = 0
for x1 in x:
    g += len(x1)
print(g)
#x, y, z, t, RFJ_tot, BIBorTOP1, pdg = Lbr.Lbr_Melange2(x_BIB, y_BIB, z_BIB, t_BIB, RFJ_BIB, x_top1, y_top1, z_top1, t_top1, RFJ_top1, TrueBIB, Truetop, pdg_BIB, pdg_top1)
RFJ = Lbr.akarray(RFJ_tot)
#RFJ = Lbr.akarray(RFJ_top)
# print(RFJ_tot[1])

####################################################################################
#Creation de l'indexage en z des layer
####################################################################################
index1 = Lbr.Lbr_IndexLayerZLDL(z)
#index1 = Lbr.Lbr_IndexLayerZLDL(z_top)
####################################################################################
#Creation du cluster
####################################################################################
rayon = 1

constituent_index = Lbr.FastJetCluster(RFJ,rayon)



# for i in range(len(constituent_index)):
#     print("AAA",len(constituent_index[i]))
####################################################################################
#Creation de l'indexage de FastJet
####################################################################################
#X1, Y1, Z1, T1, PDG1, BIBorTOP1,index_layer1 = Lbr.indexage(constituent_index,x, y, z, t, pdg,BIBorTOP1,index1)
X2, Y2, Z2, T2, PDG2, BIBorTOP2,index_layer2 = Lbr.indexage(constituent_index,x, y, z, t, pdg,BIBorTOP1,index1)


print(len(T2))




####################################################################################
#Creation d'un clustering en temps + indexage
####################################################################################
parametre = 0.15
#parametre1 = np.linspace(0.01,0.5,1000)
MaxHit = 2

Para = []
Top_EFF = []
BIB_EFF = []
Hit = []
v = 0
# for parametre in parametre1:
#     print(v)
#     v +=1
#     labelsi, n_clustersi, n_noisei,  Index_cluster, Index_layer, T, Z, BIBorTOP, PDG = Lbr.Lbr_Clustering(parametre,MaxHit,index_layer2,T2,Z2,BIBorTOP2, PDG2)

labelsi, n_clustersi, n_noisei,  Index_cluster, Index_layer, T, Z, BIBorTOP, PDG = Lbr.Lbr_Clustering(parametre,MaxHit,index_layer2,T2,Z2,BIBorTOP2, PDG2)
#     h=0
#     for i in range(len(Z)):
#         if len(Z[i]) > 1:
#             h+= len(Z[i])





####################################################################################
#Fit minuit
####################################################################################

Coef= Lbr.Lbr_MinuitFitFastJet(Z, T) #avec DBSCAN
#Coef= Lbr.Lbr_MinuitFitFastJet(Z2, T2)

####################################################################################
#Comptage des resultats
####################################################################################
G_B1, G_B2, G_B3, G_B4, G_T1, G_T2, G_T3, G_T4, G_TruePositif, G_FalsePositif, G_TrueNegative, G_FalseNegative,Nombre_Cluster, Nombre_Cluster_Bon = Lbr.Lbr_AnalyseTrace3(Coef, Index_layer, BIBorTOP,PDG,T) #avec DBSCAN
#G_B1, G_B2, G_B3, G_B4, G_T1, G_T2, G_T3, G_T4, G_TruePositif, G_FalsePositif, G_TrueNegative, G_FalseNegative,Nombre_Cluster, Nombre_Cluster_Bon = Lbr.Lbr_AnalyseTrace3(Coef, index_layer2, BIBorTOP2, PDG2, T2)

# Para.append(parametre)
# Top_EFF.append(G_FalseNegative)
# BIB_EFF.append(G_TruePositif)
# Hit.append((h*100)/g)


print('parametre =', parametre)
# print("Nombre de hit total",g)
# print("Nombre de hit dans les clusters",h)
# print("% de hit",(h*100)/g)

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


sys.exit()

top_eff_array = np.array(Top_EFF, dtype=float)
BIB_eff_array = np.array(BIB_EFF, dtype=float)
para_array = np.array(Para, dtype=float)
hit_array = np.array(Hit, dtype=float)

# Create a multidimensional array with EFF, Para, and C
data = np.column_stack((BIB_eff_array, top_eff_array, para_array, hit_array))

# Save the array to a text file
np.savetxt('DBSCAN_donnees.txt', data)

sys.exit()

x_top1, y_top1, z_top1, t_top1, RDB_top1, RFJ_top1,  pdg_top1, E_top1 = [],[],[],[],[],[],[],[]
a = []
b = []
for i in range(len(t_top)):
    x_top2, y_top2, z_top2, t_top2,  RFJ_top2,  pdg_top2 = [],[],[],[],[],[]
    x_top3, y_top3, z_top3, t_top3,  RFJ_top3,  pdg_top3= [],[],[],[],[],[]
    for j in range(len(t_top[i])):
        if t_top[i][j] < 11:
            x_top2.append(x_top[i][j])
            y_top2.append(y_top[i][j])
            z_top2.append(z_top[i][j])
            t_top2.append(t_top[i][j])
            RFJ_top2.append(RFJ_top[i][j])
            pdg_top2.append(pdg_top[i][j])

        if t_top[i][j] > 11:
            x_top3.append(x_top[i][j])
            y_top3.append(y_top[i][j])
            z_top3.append(z_top[i][j])
            t_top3.append(t_top[i][j])
            RFJ_top3.append(RFJ_top[i][j])
            pdg_top3.append(pdg_top[i][j])
    x_top1.append(x_top2)
    x_top1.append(x_top3)
    y_top1.append(y_top2)
    y_top1.append(y_top3)
    z_top1.append(z_top2)
    z_top1.append(z_top3)
    t_top1.append(t_top2)
    t_top1.append(t_top3)
    RFJ_top1.append(RFJ_top2)
    RFJ_top1.append(RFJ_top3)
    pdg_top1.append(pdg_top2)
    pdg_top1.append(pdg_top3)