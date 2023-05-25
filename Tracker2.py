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
r.gStyle.SetOptStat(0)

import Tracker_library2 as Lbr
import Tools_library as Tools


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
tfile_HGTD = r.TFile.Open("Global_s4038_BeamHalo_20MeV.HIT.root")
tree_HGTD = tfile_HGTD.Get("ntuples/SiHGTD")

tfile_top = r.TFile.Open("Global_s4038_ttbar.HIT.root")
tree_top = tfile_top.Get("ntuples/SiHGTD")

x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1 = Lbr.Lbr_ImportData(tree_HGTD) #variable x,y,z,R pour le BIB
x_top, y_top, z_top, t_top, R_top = Lbr.Lbr_ImportData(tree_top)  #variable x,y,z,R pour le top


print("Nombre d'event BIB",len(x_BIB1))
print("Nombre d'event top",len(x_top))

Tools.Comptage2(x_BIB1,"BIB")
Tools.Comptage2(x_top,"top")



TrueBIB = Lbr.Lbr_BIB(x_BIB1)
Truetop = Lbr.Lbr_top(x_top)



x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix = Lbr.Lbr_Melange(x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1,x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop)

# print(x_mix[0][2745])
# print(x_mix[0][2747])
print(len(R_mix))
print(len(R_mix[0]))

####################################################################################
#Creation de l'indexage en z des layer
####################################################################################

index1 = Lbr.Lbr_IndexLayerZLDL(z_mix)


labels2, n_clusters2, n_noise2, Cluster2, Index_cluster2, Index_layer2, t_index2, BIB_true2 = Lbr.Lbr_Clustering(R_mix, index1, t_mix, BIBorTOP_mix)


Cluster, Index_cluster, Index_layer, t_index, BIB_true = Lbr.Lbr_CleanCluster(Cluster2, Index_cluster2, Index_layer2, t_index2, BIB_true2)




Tools.Comptage2(Index_layer,"nombre de trace")
Tools.Comptage3(Index_layer,"mixed")

Lbr.Lbr_BIB_Top(BIB_true)

print(Index_cluster[1][1][-1])
Coef=[]
for i in range(len(Cluster)):
    A = Lbr.Lbr_MinuitFit(Cluster[i], Index_cluster[i], z_mix[i], t_mix[i])
    Coef.append(A)






Lbr.Lbr_AnalyseTrace(Coef,Index_layer,BIB_true)



