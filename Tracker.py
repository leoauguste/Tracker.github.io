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
r.gStyle.SetOptStat(0)

import Tracker_library as Lbr
import Tools_library as Tools

####################################################################################
#Importation des data
####################################################################################
tfile_HGTD = r.TFile.Open("Global_s4038_BeamHalo_20MeV.HIT.root")
tree_HGTD = tfile_HGTD.Get("ntuples/SiHGTD")

tfile_top = r.TFile.Open("Global_s4038_ttbar.HIT.root")
tree_top = tfile_top.Get("ntuples/SiHGTD")

x_BIB, y_BIB, z_BIB, t_BIB, R_BIB = Lbr.Lbr_ImportDataAllMixed(tree_HGTD) #variable x,y,z,R pour le BIB
x_top, y_top, z_top, t_top, R_top = Lbr.Lbr_ImportDataAllMixed(tree_top)  #variable x,y,z,R pour le top


x=x_BIB+x_top
y=y_BIB+y_top
z=z_BIB+z_top
t=t_BIB+t_top
R=R_BIB+R_top

print(R)

####################################################################################
#Creation de l'indexage en z des layer
####################################################################################
index1_BIB = Lbr.Lbr_IndexLayerZ(z_BIB)
index1_top = Lbr.Lbr_IndexLayerZ(z_top)

index1 = Lbr.Lbr_IndexLayerZ(z)


####################################################################################
#Creation des clusters et creation des Indexage entre cluster et liste de depart
####################################################################################
labels_BIB, n_clusters_BIB, n_noise_BIB, Cluster1_BIB, Index_cluster1_BIB, Index_layer1_BIB, t_index1BIB  = Lbr.Lbr_Clustering(R_BIB,index1_BIB,t_BIB)
labels_top, n_clusters_top, n_noise_top, Cluster1_top, Index_cluster1_top, Index_layer1_top,t_index1top = Lbr.Lbr_Clustering(R_top,index1_top,t_top)

labels, n_clusters, n_noise, Cluster1, Index_cluster1, Index_layer1,t_index1 = Lbr.Lbr_Clustering(R,index1,t)


####################################################################################
#Verification des clusters
####################################################################################
Cluster_BIB, Index_cluster_BIB, Index_layer_BIB,t_indexBIB = Lbr.Lbr_CleanCluster(Cluster1_BIB,Index_cluster1_BIB,Index_layer1_BIB,t_index1BIB)
Cluster_top, Index_cluster_top, Index_layer_top,t_indexTop = Lbr.Lbr_CleanCluster(Cluster1_top,Index_cluster1_top,Index_layer1_top,t_index1top)

Cluster, Index_cluster, Index_layer,t_index = Lbr.Lbr_CleanCluster(Cluster1,Index_cluster1,Index_layer1,t_index1)


####################################################################################
#Fit Minuit, A est la liste des coef
####################################################################################
A_BIB = Lbr.Lbr_MinuitFit(Cluster_BIB, Index_cluster_BIB,z_BIB,t_BIB)
A_top = Lbr.Lbr_MinuitFit(Cluster_top, Index_cluster_top,z_top,t_top)

A = Lbr.Lbr_MinuitFit(Cluster, Index_cluster,z,t)


####################################################################################
#Graph des Cluster
####################################################################################
#Lbr.Lbr_GraphCluster(R_BIB)



####################################################################################
#Compter le nb de hit
####################################################################################
Tools.Comptage(Cluster_BIB,"BIB")
Tools.Comptage(Cluster_top,"top")

Tools.Comptage2(Cluster,"mix")




# print("nombre de hit pour BIB", len(x_BIB))
# print("Nombre de point sans cluster pour BIB:", n_noise_BIB)
# print("Nombre de cluster pour BIB: " , n_clusters_BIB)	
# print("Nombre de cluster pour BIB avec 1 hit par layer:", len(Cluster_BIB))

# print("nombre de hit pour top", len(x_top))
# print("Nombre de point sans cluster pour top:", n_noise_top)	
# print("Nombre de cluster pour top:", n_clusters_top)
# print("Nombre de cluster pour top avec 1 hit par layer: ", len(Cluster_top))


# print("nombre de hit pour le deux", len(x))
# print("Nombre de point sans cluster pour les deux:", n_noise)	
# print("Nombre de cluster pour les deux:", n_clusters)
# print("Nombre de cluster pour les deux avec 1 hit par layer: ", len(Cluster))
