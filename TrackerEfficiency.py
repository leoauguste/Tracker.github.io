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

import TrackerEfficiency_library as Lbr
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

parametre = 0.002
Eff=[]
Liste_HGTD_G_NombreTotalTrace=[]
Liste_HGTD_G_NombreTotalTraceTrue=[]
Liste_EFF_g=[]
Liste_HGTD_D_NombreTotalTrace=[]
Liste_HGTD_D_NombreTotalTraceTrue=[] 
Liste_EFF_d=[] 
Liste_parametre = []
for k in range(20):
     parametre += 0.001
     tfile_HGTD = r.TFile.Open("Global_s4038_BeamHalo_20MeV.HIT.root")
     tree_HGTD = tfile_HGTD.Get("ntuples/SiHGTD")

     tfile_top = r.TFile.Open("Global_s4038_ttbar.HIT.root")
     tree_top = tfile_top.Get("ntuples/SiHGTD")

     x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1 = Lbr.Lbr_ImportData(tree_HGTD) #variable x,y,z,R pour le BIB
     x_top, y_top, z_top, t_top, R_top = Lbr.Lbr_ImportData(tree_top)  #variable x,y,z,R pour le top


     # print("Nombre d'event BIB",len(x_BIB1))
     # print("Nombre d'event top",len(x_top))

     # Tools.Comptage2(x_BIB1,"BIB")
     # Tools.Comptage2(x_top,"top")



     TrueBIB = Lbr.Lbr_BIB(x_BIB1)
     Truetop = Lbr.Lbr_top(x_top)



     x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix = Lbr.Lbr_Melange2(x_BIB1, y_BIB1, z_BIB1, t_BIB1, R_BIB1, x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop)

     # print(x_mix[0][2745])
     # print(x_mix[0][2747])
     # print(len(R_mix))
     # print(len(R_mix[0]))

     ####################################################################################
     #Creation de l'indexage en z des layer
     ####################################################################################

     index1 = Lbr.Lbr_IndexLayerZLDL(z_mix)


####################################################################################
#Creation et analyse des cluster pour différent coef de clustering
####################################################################################



     labels2, n_clusters2, n_noise2, Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2 = Lbr.Lbr_Clustering(parametre, R_mix, index1, t_mix, z_mix, BIBorTOP_mix)

     Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_true = Lbr.Lbr_CleanCluster(Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2)

     Lbr.Lbr_BIB_Top(BIB_true)
     Coef=[]

     for i in range(len(Cluster)):
          A = Lbr.Lbr_MinuitFit(Cluster[i], Index_cluster[i], z_mix[i], t_mix[i])
          Coef.append(A)

     HGTD_G_NombreTotalTrace, HGTD_G_NombreTotalTraceTrue, EFF_g, HGTD_D_NombreTotalTrace, HGTD_D_NombreTotalTraceTrue, EFF_d,  = Lbr.Lbr_AnalyseTrace2(Coef, Index_layer, BIB_true)


     print(k)
     print("Parametre=",parametre)
     Liste_parametre.append(parametre)
     Liste_HGTD_G_NombreTotalTrace.append(HGTD_G_NombreTotalTrace)
     Liste_HGTD_G_NombreTotalTraceTrue.append(HGTD_G_NombreTotalTraceTrue)
     Liste_EFF_g.append(EFF_g)
     Liste_HGTD_D_NombreTotalTrace.append(HGTD_D_NombreTotalTrace)
     Liste_HGTD_D_NombreTotalTraceTrue.append(HGTD_D_NombreTotalTraceTrue) 
     Liste_EFF_d.append(EFF_d) 

####################################################################################
#Creation d'un fichier txt pour stocker mes listes
####################################################################################


nom_fichier = "1_SansPileup.txt"

# Ouvrir le fichier en mode écriture
with open(nom_fichier, "w") as fichier:
    # Récupérer la taille de la plus grande liste
    taille_max = max(len(Liste_HGTD_G_NombreTotalTrace), len(Liste_HGTD_D_NombreTotalTrace), len(Liste_EFF_g), len(Liste_EFF_d))

    # Écrire les en-têtes des colonnes
    fichier.write("HGTD_G_NombreTotalTrace\tHGTD_G_NombreTotalTraceTrue\tEFF_g\tHGTD_D_NombreTotalTrace\tHGTD_D_NombreTotalTraceTrue\tEFF_d\n")

    # Parcourir les éléments des listes simultanément avec zip
    for i in range(taille_max):
        ligne = f"{Liste_parametre[i]}\t{Liste_HGTD_G_NombreTotalTrace[i]}\t{Liste_HGTD_G_NombreTotalTraceTrue[i]}\t{Liste_EFF_g[i]}\t{Liste_HGTD_D_NombreTotalTrace[i]}\t{Liste_HGTD_D_NombreTotalTraceTrue[i]}\t{Liste_EFF_d[i]}\n"
        fichier.write(ligne)
#Lbr.Lbr_Graph(parametre, Eff, "parametre", "Efficency", "Graphe Efficiency" )