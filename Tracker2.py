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

x_BIB, y_BIB, z_BIB, t_BIB, R_BIB = Lbr.Lbr_ImportData(tree_HGTD) #variable x,y,z,R pour le BIB
x_top, y_top, z_top, t_top, R_top = Lbr.Lbr_ImportData(tree_top)  #variable x,y,z,R pour le top


print("Nombre d'event BIB",len(x_BIB))
print("Nombre d'event top",len(x_top))

Tools.Comptage2(x_BIB,"BIB")
Tools.Comptage2(x_top,"top")

x_mix, y_mix, z_mix, t_mix, R_mix = Lbr.Lbr_Melange(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB,x_top, y_top, z_top, t_top, R_top)

