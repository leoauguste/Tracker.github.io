import ROOT as r
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
from iminuit import Minuit
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from tqdm import tqdm
import awkward as ak
import fastjet

r.gStyle.SetOptStat(0)

import Tools_library as Tools

####################################################################################
'''INFORMATION VALABLE POUR SAMPLE s4038
Espacement des layers du HGTD avec les sample s4038_..._.HIT
Si pour chaque HGTD le layer le plus proche du PI est noté 0 et le plus éloigné est 
noté 3 (donc 0 1 2 3), on a comme distance entre chaque layer:
Entre 0 et 1: 10mm  Temps de parcour pour c : 0.033 nanoseconde
Entre 1 et 2: 15mm  Temps de parcour pour c : 0.050 nanoseconde
Entre 2 et 3: 10mm  Temps de parcour pour c : 0.033 nanoseconde
En valeur absolue:
Layer 0 = 3443mm
Layer 1 = 3453mm
Layer 2 = 3468mm
Layer 3 = 3478mm
 '''
####################################################################################





####################################################################################
'''LDL Importation des data event par event avec creation du vecteur R et en 
eliminant les hit isole et les bug 
On separe BIB et top car dans les event BIB j'ai 8 event top qui se sont glisse 
et il faut que je les supprime '''
####################################################################################


########################## Pour BIB ###############################
def Lbr_ImportDataBIB(NomTree):
	x,y,z,t,pdg,E=[],[],[],[],[],[]
	x2,y2,z2,t2,pdg2,E2=[],[],[],[],[],[]
	
	R_DB=[]
	R_FJ=[]
	for event in NomTree:
		x2.append(list(event.HGTD_x))
		y2.append(list(event.HGTD_y))
		z2.append(list(event.HGTD_z))
		t2.append(list(event.HGTD_time))
		pdg2.append(list(event.HGTD_pdgId))
		E2.append(list(event.HGTD_eloss))
#pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering  (meme valeurs pour toute les variable à la dernière decimal près) 
	for x1,y1,z1,t1,pdg1,E1 in zip(x2,y2,z2,t2,pdg2,E2):
		x4,y4,z4,t4,pdg4,E4=[],[],[],[],[],[]
		for i in range(len(x1)):
			if x1[i] not in x4 and t1[i] not in t4 and t1[i] < 20 and z1[i]> 0:
				y4.append(y1[i])
				x4.append(x1[i])
				z4.append(z1[i])
				t4.append(t1[i])
				pdg4.append(pdg1[i])
				E4.append(E1[i])
		if len(x4)>1 and len(x4)<700: #Il y a des evenement top glisser dans mes samples BIB, ou du moins des evenement cheloux. je les supprimes comme ça
			y.append(y4)
			x.append(x4)
			z.append(z4)
			t.append(t4)	
			pdg.append(pdg4)	
			E.append(E4)
	for i in range(len(x)):
		X1=[]
		X1_FJ=[]
		for j in range(len(x[i])):
			if z[i][j] > 0:
				X=[]
				X_FJ=[]
				X.append(x[i][j])
				X.append(y[i][j])
				#X.append(t[i][j]) 
				X1.append(X)
				X_FJ.append((x[i][j]/t[i][j]))
				X_FJ.append((y[i][j]/t[i][j]))
				X_FJ.append((z[i][j]/t[i][j]))
				X_FJ.append(E[i][j])
				X1_FJ.append(X_FJ)
		R_DB.append(X1)
		R_FJ.append(X1_FJ)
	return x,y,z,t,R_DB,R_FJ,pdg,E
######################### Pour top #################################""
def Lbr_ImportDataTop(NomTree):
	x,y,z,t,pdg,E=[],[],[],[],[],[]
	x2,y2,z2,t2,pdg2,E2=[],[],[],[],[],[]
	
	R_DB=[]
	R_FJ=[]
	for event in NomTree:
		x2.append(list(event.HGTD_x))
		y2.append(list(event.HGTD_y))
		z2.append(list(event.HGTD_z))
		t2.append(list(event.HGTD_time))
		pdg2.append(list(event.HGTD_pdgId))
		E2.append(list(event.HGTD_eloss))
#pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering
	for x1,y1,z1,t1,pdg1,E1 in zip(x2,y2,z2,t2,pdg2,E2):
		x4,y4,z4,t4,pdg4,E4=[],[],[],[],[],[]
		for i in range(len(x1)):
			if x1[i] not in x4 and t1[i] not in t4 and z1[i] > 0 and t1[i] < 20 :
				y4.append(y1[i])
				x4.append(x1[i])
				z4.append(z1[i])
				t4.append(t1[i])
				pdg4.append(pdg1[i])
				E4.append(E1[i])
		if len(x4)>1: 
			y.append(y4)
			x.append(x4)
			z.append(z4)
			t.append(t4)	
			pdg.append(pdg4)	
			E.append(E4)

	for i in range(len(x)):
		X1=[]
		X1_FJ=[]
		for j in range(len(x[i])):
			if z[i][j] > 0:
				X=[]
				X_FJ=[]
				X.append(x[i][j])
				X.append(y[i][j])
				#X.append(t[i][j]) 
				X1.append(X)
				X_FJ.append((x[i][j]/t[i][j]))
				X_FJ.append((y[i][j]/t[i][j]))
				X_FJ.append((z[i][j]/t[i][j]))
				X_FJ.append(E[i][j])
				X1_FJ.append(X_FJ)
		R_DB.append(X1)
		R_FJ.append(X1_FJ)

	return x,y,z,t,R_DB,R_FJ,pdg,E


####################################################################################
'''Création du  ak.array '''
####################################################################################

def akarray(RFJ):
	datatop2=[]
	for i in range(len(RFJ)):
		datatop=[]
		for hit in RFJ[i]:
			dictionnaire = {"px": hit[0], "py": hit[1], "pz": hit[2], "E": hit[3], "ex": 0.0}
			datatop.append(dictionnaire)
		datatop2.append(datatop)
	datatop2 = ak.Array(datatop2)
	return datatop2

####################################################################################
'''LDL Mélanger event par event le BIB et le top, sachant que plusieurs
event BIB iront dans chaque event top (400 event top pour 1380 event BIB) 
donc on rassemle d'abord tout les event BIB en 400 event'''
####################################################################################
def Lbr_Melange(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB,x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop):
		x_mix=[[] for i in range(len(z_top))]
		y_mix=[[] for i in range(len(z_top))]
		z_mix=[[] for i in range(len(z_top))]
		t_mix=[[] for i in range(len(z_top))]
		R_mix=[[] for i in range(len(z_top))]
		BIBorTOP_mix=[[] for i in range(len(z_top))]
		for i in range(len(x_BIB)):
			index = i % len(x_mix)
			x_mix[index].extend(x_BIB[i])
			y_mix[index].extend(y_BIB[i])
			z_mix[index].extend(z_BIB[i])
			t_mix[index].extend(t_BIB[i])
			R_mix[index].extend(R_BIB[i])
			BIBorTOP_mix[index].extend(TrueBIB[i])

		for i in range(len(x_top)):
			x_mix[i].extend(x_top[i])
			y_mix[i].extend(y_top[i])
			z_mix[i].extend(z_top[i])
			t_mix[i].extend(t_top[i])
			R_mix[i].extend(R_top[i])
			BIBorTOP_mix[i].extend(Truetop[i])
		return x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix

####################################################################################
'''On mélange le BIB et le top mais en gardant chaque event séparé.'''
####################################################################################
def Lbr_Melange2(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB,x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop, pdg_BIB, pdg_top):
	x_mix= x_BIB + x_top
	y_mix= y_BIB + y_top
	z_mix= z_BIB + z_top
	t_mix= t_BIB + t_top
	R_mix= R_BIB + R_top
	pdg_mix = pdg_BIB + pdg_top
	BIBorTOP_mix= TrueBIB +  Truetop  
	

	return x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix, pdg_mix

####################################################################################
'''LDL: Creation de l'indexage des layers pour une liste de liste -4 -3 -2 -1 0 1 2 3
pour HGTD droit (z negatif) du PI -> exterieur: 0 1 2 3
pour HGTD gauche (z positif) du PI -> exterieur: -1 -2 -3 -4  '''
####################################################################################
def Lbr_IndexLayerZLDL(VarZ):
	index1i=[]
	for j in range(len(VarZ)):
		index1=[]
		for i in range(len(VarZ[j])):
			if int(VarZ[j][i])== -3478:
				index2 = 3
				index1.append(index2)
			if int(VarZ[j][i])== -3468:
				index2 = 2
				index1.append(index2)
			if int(VarZ[j][i])== -3453:
				index2 = 1
				index1.append(index2)
			if int(VarZ[j][i])== -3443:
				index2 = 0
				index1.append(index2)
			if int(VarZ[j][i])== 3443:
				index2 = -1
				index1.append(index2)
			if int(VarZ[j][i])== 3453:
				index2 = -2
				index1.append(index2)
			if int(VarZ[j][i])== 3468:
				index2 = -3
				index1.append(index2)
			if int(VarZ[j][i])== 3478:
				index2 = -4
				index1.append(index2)
		index1i.append(index1)
	return index1i



    

####################################################################################
'''LDL Regarde les events qui traverse les deux HGTD et touche au moins 3 layer
puis stock ces event dans IndexAllHGTD'''
####################################################################################
IndexAllHGTD=[]
def Lbr_TraverseAll(index1):
	for i in range(len(index1)):
		ind_lay=[]
		for j in range(1, len(index1[i])):
			if (index1[i][j-1] - index1[i][j])>0 and index1[i][j] not in ind_lay: 
				ind_lay.append(index1[i][j-1])
		if len(ind_lay)>3:
			if (7  in ind_lay or  6 in ind_lay  or 5  in ind_lay  or 4 in ind_lay) and (3 in ind_lay or 2 in ind_lay  or 1 in ind_lay  or 0 in ind_lay ):
				IndexAllHGTD.append(ind_lay)
	return IndexAllHGTD



####################################################################################
'''LDL Indexage du BIB et top pour différencier les deux
0 -> BIB
1 -> top
'''
####################################################################################

def Lbr_BIB(x_BIB):
	TrueBIB=[]
	for event in x_BIB:
		x=[]
		for hit in event:
			x.append(0)
		TrueBIB.append(x)
	return TrueBIB



def Lbr_top(x_top):
	Truetop=[]
	for event in x_top:
		x=[]
		for hit in event:
			x.append(1)
		Truetop.append(x)
	return Truetop




####################################################################################
#On compte le nombre de cluster True avec pdg
####################################################################################
def Lbr_NombreClusterTrue(t_mix, pdg_mix, index1, R_mix):
    t, pdg, index, R = [], [], [], []
    for i in range(len(t_mix)):
        t2, pdg2, index2, R2 = [], [], [], []
        for j in range(len(t_mix[i])):
            if pdg_mix[i][j] != -9999.0:
                t2.append(t_mix[i][j])
                pdg2.append(pdg_mix[i][j])
                index2.append(index1[i][j])
                R2.append(R_mix[i][j])
        t.append(t2)
        pdg.append(pdg2)
        index.append(index2)
        R.append(R2)
    return t, pdg, index, R



####################################################################################
'''Creation du clustering avec fastjet'''
####################################################################################
def FastJetCluster(RFJ,rayon):
	constituent_index = []
	for event in RFJ:
		jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, rayon)

		cluster = fastjet.ClusterSequence(event, jetdef)
		jets1 = cluster.inclusive_jets()

		constituent1 = cluster.constituents()

		constituent_index1 = cluster.constituent_index()

		constituent_index1 = constituent_index1.to_list()
		constituent_index.append(constituent_index1)
	return constituent_index


####################################################################################
'''Creation de l'indexage pour FastJet'''
####################################################################################
def indexage2(constituent_index1,VarX, VarY, VarZ, VarT, Varpdg, BiBorTop, index):
	X1,Y1,Z1,T1,PDG,BiBorTop1,index1 = [], [], [], [], [], [], []
	
	for m,constituent_index in enumerate(constituent_index1):
		x_ind, y_ind, z_ind, t_ind, pdg_ind, BIBor, ind = [], [], [], [], [], [], []
		for i in range(len(constituent_index)):
			x = [VarX[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))] #m -> eveny, i -> Cluster dans event, j -> hit dans cluster
			y = [VarY[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			z = [VarZ[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			t = [VarT[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			pdg = [Varpdg[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			BiB = [BiBorTop[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			Ind = [index[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			x_ind.append(x)
			y_ind.append(y)
			z_ind.append(z)
			t_ind.append(t)
			pdg_ind.append(pdg)
			BIBor.append(BiB)
			ind.append(Ind)
		X1.append(x_ind)
		Y1.append(y_ind)
		Z1.append(z_ind)
		T1.append(t_ind)
		PDG.append(pdg_ind)
		BiBorTop1.append(BIBor)
		index1.append(ind)
	return X1, Y1, Z1, T1, PDG, BiBorTop1, index1

####################################################################################
'''Creation de l'indexage pour FastJet'''
####################################################################################
def indexage(constituent_index1,VarX, VarY, VarZ, VarT, Varpdg, BiBorTop, index):
	X1,Y1,Z1,T1,PDG,BiBorTop1,index1 = [], [], [], [], [], [], []
	
	for m,constituent_index in enumerate(constituent_index1):
		x_ind, y_ind, z_ind, t_ind, pdg_ind, BIBor, ind = [], [], [], [], [], [], []
		for i in range(len(constituent_index)):
			x = [VarX[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))] #m -> eveny, i -> Cluster dans event, j -> hit dans cluster
			y = [VarY[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			z = [VarZ[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			t = [VarT[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			pdg = [Varpdg[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			BiB = [BiBorTop[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			Ind = [index[m][constituent_index[i][j]] for j in range(len(constituent_index[i]))]
			X1.append(x)
			Y1.append(y)
			Z1.append(z)
			T1.append(t)
			PDG.append(pdg)
			BiBorTop1.append(BiB)
			index1.append(Ind)

	return X1, Y1, Z1, T1, PDG, BiBorTop1, index1






####################################################################################
'''LDL: Creation du clustering avec les vecteur R  pour une liste de liste
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################

def Lbr_Clustering(parametre,MaxHit,index1,VarT,VarZ,BIB, Varpdg):
	labelsi, n_clustersi, n_noisei = [], [], []
	Cluster, Index_cluster, Index_layer, t_index, z_index, BIBorTop, pdg_ind = [], [], [], [], [], [], []
	for i in tqdm(range(len(VarT))):
		R1 = np.array(VarT[i]).reshape(-1, 1)

		db = DBSCAN(eps=parametre, min_samples=MaxHit) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
		db.fit(R1) 
		labels = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
		n_noise = list(labels).count(-1) #compte le nombre de point sans cluster
		labelsi.append(labels)
		n_clustersi.append(n_clusters)
		n_noisei.append(n_noise)		
	#Indexage des cluster
		Cluster1 = []



		for b in range(n_clusters):
			T,K,B,Z,P = [], [], [], [], []
			M=[]
			Cluster1.append(R1[labels == b])
			cluster_hits = R1[labels == b]
			for j in range(len(cluster_hits)):
				hit_indices = np.where((R1 == cluster_hits[j]).all(axis=1))[0]
				# print("iiiii=",i)
				# print(hit_indices)
				M.append(hit_indices)
				a=int(hit_indices)
				K.append(index1[i][a])
				T.append(VarT[i][a])
				B.append(BIB[i][a])
				Z.append(VarZ[i][a])
				P.append(Varpdg[i][a])
			pdg_ind.append(P)
			Index_layer.append(K)
			t_index.append(T)
			z_index.append(Z)
			BIBorTop.append(B)


	return labelsi, n_clustersi, n_noisei, Index_cluster, Index_layer, t_index, z_index, BIBorTop, pdg_ind







####################################################################################
'''Creation d'un fit sur les trajectoire avec minuit pour fastjet'''
####################################################################################


def Lbr_MinuitFitFastJet( VarZ, VarT):
	A=[]
	for i in tqdm(range(len(VarZ))): #cluster
		if len(VarZ[i])>1: 
			def least_squares( a, b):
				return np.sum((np.array(VarT[i]) - fonction(np.array(VarZ[i]), a, b))**2)
			init_a = 0.01
			init_b = 0.0
			
			# Création d'un objet Minuit avec les paramètres initiaux 
			m = Minuit(least_squares, a=init_a, b=init_b)
			
			# Lancement de l'ajustement 
			m.migrad()
			A.append(m.values['a'])
		else: 
			A.append(0)
	return A


####################################################################################
#On filtre les cluster par rapport au temps
####################################################################################

def FilterT(X1, Y1, Z1, T1, PDG1, BIBorTOP1,index_layer1, d):
	X, Y, Z, T, PDG, BIBorTOP, index_layer = [], [], [], [], [], [], []
	for i in range(len(T1)):
		X2, Y2, Z2, T2, PDG2, BIBorTOP2, index_layer2 = [], [], [], [], [], [], []
		for j in range(len(T1[i])):
			while len(T1[i][j]) > 0:
				# Réinitialisation desT[i][j]s pour chaque itération
				ref = T1[i][j][0]
				Xr, Yr, Zr, Tr, PDGr, BIBorTOPr, index_layerr=  [X1[i][j][0]], [Y1[i][j][0]], [Z1[i][j][0]], [T1[i][j][0]], [PDG1[i][j][0]], [BIBorTOP1[i][j][0]], [index_layer1[i][j][0]]
				Xa, Ya, Za, Ta, PDGa, BIBorTOPa, index_layera= [], [], [], [], [], [], []
				# Parcours de laT[i][j] à partir du deuxième nombre
				for k in range(len(T1[i][j])):
					if abs(T1[i][j][k] - ref) <= d:
						Xr.append(X1[i][j][k])
						Yr.append(Y1[i][j][k])
						Zr.append(Z1[i][j][k])
						Tr.append(T1[i][j][k])
						PDGr.append(PDG1[i][j][k])
						index_layerr.append(index_layer1[i][j][k])
						BIBorTOPr.append(BIBorTOP1[i][j][k])
					if abs(T1[i][j][k] - ref) > d:
						Xa.append(X1[i][j][k])
						Ya.append(Y1[i][j][k])
						Za.append(Z1[i][j][k])
						Ta.append(T1[i][j][k])
						PDGa.append(PDG1[i][j][k])
						index_layera.append(index_layer1[i][j][k])
						BIBorTOPa.append(BIBorTOP1[i][j][k])
				X2.append(Xr)
				Y2.append(Yr)
				Z2.append(Zr)
				T2.append(Tr)
				PDG2.append(PDGr)
				BIBorTOP2.append(BIBorTOPr)
				index_layer2.append(index_layerr)
				# Mise à jour de laT[i][j] avec les autres nombres pour la prochaine itération
				X1[i][j], Y1[i][j], Z1[i][j], T1[i][j], PDG1[i][j], BIBorTOP1[i][j], index_layer1[i][j]  = Xa, Ya, Za, Ta, PDGa, BIBorTOPa, index_layera
		if len(X2) > 0:
			X.append(X2)
			Y.append(Y2)
			Z.append(Z2)
			T.append(T2)
			PDG.append(PDG2)
			BIBorTOP.append(BIBorTOP2)
			index_layer.append(index_layer2)
	return X, Y, Z, T, PDG, BIBorTOP, index_layer


def FilterT2(T1, d):
	T= []
	for i in range(len(T1)):
		T2 = []
		for j in range(len(T1[i])):
			
			while len(T1[i][j]) > 0:
				# Réinitialisation desT[i][j]s pour chaque itération
				ref = T1[i][j][0]
				Tr =  []
				Ta= []
				# Parcours de laT[i][j] à partir du deuxième nombre
				for k in range(len(T1[i][j])):
					if abs(T1[i][j][k] - ref) <= d:
						Tr.append(T1[i][j][k])
					if abs(T1[i][j][k] - ref) > d:
						Ta.append(T1[i][j][k])
					# print("AAAA",Tr)
				T2.append(Tr)
				# Mise à jour de laT[i][j] avec les autres nombres pour la prochaine itération
				T1[i][j] = Ta
		if len(T2) > 0:
			T.append(T2)
	return T
####################################################################################
'''Comptage des bon fit pour evenement top seulement (pour la recherche d'un rayon
optimum).'''
####################################################################################

def comptage(VarZ,A):
	B = []
	non=0
	oui=0
	c=0
	for i in range(len(VarZ)): #event
		B1 = []
		for j in range(len(VarZ[i])): #cluster
			if len(VarZ[i][j])<2:
				B1.append("NaN")
			if len(VarZ[i][j]) > 1 and Tools.positif(VarZ[i][j]):
				B1.append("P")
			if len(VarZ[i][j]) > 1 and Tools.negatif(VarZ[i][j]):
				B1.append("N")
			if len(VarZ[i][j]) > 1 and not Tools.positif(VarZ[i][j]) and not Tools.negatif(VarZ[i][j]):
				B1.append("NaN")
		B.append(B1)

	Validation = []

	for i in range(len(A)): #event
		for j in range(len(A[i])): #cluster
			if B[i][j] == "NaN":
				Validation.append("NaN")
				c+=1
			if float(A[i][j]) > 0 and B[i][j] == "P":
				Validation.append("OUI")
				oui += 1
			if float(A[i][j]) < 0 and B[i][j] == "N":
				Validation.append("OUI")
				oui += 1
			if float(A[i][j]) > 0 and B[i][j] == "N" or float(A[i][j]) < 0 and B[i][j] == "P":
				Validation.append("NON")
				non += 1
	
	return oui, non, c





####################################################################################
'''On regarde le nombre de cluster valide'''
####################################################################################

def Lbr_BIB_Top(BIB_true3):
	a=0
	for event in BIB_true3:
		for cluster in event:
			if Tools.ZeroBIB(cluster) or Tools.OneTop(cluster):
				a+=1
	print("Le nombre de cluster bon est:",a)


####################################################################################
'''Création graph avec root'''
####################################################################################

def Lbr_Graph(VarAbs, VarOrd, AbsTitre, OrdTitre, Titre ):
	canvas = r.TCanvas("canvas", Titre, 1200, 600)
	graph = r.TGraph(len(VarAbs), np.array(VarAbs), np.array(VarOrd))
	graph.SetLineColor(r.kMagenta+2)
	graph.SetLineWidth(2)
	graph.GetXaxis().SetTitle(AbsTitre)
	graph.GetYaxis().SetTitle(OrdTitre)

	canvas.SetLeftMargin(0.15)
	canvas.SetRightMargin(0.05)
	canvas.SetTopMargin(0.05)
	canvas.SetBottomMargin(0.15)
	canvas.Update()

	graph.Draw()
	r.gPad.SetLogx()

	legend = r.TLegend(0.25, 0.65, 0.35, 0.85)
	legend.SetBorderSize(0)
	legend.SetTextSize(0.03)
	legend.SetFillColor(0)
	legend.AddEntry(graph,'Eff', "l")
	legend.Draw()

	canvas.Draw()
	canvas.Show()
	canvas.SaveAs("Efficiency.pdf")

####################################################################################
'''Creation d'un fit sur les trajectoire avec minuit'''
####################################################################################

def fonction(z1, a, b):
	return a*z1 + b

def Lbr_MinuitFit(Cluster, Index_cluster, VarZ, VarT):
	A=[]
	z_cl=[]
	t_cl=[]
	for i in range(len(Cluster)):
		z_cl1=[]
		t_cl1=[]
		for j in range(len(Cluster[i])):
			p=int(Index_cluster[i][j])
			if p >=len(VarZ):
				print("ici, p=",p)
			z_cl1.append(VarZ[p])
			t_cl1.append(VarT[p])
		z_cl.append(z_cl1)
		t_cl.append(t_cl1)
		def least_squares( a, b):
			return np.sum((np.array(t_cl[i]) - fonction(np.array(z_cl[i]), a, b))**2)
		

		init_a = 1.0
		init_b = 1.0
		
		# Création d'un objet Minuit avec les paramètres initiaux 
		m = Minuit(least_squares, a=init_a, b=init_b)
		
		# Lancement de l'ajustement 
		m.migrad()
		A.append(m.values['a'])
	return A



####################################################################################
'''On regarde les traces qui sont valide pour different coef de clustering
Utiliser les indices T et B pour y comprendre quelque chose. 
D pour HGTD Droit, G pour HGTD Gauche.
Le BIB va de Gauche a droite dans nos samples
+----------------------+-------------+
| 				 HGTD       	   |
+----------------------+-------------+
|  Sample     |  BIB  |   Top      |
+----------------------+-------------+
|  a > 0      |   B1  |   T1       |
+----------------------+-------------+
|  a < 0      |   B2  |   T2       |
+----------------------+-------------+
|  NaN        |   B3  |   T3       |
+----------------------+-------------+
|  total      |   B4  |   T4       |
+----------------------+-------------+ 
####################################################################################
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

def Lbr_AnalyseTrace3(Coef, Index_layer, BIB_true,PDG,T):

	Nombre_Cluster=0

	#HGTD gauche -> HGTD_G
	G_B1 = 0
	G_B2 = 0
	G_B3 = 0
	G_T1 = 0 
	G_T2 = 0
	G_T3 = 0

	#HGTD Droit -> HGTD_D
	D_B1 = 0
	D_B2 = 0
	D_B3 = 0
	D_T1 = 0
	D_T2 = 0
	D_T3 = 0

	for i in range(len(Coef)):
		Nombre_Cluster += 1
####################  HGTD GAUCHE  ###########################################
		if Tools.negatif(Index_layer[i]): # On se place dans HGTD gauche
			if Coef[i] > 0:  #event qu'on assimile a du top
				if Tools.OneTop(BIB_true[i]): #assumption correct, top est bien top
#On regarde si la valeur du coef a "diverge" pas, sinon on envoie dans NaN
						G_T1 += 1
				if Tools.ZeroBIB(BIB_true[i]): # assumption fausse, c'est du BIB que notre methode detecte comme du to
						G_B1 += 1
						# print("##################################")
						# print(PDG[i])
						# print(T[i])
			if Coef[i] < 0: # event qu'on assimile a du BIB
				if Tools.ZeroBIB(BIB_true[i]): #assumption correct, BIB est bien BIB
						G_B2 += 1 
				if Tools.OneTop(BIB_true[i]): # assumption fausse, c'est du top que notre methode detecte comme du BIB
						G_T2 += 1
						# print("##################################")
						# print(PDG[i])
						# print(T[i])


####################  HGTD DROIT  ############################################
		# if Tools.positif(Index_layer[i][j]): # On se place dans HGTD droit
		# 	if Coef[i][j] < 0: #event qu'on assimile a du top
		# 		if Tools.OneTop(BIB_true[i][j]): #assumption correct, top est bien top
		# 				D_T2 += 1
		# 		if Tools.ZeroBIB(BIB_true[i][j]): # assumption fausse, c'est du BIB que notre methode detecte comme du top
		# 				D_B2 += 1
		# 	if Coef[i][j] > 0: # event qu'on assimile a du BIB
		# 		if Tools.ZeroBIB(BIB_true[i][j]): #assumption correct, BIB est bien BIB
		# 				D_B1 += 1
		# 		if Tools.OneTop(BIB_true[i][j]): # assumption fausse, c'est du tp que notre methode detecte comme du BIB
		# 				D_T1 += 1



	G_B4 = G_B1 +  G_B2 + G_B3
	G_T4 = G_T1 + G_T2  + G_T3

	D_B4 = D_B1 + D_B2 + D_B3
	D_T4 = D_T1 + D_T2 + D_T3
	Nombre_Cluster_Bon = G_B2 + G_T1 + D_B1 + D_T2
	G_TruePositif = 100 * (G_B2/G_B4) 
	G_TrueNegative = 100 *((G_B1+G_B3)/G_B4)
	G_FalsePositif = 100 * ((G_T2+G_T3)/G_T4)
	G_FalseNegative =100 * (G_T1/G_T4)

	# D_TruePositif = 100 * (D_B1/D_B4) 
	# D_TrueNegative = 100 *((D_B2+D_B3)/D_B4)
	# D_FalseNegative =100 * (D_T2/D_T4)
	# D_FalsePositif = 100 * ((D_T1+D_T3)/D_T4)
	
	
	return	 G_B1, G_B2, G_B3, G_B4, G_T1, G_T2, G_T3, G_T4, G_TruePositif, G_FalsePositif, G_TrueNegative, G_FalseNegative, \
		     Nombre_Cluster, Nombre_Cluster_Bon

#		     D_B1, D_B2, D_B3, D_B4, D_T1, D_T2, D_T3, D_T4,  D_TruePositif, D_FalsePositif, D_TrueNegative, D_FalseNegative, \
####################################################################################
'''Pour créer un graphe des clusters'''
####################################################################################

from itertools import cycle

def Lbr_GraphCluster2(R, para):
    
    R = StandardScaler().fit_transform(R)

    plt.scatter(R[:, 0], R[:, 1])
    db = DBSCAN(para, min_samples=2).fit(R)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Utiliser une palette de couleurs contenant 10 couleurs distinctes
    colors = cycle(plt.cm.tab20(np.linspace(0, 1, 20)))
    total_points_in_clusters = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noir utilisé pour le bruit (noise)
            col = [0, 0, 0, 1]

	    
        class_member_mask = labels == k

        xy = R[class_member_mask & core_samples_mask]
        if k != -1:
            total_points_in_clusters += len(xy)
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = R[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
	


    plt.text(0.5, -0.070,f"Total points in clusters: {total_points_in_clusters}           Number of noise points: {n_noise}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.11,f"Total points: {total_points_in_clusters + n_noise}" , ha='center', va='center', transform=plt.gca().transAxes)
    plt.title(f"Estimated number of clusters: {n_clusters}")
    plt.show()