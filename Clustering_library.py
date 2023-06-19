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
	x,y,z,t,pdg=[],[],[],[],[]
	x2,y2,z2,t2,pdg2=[],[],[],[],[]
	
	R=[]

	for event in NomTree:
		x2.append(list(event.HGTD_x))
		y2.append(list(event.HGTD_y))
		z2.append(list(event.HGTD_z))
		t2.append(list(event.HGTD_time))
		pdg2.append(list(event.HGTD_pdgId))
#pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering
	for x1,y1,z1,t1,pdg1 in zip(x2,y2,z2,t2,pdg2):
		x4,y4,z4,t4,pdg4=[],[],[],[],[]
		for i in range(len(x1)):
			if x1[i] not in x4 :
				y4.append(y1[i])
				x4.append(x1[i])
				z4.append(z1[i])
				t4.append(t1[i])
				pdg4.append(pdg1[i])
		if len(x4)>2 and len(x4)<700: #Il y a des evenement top glisser dans mes samples BIB, ou du moins des evenement cheloux. je les supprimes comme ça
			y.append(y4)
			x.append(x4)
			z.append(z4)
			t.append(t4)	
			pdg.append(pdg4)	

	for j in range(len(x)):
		X1=[]
		
		for i in range(len(x[j])):
			X=[]
			X.append(x[j][i])
			X.append(y[j][i])
			#X.append(t[j][i]) 
			X1.append(X)
		R.append(X1)

	return x,y,z,t,R,pdg
######################### Pour top #################################""
def Lbr_ImportDataTop(NomTree):
	x,y,z,t,pdg=[],[],[],[],[]
	x2,y2,z2,t2,pdg2=[],[],[],[],[]
	
	R=[]

	for event in NomTree:
		x2.append(list(event.HGTD_x))
		y2.append(list(event.HGTD_y))
		z2.append(list(event.HGTD_z))
		t2.append(list(event.HGTD_time))
		pdg2.append(list(event.HGTD_pdgId))
#pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering
	for x1,y1,z1,t1,pdg1 in zip(x2,y2,z2,t2,pdg2):
		x4,y4,z4,t4,pdg4=[],[],[],[],[]
		for i in range(len(x1)):
			if x1[i] not in x4 :
				y4.append(y1[i])
				x4.append(x1[i])
				z4.append(z1[i])
				t4.append(t1[i])
				pdg4.append(pdg1[i])
		if len(x4)>2 : 
			y.append(y4)
			x.append(x4)
			z.append(z4)
			t.append(t4)		
			pdg.append(pdg4)

	for j in range(len(x)):
		X1=[]
		
		for i in range(len(x[j])):
			X=[]
			X.append(x[j][i])
			X.append(y[j][i])
			#X.append(t[j][i])
			X1.append(X)
		R.append(X1)

	return x,y,z,t,R,pdg


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
	BIBorTOP_mix= TrueBIB + Truetop
	

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
'''LDL: Creation du clustering avec les vecteur R  pour une liste de liste
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################

def Lbr_Clustering(parametre,MaxHit, R_mix,index1,VarT,VarZ,BIB, Varpdg):
    labelsi, n_clustersi, n_noisei = [], [], []
    Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_True, pdg_ind = [], [], [], [], [], [], []
    for i in tqdm(range(len(R_mix))):
        R1=R_mix[i]
        R1 = StandardScaler().fit_transform(R1)  #on standardise pour utiliser dbscan
        x=[row[0] for row in R1]
        y=[row[1] for row in R1]
        t=[row[2] for row in R1]
        plt.scatter(x, y, t) #Les 3 colone x,y,t qui nous serviront pour faire les cluster
        db = DBSCAN(eps=parametre, min_samples=MaxHit).fit(R1) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
        labels = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
        n_noise = list(labels).count(-1) #compte le nombre de point sans cluster
        labelsi.append(labels)
        n_clustersi.append(n_clusters)
        n_noisei.append(n_noise)		
	#Indexage des cluster
        Cluster1, Index_cluster1, Index_layer1, t_index1, z_index1, BIB_true, pdg_ind1 = [], [], [], [], [], [], []
        for b in range(n_clusters):
            M=[]
            Cluster1.append(R1[labels == b])
            cluster_hits = R1[labels == b]
            for j in range(len(cluster_hits)):
                hit_indices = np.where((R1 == cluster_hits[j]).all(axis=1))[0]
                M.append(hit_indices)
            Index_cluster1.append(M)
        Cluster.append(Cluster1)
        Index_cluster.append(Index_cluster1)
    #Indexage des layers en z, t ...
        for k in range(len(Cluster1)):	
            T,K,B,Z,P = [], [], [], [], []
            for l in range(len(Cluster1[k])):
                a=Index_cluster1[k][l]
                a=int(a)
                K.append(index1[i][a])
                T.append(VarT[i][a])
                B.append(BIB[i][a])
                Z.append(VarZ[i][a])
                P.append(Varpdg[i][a])
            pdg_ind1.append(P)
            Index_layer1.append(K)
            t_index1.append(T)
            BIB_true.append(B)
            z_index1.append(Z)
        pdg_ind.append(pdg_ind1)
        Index_layer.append(Index_layer1)
        t_index.append(t_index1)
        z_index.append(z_index1)
        BIB_True.append(BIB_true)

    return labelsi, n_clustersi, n_noisei, Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_True, pdg_ind


##########################################################
#Clustering seulement en x,y


def Lbr_Clustering2(parametre,MaxHit, R_mix,index1,VarT,VarZ,BIB, Varpdg):
    labelsi, n_clustersi, n_noisei = [], [], []
    Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_True, pdg_ind = [], [], [], [], [], [], []
    for i in tqdm(range(len(R_mix))):
        R1=R_mix[i]
        R1 = StandardScaler().fit_transform(R1)  #on standardise pour utiliser dbscan
        x=[row[0] for row in R1]
        y=[row[1] for row in R1]
        plt.scatter(x, y) #Les 3 colone x,y qui nous serviront pour faire les cluster
        db = DBSCAN(eps=parametre, min_samples=MaxHit).fit(R1) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
        labels = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
        n_noise = list(labels).count(-1) #compte le nombre de point sans cluster
        labelsi.append(labels)
        n_clustersi.append(n_clusters)
        n_noisei.append(n_noise)		
	#Indexage des cluster
        Cluster1, Index_cluster1, Index_layer1, t_index1, z_index1, BIB_true, pdg_ind1 = [], [], [], [], [], [], []
        for b in range(n_clusters):
            M=[]
            Cluster1.append(R1[labels == b])
            cluster_hits = R1[labels == b]
            for j in range(len(cluster_hits)):
                hit_indices = np.where((R1 == cluster_hits[j]).all(axis=1))[0]
                M.append(hit_indices)
            Index_cluster1.append(M)
        Cluster.append(Cluster1)
        Index_cluster.append(Index_cluster1)
    #Indexage des layers en z, t ...
        for k in range(len(Cluster1)):	
            T,K,B,Z,P = [], [], [], [], []
            for l in range(len(Cluster1[k])):
                a=Index_cluster1[k][l]
                a=int(a)
                K.append(index1[i][a])
                T.append(VarT[i][a])
                B.append(BIB[i][a])
                Z.append(VarZ[i][a])
                P.append(Varpdg[i][a])
            pdg_ind1.append(P)
            Index_layer1.append(K)
            t_index1.append(T)
            BIB_true.append(B)
            z_index1.append(Z)
        pdg_ind.append(pdg_ind1)
        Index_layer.append(Index_layer1)
        t_index.append(t_index1)
        z_index.append(z_index1)
        BIB_True.append(BIB_true)

    return labelsi, n_clustersi, n_noisei, Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_True, pdg_ind

####################################################################################
'''clean des cluster en gardant que 1 hit par layer et en enlevant les cluster non 
physique
Cluster1=[[[],[],[]...],...] =Liste_Cluster -> 10 event -> cluster dans event -> hit dans cluster
'''
####################################################################################

def Valid_cluster(ind_lay, t_ind):
    return (Tools.croissante(ind_lay) or Tools.decroissante(ind_lay)) and (Tools.croissante(t_ind) or Tools.decroissante(t_ind)) and (Tools.positif(ind_lay) or Tools.negatif(ind_lay))

def Lbr_CleanCluster(Cluster1,Index_cluster1,Index_layer1,t_index1,z_index1,BIB_t, pdg1):
    Cluster3, Index_cluster3, Index_layer3, t_index3, z_index3, BIB_true3, pdg3=[], [], [], [], [], [], []
    for f in range(len(Cluster1)): 
        Cluster, Index_cluster, Index_layer, t_index, z_index, BIB_true, pdg = [], [], [], [], [], [], [] 
        Cluster2, Index_cluster2, Index_layer2, t_index2, z_index2, BIB_true2, pdg2 =[], [], [], [], [], [], [] 
        for i in range(len(Cluster1[f])): 
            clus, ind_clu, ind_lay, t_ind, z_ind, BIB, pdg_ind = [], [], [], [], [], [], [] 
            for j in range(len(Cluster1[f][i])): 
                if Index_layer1[f][i][j] not in ind_lay: #On regarde hit par hit si il y a pas plusieurs hit sur le même layer, methode a revoir
                    clus.append(Cluster1[f][i][j])
                    ind_clu.append(Index_cluster1[f][i][j])
                    ind_lay.append(Index_layer1[f][i][j])
                    t_ind.append(t_index1[f][i][j])
                    z_ind.append(z_index1[f][i][j])
                    BIB.append(BIB_t[f][i][j])
                    pdg_ind.append(pdg1[f][i][j])
            if len(ind_clu)>1: #Si notre cluster a toujours 2 hit ou plus, on garde
                Cluster2.append(clus)
                Index_cluster2.append(ind_clu)	
                Index_layer2.append(ind_lay)
                t_index2.append(t_ind)
                z_index2.append(z_ind)
                BIB_true2.append(BIB)
                pdg2.append(pdg_ind)
        for i in range(len(Index_layer2)): 
            if  ( Tools.croissante(Index_layer2[i]) or Tools.decroissante(Index_layer2[i]) ) and ( Tools.croissante(t_index2[i]) or Tools.decroissante(t_index2[i]) ) \
				and (Tools.positif(Index_layer2[i]) or Tools.negatif(Index_layer2[i]) ): #on regarde si ça a du sens physiquement (lineaire en temps et en espace + on differencie HGTD Gauche et droite)
                Cluster.append(Cluster2[i])
                Index_cluster.append(Index_cluster2[i])	
                Index_layer.append(Index_layer2[i])
                t_index.append(t_index2[i])
                z_index.append(z_index2[i])
                BIB_true.append(BIB_true2[i])
                pdg.append(pdg2[i])
        Cluster3.append(Cluster)
        Index_cluster3.append(Index_cluster)
        Index_layer3.append(Index_layer)
        t_index3.append(t_index)
        z_index3.append(z_index)
        BIB_true3.append(BIB_true)
        pdg3.append(pdg)


    return Cluster3, Index_cluster3, Index_layer3, t_index3, z_index3, BIB_true3, pdg3



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
+----------------------+-------------+ '''
####################################################################################

def Lbr_AnalyseTrace3(Coef, Index_layer, BIB_true):

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
		if len(Coef[i]) > 0:
			for j in range(len(Coef[i])):
				Nombre_Cluster += 1
		####################  HGTD GAUCHE  ###########################################
				if Tools.negatif(Index_layer[i][j]): # On se place dans HGTD gauche
					if Coef[i][j] > 0:  #event qu'on assimile a du top
						if Tools.OneTop(BIB_true[i][j]): #assumption correct, top est bien top
							if Coef[i][j] < 0.004 and Coef[i][j] > 0.003: #On regarde si la valeur du coef a "diverge" pas, sinon on envoie dans NaN
								G_T1 += 1
							else: 
								G_T3 += 1
						if Tools.ZeroBIB(BIB_true[i][j]): # assumption fausse, c'est du BIB que notre methode detecte comme du top
							if Coef[i][j] < 0.004 and Coef[i][j] > 0.003:
								G_B1 += 1
							else:
								G_B3 += 1
					if Coef[i][j] < 0: # event qu'on assimile a du BIB
						if Tools.ZeroBIB(BIB_true[i][j]): #assumption correct, BIB est bien BIB
							if Coef[i][j] > -0.004 and Coef[i][j] < -0.003: 
								G_B2 += 1 
							else:
								G_B3 +=1
						if Tools.OneTop(BIB_true[i][j]): # assumption fausse, c'est du top que notre methode detecte comme du BIB
							if Coef[i][j] > -0.004 and Coef[i][j] < -0.003: 
								G_T2 += 1
							else: 
								G_T3 += 1
		####################  HGTD DROIT  ############################################
				if Tools.positif(Index_layer[i][j]): # On se place dans HGTD droit
					if Coef[i][j] < 0: #event qu'on assimile a du top
						if Tools.OneTop(BIB_true[i][j]): #assumption correct, top est bien top
							if Coef[i][j] > -0.004 and Coef[i][j] < -0.003: 
								D_T2 += 1
							else: 
								D_T3 += 1
						if Tools.ZeroBIB(BIB_true[i][j]): # assumption fausse, c'est du BIB que notre methode detecte comme du top
							if Coef[i][j] > -0.004 and Coef[i][j] < -0.003: 
								D_B2 += 1
							else: 
								D_B3 += 1
					if Coef[i][j] > 0: # event qu'on assimile a du BIB
						if Tools.ZeroBIB(BIB_true[i][j]): #assumption correct, BIB est bien BIB
							if Coef[i][j] < 0.004 and Coef[i][j] > 0.003:
								D_B1 += 1
							else: 
								D_B3 += 1
						if Tools.OneTop(BIB_true[i][j]): # assumption fausse, c'est du tp que notre methode detecte comme du BIB
							if Coef[i][j] < 0.004 and Coef[i][j] > 0.003:
								D_T1 += 1
							else:
								D_T3 += 1



	G_B4 = G_B1 +  G_B2 + G_B3
	G_T4 = G_T1 + G_T2  + G_T3

	D_B4 = D_B1 + D_B2 + D_B3
	D_T4 = D_T1 + D_T2 + D_T3
	Nombre_Cluster_Bon = G_B2 + G_T1 + D_B1 + D_T2
	G_TruePositif = 100 * (G_B2/G_B4) 
	G_TrueNegative = 100 *((G_B1+G_B3)/G_B4)
	G_FalsePositif = 100 * ((G_T2+G_T3)/G_T4)
	G_FalseNegative =100 * (G_T1/G_T4)

	D_TruePositif = 100 * (D_B1/D_B4) 
	D_TrueNegative = 100 *((D_B2+D_B3)/D_B4)
	D_FalseNegative =100 * (D_T2/D_T4)
	D_FalsePositif = 100 * ((D_T1+D_T3)/D_T4)
	
	
	return	 G_B1, G_B2, G_B3, G_B4, G_T1, G_T2, G_T3, G_T4, G_TruePositif, G_FalsePositif, G_TrueNegative, G_FalseNegative, \
		     D_B1, D_B2, D_B3, D_B4, D_T1, D_T2, D_T3, D_T4,  D_TruePositif, D_FalsePositif, D_TrueNegative, D_FalseNegative, \
		     Nombre_Cluster, Nombre_Cluster_Bon


####################################################################################
'''Pour créer un graphe des clusters'''
####################################################################################
def Lbr_GraphCluster(R):	
	R = StandardScaler().fit_transform(R)

	plt.scatter(R[:, 0], R[:, 1]) 
	db = DBSCAN(eps=0.05, min_samples=2).fit(R) 
	labels = db.labels_  
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise = list(labels).count(-1) 
	unique_labels = set(labels)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = [0, 0, 0, 1]

		class_member_mask = labels == k

		xy = R[class_member_mask & core_samples_mask]
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

	plt.title(f"Estimated number of clusters: {n_clusters}")
	plt.text(0.5, -0.1, f"Number of noise points: {n_noise}", ha='center', va='center', transform=plt.gca().transAxes)

	plt.show()



from itertools import cycle

def Lbr_GraphCluster2(R):
    R = StandardScaler().fit_transform(R)

    plt.scatter(R[:, 0], R[:, 1])
    db = DBSCAN(eps=0.05, min_samples=2).fit(R)
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