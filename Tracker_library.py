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

import Tools_library as Tools

####################################################################################
'''Information
Espacement des layers du HGTD avec les sample s4038_..._.HIT
Si pour chaque HGTD le layer le plus proche du PI est noté 0 et le plus éloigné est 
noté 3 (donc 0 1 2 3), on a comme distance entre chaque layer:
Entre 0 et 1: 10mm  Temps de parcour pour c : 0.33 nanoseconde
Entre 1 et 2: 15mm  Temps de parcour pour c : 0.50 nanoseconde
Entre 2 et 3: 10mm  Temps de parcour pour c : 0.33 nanoseconde
 '''
####################################################################################



####################################################################################
'''Importation des data tout les hit mélangé avec creation du vecteur R et en 
eliminant les hit isole et les bug  '''
####################################################################################
def Lbr_ImportDataAllMixed(NomTree):
    x,y,z,t=[],[],[],[]
    x2,y2,z2,t2=[],[],[],[]
    x3,y3,z3,t3=[],[],[],[]
    R=[]

    for event in NomTree:
	    x2.append(list(event.HGTD_x))
	    y2.append(list(event.HGTD_y))
	    z2.append(list(event.HGTD_z))
	    t2.append(list(event.HGTD_time))

    for x1,y1,z1,t1 in zip(x2,y2,z2,t2):
        if len(x1)>1:
	        x3.extend(x1)
	        y3.extend(y1)
	        z3.extend(z1)
	        t3.extend(t1)  

    for x1,y1,z1,t1 in zip(x3,y3,z3,t3):
	    if x1 not in x:
        		y.append(y1)
        		x.append(x1)
        		z.append(z1)
        		t.append(t1)

    for i in range(len(x)):
    	X=[]
    	X.append(x[i])
    	X.append(y[i])
    	X.append(t[i])
    	R.append(X)

    return x,y,z,t,R

####################################################################################
'''LDL Importation des data event par event avec creation du vecteur R et en 
eliminant les hit isole et les bug  '''
####################################################################################

def Lbr_ImportData(NomTree):
	x,y,z,t=[],[],[],[]
	x2,y2,z2,t2=[],[],[],[]
	x3,y3,z3,t3=[],[],[],[]
	R=[]

	for event in NomTree:
		x2.append(list(event.HGTD_x))
		y2.append(list(event.HGTD_y))
		z2.append(list(event.HGTD_z))
		t2.append(list(event.HGTD_time))


	for x1,y1,z1,t1 in zip(x2,y2,z2,t2):
		if x1 not in x and len(x1)>2:
				y.append(y1)
				x.append(x1)
				z.append(z1)
				t.append(t1)

	for j in range(len(x)):
		X1=[]
		
		for i in range(len(x[j])):
			X=[]
			X.append(x[j][i])
			X.append(y[j][i])
			X.append(t[j][i])
			X1.append(X)
		R.append(X1)

	return x,y,z,t,R



####################################################################################
'''LDL Mélanger event par event le BIB et le top, sachant que plusieurs
event BIB iront dans chaque event top (10 event top pour 1380 event BIB) 
donc on rassemle d'abord tout les event BIB en 10 event'''
####################################################################################
def Lbr_Melange(x_BIB, y_BIB, z_BIB, t_BIB, R_BIB,x_top, y_top, z_top, t_top, R_top, TrueBIB, Truetop):
	x_mix=[[] for i in range(len(z_top))]
	y_mix=[[] for i in range(len(z_top))]
	z_mix=[[] for i in range(len(z_top))]
	t_mix=[[] for i in range(len(z_top))]
	R_mix=[[] for i in range(len(z_top))]
	BIBorTOP_mix=[[] for i in range(len(z_top))]
	a=0
	for i in range(len(x_BIB)):
		if a==0:
			x_mix[0].extend(x_BIB[i])
			y_mix[0].extend(y_BIB[i])
			z_mix[0].extend(z_BIB[i])
			t_mix[0].extend(t_BIB[i])
			R_mix[0].extend(R_BIB[i])
			BIBorTOP_mix[0].extend(TrueBIB[i])
		if a==1:
			x_mix[1].extend(x_BIB[i])
			y_mix[1].extend(y_BIB[i])
			z_mix[1].extend(z_BIB[i])
			t_mix[1].extend(t_BIB[i])
			R_mix[1].extend(R_BIB[i])
			BIBorTOP_mix[1].extend(TrueBIB[i])
		if a==2:
			x_mix[2].extend(x_BIB[i])
			y_mix[2].extend(y_BIB[i])
			z_mix[2].extend(z_BIB[i])
			t_mix[2].extend(t_BIB[i])
			R_mix[2].extend(R_BIB[i])
			BIBorTOP_mix[2].extend(TrueBIB[i])
		if a==3:
			x_mix[3].extend(x_BIB[i])
			y_mix[3].extend(y_BIB[i])
			z_mix[3].extend(z_BIB[i])
			t_mix[3].extend(t_BIB[i])
			R_mix[3].extend(R_BIB[i])
			BIBorTOP_mix[3].extend(TrueBIB[i])
		if a==4:
			x_mix[4].extend(x_BIB[i])
			y_mix[4].extend(y_BIB[i])
			z_mix[4].extend(z_BIB[i])
			t_mix[4].extend(t_BIB[i])
			R_mix[4].extend(R_BIB[i])
			BIBorTOP_mix[4].extend(TrueBIB[i])
		if a==5:
			x_mix[5].extend(x_BIB[i])
			y_mix[5].extend(y_BIB[i])
			z_mix[5].extend(z_BIB[i])
			t_mix[5].extend(t_BIB[i])
			R_mix[5].extend(R_BIB[i])
			BIBorTOP_mix[5].extend(TrueBIB[i])
		if a==6:
			x_mix[6].extend(x_BIB[i])
			y_mix[6].extend(y_BIB[i])
			z_mix[6].extend(z_BIB[i])
			t_mix[6].extend(t_BIB[i])
			R_mix[6].extend(R_BIB[i])
			BIBorTOP_mix[6].extend(TrueBIB[i])
		if a==7:
			x_mix[7].extend(x_BIB[i])
			y_mix[7].extend(y_BIB[i])
			z_mix[7].extend(z_BIB[i])
			t_mix[7].extend(t_BIB[i])
			R_mix[7].extend(R_BIB[i])
			BIBorTOP_mix[7].extend(TrueBIB[i])
		if a==8:
			x_mix[8].extend(x_BIB[i])
			y_mix[8].extend(y_BIB[i])
			z_mix[8].extend(z_BIB[i])
			t_mix[8].extend(t_BIB[i])
			R_mix[8].extend(R_BIB[i])
			BIBorTOP_mix[8].extend(TrueBIB[i])
		if a==9:
			x_mix[9].extend(x_BIB[i])
			y_mix[9].extend(y_BIB[i])
			z_mix[9].extend(z_BIB[i])
			t_mix[9].extend(t_BIB[i])
			R_mix[9].extend(R_BIB[i])
			BIBorTOP_mix[9].extend(TrueBIB[i])
			a=-1
		a+=1
	for i in range(len(x_top)):
		x_mix[i].extend(x_top[i])
		y_mix[i].extend(y_top[i])
		z_mix[i].extend(z_top[i])
		t_mix[i].extend(t_top[i])
		R_mix[i].extend(R_top[i])
		BIBorTOP_mix[i].extend(Truetop[i])
	return x_mix, y_mix, z_mix, t_mix, R_mix, BIBorTOP_mix


####################################################################################
'''Creation de l'indexage des layers pour une liste
pour HGTD droit (z negatif) du PI -> exterieur: 3 2 1 0
pour HGTD gauche (z positif) du PI -> exterieur: -1 -2 -3 -4  '''
####################################################################################
def Lbr_IndexLayerZ(VarZ):
	index1=[]
	for i in range(len(VarZ)):
		if int(VarZ[i])== -3478:
			index2 = 3
			index1.append(index2)
		if int(VarZ[i])== -3468:
			index2 = 2
			index1.append(index2)
		if int(VarZ[i])== -3453:
			index2 = 1
			index1.append(index2)
		if int(VarZ[i])== -3443:
			index2 = 0
			index1.append(index2)
		if int(VarZ[i])== 3443:
			index2 = -1
			index1.append(index2)
		if int(VarZ[i])== 3453:
			index2 = -2
			index1.append(index2)
		if int(VarZ[i])== 3468:
			index2 = -3
			index1.append(index2)
		if int(VarZ[i])== 3478:
			index2 = -4
			index1.append(index2)
	return index1


####################################################################################
'''LDL: Creation de l'indexage des layers pour une liste de liste 
pour HGTD droit (z negatif) du PI -> exterieur: 3 2 1 0
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
	return index1




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
'''Creation du clustering avec les vecteur R  pour une liste
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre  liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################
def Lbr_Clustering(R,index1,VarT):
#Creation cluster
	R = StandardScaler().fit_transform(R)  #on standardise pour utiliser dbscan
	plt.scatter(R[:, 0], R[:, 1], R[:, 2]) #Les 3 colone x,y,t qui nous serviront pour faire les cluster

	db = DBSCAN(eps=0.005, min_samples=2).fit(R) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
	labels = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
	n_noise = list(labels).count(-1) #compte le nombre de point sans cluster

#Indexage des cluster
	Cluster1=[]
	Index_cluster1=[]	
	Index_layer1=[]
	t_index1=[]
	for i in range(n_clusters):
		M=[]
		Cluster1.append(R[labels == i])
		cluster_hits = R[labels == i]
		for j in range(len(cluster_hits)):
			hit_indices = np.where((R == cluster_hits[j]).all(axis=1))[0]
			M.append(hit_indices)
		Index_cluster1.append(M)
#Indexage des layers en z
	for i in range(len(Cluster1)):	
		T=[]
		K=[]
		for j in range(len(Cluster1[i])):
			a=Index_cluster1[i][j]
			a=int(a)
			K.append(index1[a])
			T.append(VarT[a])
		Index_layer1.append(K)
		t_index1.append(T)

	return labels, n_clusters, n_noise, Cluster1, Index_cluster1, Index_layer1 , t_index1



####################################################################################
'''LDL: Creation du clustering avec les vecteur R  pour une liste de liste
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre  liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################
def Lbr_ClusteringLDL3(R, index1, VarT, BIBorTOP_mix):
	labels=[]
	n_clusters=[]
	n_noise=[]
#Creation cluster
	for i in range(len(R)):
		R[i] = StandardScaler().fit_transform(R[i])  #on standardise pour utiliser dbscan
		plt.scatter(R[i][:, 0], R[i][:, 1], R[i][:, 2]) #Les 3 colone x,y,t qui nous serviront pour faire les cluster

		db = DBSCAN(eps=0.005, min_samples=2).fit(R[i]) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
		labelsi = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
		n_clustersi = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
		n_noisei = list(labels).count(-1) #compte le nombre de point sans cluster
		#labels.append(labelsi)
		n_clusters.append(n_clustersi)
		n_noise.append(n_noisei)
	return labels, n_clusters, n_noise


####################################################################################
'''LDL: Creation du clustering avec les vecteur R  pour une liste de liste
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre  liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################
def Lbr_ClusteringLDL(R, index1, VarT, BIBorTOP_mix):
	labels=[]
	n_clusters=[]
	n_noise=[]
#Creation cluster
	R[0] = StandardScaler().fit_transform(R[0])  #on standardise pour utiliser dbscan
	plt.scatter(R[0][:, 0], R[0][:, 1], R[0][:, 2]) #Les 3 colone x,y,t qui nous serviront pour faire les cluster

	db = DBSCAN(eps=0.05, min_samples=2).fit(R[0]) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
	labelsi = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
	n_clustersi = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
	n_noisei = list(labels).count(-1) #compte le nombre de point sans cluster
	#labels.append(labelsi)
	n_clusters.append(n_clustersi)
	n_noise.append(n_noisei)
	return labels, n_clusters, n_noise

####################################################################################
'''clean des cluster en gardant que 1 hit par layer et en enlevant les cluster non 
physique'''
####################################################################################

def Lbr_CleanCluster(Cluster1,Index_cluster1,Index_layer1,t_index1):
	print(len(Cluster1))
	Cluster=[]
	Cluster2=[]
	Index_cluster=[]
	Index_cluster2=[]	
	Index_layer=[]
	Index_layer2=[]	
	t_index2=[]
	t_index=[]
	for i in range(len(Cluster1)):
		clus=[]
		ind_clu=[]
		ind_lay=[]
		t_ind=[]
		for j in range(len(Cluster1[i])):
			if Index_layer1[i][j] not in ind_lay:
				clus.append(Cluster1[i][j])
				ind_clu.append(Index_cluster1[i][j])
				ind_lay.append(Index_layer1[i][j])
				t_ind.append(t_index1[i][j])
		if len(ind_clu)>1:
			Cluster2.append(clus)
			Index_cluster2.append(ind_clu)	
			Index_layer2.append(ind_lay)
			t_index2.append(t_ind)
	print("Cluster2=",len(Cluster2))
	t_index=[]
	Index_layer=[]
	Index_cluster=[]
	Cluster=[]

	for i in range(len(Index_layer2)):
		if ( Tools.croissante(Index_layer2[i]) or Tools.decroissante(Index_layer2[i]) ) and ( Tools.positif(Index_layer2[i]) or Tools.negatif(Index_layer2[i]) ) and ( Tools.croissante(t_index2[i]) or Tools.decroissante(t_index2[i]) ):
			Cluster.append(Cluster2[i])
			Index_cluster.append(Index_cluster2[i])	
			Index_layer.append(Index_layer2[i])
			t_index.append(t_index2[i])
	print("Tout les mask",len(Cluster))
	return Cluster, Index_cluster, Index_layer, t_index







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
			z_cl1.append(VarZ[p])
			t_cl1.append(VarT[p])
		z_cl.append(z_cl1)
		t_cl.append(t_cl1)
		def least_squares( a, b):
			return np.sum((np.array(t_cl[i]) - fonction(np.array(z_cl[i]), a, b))**2)
		

		init_a = 1.0
		init_b = 1.0
		
		# Création d'un objet Minuit avec les paramè( est_croissante(VarT[int()]) or est_decroissante(Index_layer2[i]) )tres initiaux 
		m = Minuit(least_squares, a=init_a, b=init_b)
		
		# Lancement de l'ajustement 
		m.migrad()
		A.append(m.values['a'])
	return A


####################################################################################
'''Pour créer un graphe des clusters'''
####################################################################################
def Lbr_GraphCluster(R):	
	R = StandardScaler().fit_transform(R)
	db = DBSCAN(eps=0.02, min_samples=3).fit(R)
	plt.scatter(R[:, 0], R[:, 1], R[:, 2]) 
	db = DBSCAN(eps=0.008, min_samples=3).fit(R) 
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
	plt.show()




