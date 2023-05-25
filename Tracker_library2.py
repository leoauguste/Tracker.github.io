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
r.gStyle.SetOptStat(0)

import Tools_library as Tools

####################################################################################
'''INFORMATION VALABLE POUR SAMPLE s4038
Espacement des layers du HGTD avec les sample s4038_..._.HIT
Si pour chaque HGTD le layer le plus proche du PI est noté 0 et le plus éloigné est 
noté 3 (donc 0 1 2 3), on a comme distance entre chaque layer:
Entre 0 et 1: 10mm  Temps de parcour pour c : 0.33 nanoseconde
Entre 1 et 2: 15mm  Temps de parcour pour c : 0.50 nanoseconde
Entre 2 et 3: 10mm  Temps de parcour pour c : 0.33 nanoseconde
En valeur absolue:
Layer 0 = 3443mm
Layer 1 = 3453mm
Layer 2 = 3468mm
Layer 3 = 3478mm
 '''
####################################################################################





####################################################################################
'''LDL Importation des data event par event avec creation du vecteur R et en 
eliminant les hit isole et les bug  '''
####################################################################################


#2745 2747
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
#pour enlever les event avec 1 hit et parce que certain hit sont en double ce qui fait bug le clustering
	for x1,y1,z1,t1 in zip(x2,y2,z2,t2):
		x4,y4,z4,t4=[],[],[],[]
		for i in range(len(x1)):
			if x1[i] not in x4 :
				y4.append(y1[i])
				x4.append(x1[i])
				z4.append(z1[i])
				t4.append(t1[i])
		if len(x4)>2:
			y.append(y4)
			x.append(x4)
			z.append(z4)
			t.append(t4)		




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
Creation de l'indexage pour passer d'un hit dans un cluster a un hit de notre  liste de depart (x_BIB,...) facilement  
-index_layer1: indexage des layer 
-Index_cluster1: indexage des hits entre le clustering et le sample
-Cluster1: Liste des cluster 
Tout les indexe on la meme forme  -> index_layer[i][j] -> Index_cluster[i][j]  -> Cluster[i][j]'''
####################################################################################

def Lbr_Clustering(R_mix,index1,VarT,BIB):
	labelsi=[]
	n_clustersi=[]
	n_noisei=[]	

	Cluster=[]
	Index_cluster=[]	
	Index_layer=[]
	t_index=[]
	BIB_True=[]
	for i in range(len(R_mix)):
		R1=R_mix[i]
		R1 = StandardScaler().fit_transform(R1)  #on standardise pour utiliser dbscan
		x=[row[0] for row in R1]
		y=[row[1] for row in R1]
		t=[row[2] for row in R1]
		plt.scatter(x, y, t) #Les 3 colone x,y,t qui nous serviront pour faire les cluster
		db = DBSCAN(eps=0.05, min_samples=2).fit(R1) #esp= distance entre chaque hit, min_samples -> nombre minimum de hit pour faire un cluster 
		labels = db.labels_  #etiquettes les hit dans les clusers -> labels[i] contient l'etiquette du i-eme point de donnees. 
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #compte le nombre de cluster
		n_noise = list(labels).count(-1) #compte le nombre de point sans cluster
		labelsi.append(labels)
		n_clustersi.append(n_clusters)
		n_noisei.append(n_noise)		

	#Indexage des cluster
		Cluster1=[]
		Index_cluster1=[]	
		Index_layer1=[]
		t_index1=[]
		BIB_true=[]
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

	#Indexage des layers en z
		for k in range(len(Cluster1)):	
			T=[]
			K=[]
			B=[]
			for l in range(len(Cluster1[k])):
				a=Index_cluster1[k][l]
				#print(a)  
				a=int(a)
				K.append(index1[i][a])
				T.append(VarT[i][a])
				B.append(BIB[i][a])
			Index_layer1.append(K)
			t_index1.append(T)
			BIB_true.append(B)
		Index_layer.append(Index_layer1)
		t_index.append(t_index1)
		BIB_True.append(BIB_true)

	return labelsi, n_clustersi, n_noisei, Cluster, Index_cluster, Index_layer, t_index, BIB_True




####################################################################################
'''clean des cluster en gardant que 1 hit par layer et en enlevant les cluster non 
physique
Cluster1=[[[],[],[]...],...] =Liste_Cluster -> 10 event -> cluster dans event -> hit dans cluster
'''
####################################################################################

def Lbr_CleanCluster(Cluster1,Index_cluster1,Index_layer1,t_index1,BIB_t):
	Cluster3=[]
	Index_cluster3=[]
	Index_layer3=[]
	t_index3=[]
	BIB_true3=[]
	for f in range(len(Cluster1)): #on boucle sur les 10 event
		Cluster=[]
		Cluster2=[]
		Index_cluster=[]
		Index_cluster2=[]	
		Index_layer=[]
		Index_layer2=[]	
		t_index2=[]
		t_index=[]
		BIB_true2=[]
		BIB_true=[]
		for i in range(len(Cluster1[f])): #on se place dans l'event f
			clus=[]
			ind_clu=[]
			ind_lay=[]
			t_ind=[]
			BIB=[]
			for j in range(len(Cluster1[f][i])): #On se place dans les cluster de l'event f
				if Index_layer1[f][i][j] not in ind_lay: #On regarde hit par hit si il y a pas plusieurs hit sur la même layer
					clus.append(Cluster1[f][i][j])
					ind_clu.append(Index_cluster1[f][i][j])
					ind_lay.append(Index_layer1[f][i][j])
					t_ind.append(t_index1[f][i][j])
					BIB.append(BIB_t[f][i][j])
			if len(ind_clu)>1:
				Cluster2.append(clus)
				Index_cluster2.append(ind_clu)	
				Index_layer2.append(ind_lay)
				t_index2.append(t_ind)
				BIB_true2.append(BIB)
		for i in range(len(Index_layer2)): 
			if ( Tools.croissante(Index_layer2[i]) or Tools.decroissante(Index_layer2[i]) ) and ( Tools.croissante(t_index2[i]) or Tools.decroissante(t_index2[i]) ): #on regarde si ça a du sens physiquement
				Cluster.append(Cluster2[i])
				Index_cluster.append(Index_cluster2[i])	
				Index_layer.append(Index_layer2[i])
				t_index.append(t_index2[i])
				BIB_true.append(BIB_true2[i])

		Cluster3.append(Cluster)
		Index_cluster3.append(Index_cluster)
		Index_layer3.append(Index_layer)
		t_index3.append(t_index)
		BIB_true3.append(BIB_true)


	return Cluster3, Index_cluster3, Index_layer3, t_index3, BIB_true3




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
		
		# Création d'un objet Minuit avec les paramètres initiaux 
		m = Minuit(least_squares, a=init_a, b=init_b)
		
		# Lancement de l'ajustement 
		m.migrad()
		A.append(m.values['a'])
	return A




####################################################################################
'''On regarde les traces qui sont valide'''
####################################################################################
def Lbr_AnalyseTrace(Coef,Index_layer,BIB_true):
	NombreTotalTrace=0
	NombreTotalTraceTrue=0
	#HGTD gauche -> HGTDG
	HGTDG_NombreTotalTrace=0
	HGTDG_NombreTotalTraceTrue=0
	HGTDG_NombreBIB=0
	HGTDG_NombreBIBTrue=0
	HGTDG_NombreTop=0
	HGTDG_NombreTopTrue=0
	#HGTD Droit -> HGTDD
	HGTDD_NombreTotalTrace=0
	HGTDD_NombreTotalTraceTrue=0
	HGTDD_NombreBIB=0
	HGTDD_NombreBIBTrue=0
	HGTDD_NombreTop=0
	HGTDD_NombreTopTrue=0


	for i in range(len(Coef)):
		for j in range(len(Coef[i])):
			NombreTotalTrace += 1
			if Coef[i][j] > 0: #top
				HGTDG_NombreTop += 1 
				if Tools.negatif(Index_layer[i][j]) and Tools.OneTop(BIB_true[i][j]):
					HGTDG_NombreTopTrue += 1
					NombreTotalTraceTrue += 1
			if Coef[i][j] < 0:
				if Tools.negatif(Index_layer[i][j]):
					HGTDG_NombreBIB += 1
					HGTDG_NombreTotalTrace += 1
					if Tools.ZeroBIB(BIB_true[i][j]):
						HGTDG_NombreBIBTrue += 1 
						NombreTotalTraceTrue += 1
						HGTDG_NombreTotalTraceTrue += 1
				if Tools.positif(Index_layer[i][j]):
					HGTDD_NombreTotalTrace += 1
					HGTDD_NombreTop += 1
					if Tools.ZeroBIB(BIB_true[i][j]):
						HGTDD_NombreTotalTraceTrue += 1
						NombreTotalTraceTrue += 1
						HGTDD_NombreBIBTrue += 1
					if Tools.OneTop(BIB_true[i][j]):
						HGTDD_NombreTotalTraceTrue += 1
						NombreTotalTraceTrue += 1
						HGTDD_NombreTopTrue += 1

	#Total
	dataTotal = {'eps=05': ['Nombre de Cluster'],
        'Total':  [NombreTotalTrace],
        'True': [NombreTotalTraceTrue],
		'Pourcentage': [(100*NombreTotalTraceTrue)/NombreTotalTrace]}
	df_total = pd.DataFrame(dataTotal)

	fig, ax = plt.subplots()
	ax.axis('off')
	ax.table(cellText=df_total.values, colLabels=df_total.columns, loc='center')

	plt.savefig('dataTotal_eps05.pdf', format='pdf', bbox_inches='tight')

	#HGTD Gauche
	dataHGTDG = {'HGTD Gauche eps=05': ['Nombre de Cluster','Nombre BIB', 'Nombre top','Differenciation'],
        'Total':  [HGTDG_NombreTotalTrace + HGTDG_NombreTop, HGTDG_NombreBIB, HGTDG_NombreTop, HGTDG_NombreTotalTraceTrue+HGTDG_NombreTopTrue ],
        'True': [HGTDG_NombreTotalTraceTrue+HGTDG_NombreTopTrue,HGTDG_NombreBIBTrue , HGTDG_NombreTopTrue, HGTDG_NombreTotalTraceTrue+HGTDG_NombreTopTrue ],
		'Pourcentage': [(100*(HGTDG_NombreTotalTraceTrue+HGTDG_NombreTopTrue))/(HGTDG_NombreTotalTrace + HGTDG_NombreTop),(100*HGTDG_NombreBIBTrue)/HGTDG_NombreBIB ,(100*HGTDG_NombreTopTrue)/HGTDG_NombreTop, 100 ]}
	df_HGTDG = pd.DataFrame(dataHGTDG)

	fig, ax = plt.subplots()
	ax.axis('off')
	ax.table(cellText=df_HGTDG.values, colLabels=df_HGTDG.columns, loc='center')

	plt.savefig('dataHGTD_Gauche_eps05.pdf', format='pdf', bbox_inches='tight')


	#HGTD_Droit
	dataHGTDD = {'HGTD Droit eps=05': ['Nombre de Cluster','Nombre BIB', 'Nombre top','Differenciation'],
        'Total':  [HGTDD_NombreTotalTrace , HGTDD_NombreBIB, HGTDD_NombreTop, HGTDD_NombreTotalTrace ],
        'True': [HGTDD_NombreTotalTraceTrue, HGTDD_NombreBIBTrue , HGTDD_NombreTopTrue,  HGTDD_NombreTopTrue ],
		'Pourcentage': [(100*(HGTDD_NombreTotalTraceTrue))/(HGTDD_NombreTotalTrace ),0 ,(100*HGTDD_NombreTopTrue)/HGTDD_NombreTop, ( HGTDD_NombreTopTrue*100)/HGTDD_NombreTotalTrace ]}
	df_HGTDD = pd.DataFrame(dataHGTDD)

	fig, ax = plt.subplots()
	ax.axis('off')
	ax.table(cellText=df_HGTDD.values, colLabels=df_HGTDD.columns, loc='center')

	plt.savefig('dataHGTD_Droit_eps05.pdf', format='pdf', bbox_inches='tight')

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




