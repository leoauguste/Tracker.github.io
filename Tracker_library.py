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



####################################################################################
'''Importation des data avec creation du vecteur R et en eliminant les hit isole et 
les bug '''
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
'''Creation de l'indexage des layers
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
'''Creation du clustering avec les vecteur R  
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
'''clean des cluster en gardant que 1 hit par layer et en enlevant les cluster non 
physique'''
####################################################################################


		
def est_croissante(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

def est_decroissante(lst):
    return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))

def est_positif(lst):
    return all(element >= 0 for element in lst)

def est_negatif(lst):
    return all(element <= 0 for element in lst)


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


	for i in range(len(Index_layer2)):
		if ( est_croissante(Index_layer2[i]) or est_decroissante(Index_layer2[i]) ):
			Cluster.append(Cluster2[i])
			Index_cluster.append(Index_cluster2[i])	
			Index_layer.append(Index_layer2[i])
			t_index.append(t_index2[i])
	print("mask sur les layers",len(Cluster))


	t_index=[]
	Index_layer=[]
	Index_cluster=[]
	Cluster=[]
	for i in range(len(Index_layer2)):
		if  ( est_positif(Index_layer2[i]) or est_negatif(Index_layer2[i]) ) :
			Cluster.append(Cluster2[i])
			Index_cluster.append(Index_cluster2[i])	
			Index_layer.append(Index_layer2[i])
			t_index.append(t_index2[i])
	print("mask sur les positif et negatif",len(Cluster))




	t_index=[]
	Index_layer=[]
	Index_cluster=[]
	Cluster=[]
	for i in range(len(Index_layer2)):
		if  ( est_positif(Index_layer2[i]) or est_negatif(Index_layer2[i]) ) :
			Cluster.append(Cluster2[i])
			Index_cluster.append(Index_cluster2[i])	
			Index_layer.append(Index_layer2[i])
			t_index.append(t_index2[i])
	print("mask sur le temps",len(Cluster))



	t_index=[]
	Index_layer=[]
	Index_cluster=[]
	Cluster=[]

	for i in range(len(Index_layer2)):
		if ( est_croissante(Index_layer2[i]) or est_decroissante(Index_layer2[i]) ) and ( est_positif(Index_layer2[i]) or est_negatif(Index_layer2[i]) ) and ( est_croissante(t_index2[i]) or est_decroissante(t_index2[i]) ):
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




####################################################################################
'''Compter le nombre de hit qu'on garde'''
####################################################################################
def Lbr_Comptage(Cluster):
	a=0
	for event in Cluster:
		for hit in event:
			a+=1
	print("Nombre de Hit total", a)