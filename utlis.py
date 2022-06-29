import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_score
from collections import defaultdict

def sparse_to_adjlist(sp_matrix):

	"""Transfer sparse matrix to adjacency list"""

	#add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	#creat adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	adj_lists = {keya:random.sample(adj_lists[keya],10) if len(adj_lists[keya])>=10 else adj_lists[keya] for i, keya in enumerate(adj_lists)}

	return adj_lists

def load_data(data):

	"""load data"""

	if data == 'yelp':

		yelp = loadmat('data/YelpChi.mat')
		homo = sparse_to_adjlist(yelp['homo'])
		relation1 = sparse_to_adjlist(yelp['net_rur'])
		relation2 = sparse_to_adjlist(yelp['net_rtr'])
		relation3 = sparse_to_adjlist(yelp['net_rsr'])
		feat_data = yelp['features'].toarray()
		labels = yelp['label'].flatten()

	elif data == 'amazon':

		amz = loadmat('data/Amazon.mat')
		homo = sparse_to_adjlist(amz['homo'])
		relation1 = sparse_to_adjlist(amz['net_upu'])
		relation2 = sparse_to_adjlist(amz['net_usu'])
		relation3 = sparse_to_adjlist(amz['net_uvu'])
		feat_data = amz['features'].toarray()
		labels = amz['label'].flatten()


	return homo, relation1, relation2, relation3, feat_data, labels


def normalize(mx):

	"""Row-normalize sparse matrix"""

	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

#!
def test_model(test_cases, labels, model):
	"""
	test the performance of model
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	"""
	gnn_prob = model.to_prob(test_cases, train_flag = False)

	auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
	precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
	a_p = average_precision_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
	recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
	f1 = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

	#print(gnn_prob.data.cpu().numpy().argmax(axis=1))

	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN precision: {precision_gnn:.4f}")
	print(f"GNN a_precision: {a_p:.4f}")
	print(f"GNN Recall: {recall_gnn:.4f}")
	print(f"GNN f1: {f1:.4f}")

	return auc_gnn, precision_gnn, a_p, recall_gnn, f1
