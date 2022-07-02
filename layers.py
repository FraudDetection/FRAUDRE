import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda):
	
	"""
	Weight inter-relation aggregator
	:param num_relations: number of relations in the graph
	:param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation
	:param embed_dim: the dimension of output embedding
	:param alpha: weight paramter for each relation
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	"""

	neigh_h = neigh_feats.t()

	w = F.softmax(alpha, dim = 1)
	
	if cuda:
		aggregated = torch.zeros(size=(embed_dim, n)).cuda() #
	else:
		aggregated = torch.zeros(size=(embed_dim, n))

	for r in range(num_relations):

		aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1,n), neigh_h[:, r*n:(r+1)*n])

	return aggregated.t()


class MLP_(nn.Module):

	"""
	the ego-feature embedding module
	"""

	def __init__(self, features, input_dim, output_dim, cuda = False):

		super(MLP_, self).__init__()

		self.features = features
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.cuda = cuda
		self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)

	def forward(self, nodes):

		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(nodes))
		else:
			batch_features = self.features(torch.LongTensor(nodes))

		if self.cuda:
			self.mlp_layer.cuda()

		result = self.mlp_layer(batch_features)

		result = F.relu(result)


		return result

class InterAgg(nn.Module):

	"""
	the fraud-aware convolution module
	Inter aggregation layer
	"""

	def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):

		"""
		Initialize the inter-relation aggregator
		:param features: the input embeddings for all nodes
		:param embed_dim: the dimension need to be aggregated
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregatore used by each single-relation graph
		:param cuda: whether to use GPU
		"""

		super(InterAgg, self). __init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda

		if self.cuda:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3)).cuda()

		else:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3))

		init.xavier_uniform_(self.alpha)


	def forward(self, nodes, train_flag = True):

		"""
		nodes: a list of batch node ids
		"""
		
		if (isinstance(nodes,list)==False):
			nodes = nodes.cpu().numpy().tolist()
		
		to_neighs = []

		#adj_lists = [relation1, relation2, relation3]

		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		#to_neighs: [[set, set, set], [set, set, set], [set, set, set]]
		
		#find unique nodes and their neighbors used in current batch   #set(nodes)
		unique_nodes =  set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),set.union(*to_neighs[2], set(nodes)))

		#id mapping
		unique_nodes_new_index = {n: i for i, n in enumerate(list(unique_nodes))}
		

		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))


		#get neighbor node id list for each batch node and relation
		r1_list = [set(to_neigh) for to_neigh in to_neighs[0]] # [[set],[set],[ser]]  //   [[list],[list],[list]]
		r2_list = [set(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [set(to_neigh) for to_neigh in to_neighs[2]]

		center_nodes_new_index = [unique_nodes_new_index[int(n)] for n in nodes]################
		'''
		if self.cuda and isinstance(nodes, list):
			self_feats = self.features(torch.cuda.LongTensor(nodes))
		else:
			self_feats = self.features(index)
		'''

		#center_feats = self_feats[:, -self.embed_dim:]
		
		self_feats = batch_features[center_nodes_new_index]

		r1_feats = self.intra_agg1.forward(batch_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
		r2_feats = self.intra_agg2.forward(batch_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
		r3_feats = self.intra_agg3.forward(batch_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])

		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim = 0)

		n=len(nodes)
		
		attention_layer_outputs = weight_inter_agg(len(self.adj_lists), neigh_feats, self.embed_dim * 2, self.alpha, n, self.cuda)

		result = torch.cat((self_feats, attention_layer_outputs), dim = 1)

		return result

class IntraAgg(nn.Module):

	"""
	the fraud-aware convolution module
	Intra Aggregation Layer
	"""

	def __init__(self, cuda = False):

		super(IntraAgg, self).__init__()

		self.cuda = cuda

	def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):

		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param embedding: embedding of all nodes in a batch
		:param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]]
		:param unique_nodes_new_index
		"""

		#find unique nodes
		unique_nodes_list = list(set.union(*neighbor_lists))

		#id mapping
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		mask = Variable(torch.zeros(len(neighbor_lists), len(unique_nodes)))
		
		column_indices = [unique_nodes[n] for neighbor_list in neighbor_lists for n in neighbor_list ]
		row_indices = [i for i in range(len(neighbor_lists)) for _ in range(len(neighbor_lists[i]))]

		mask[row_indices, column_indices] = 1


		num_neigh = mask.sum(1,keepdim=True)
		#mask = torch.true_divide(mask, num_neigh)
		mask = torch.div(mask, num_neigh)

		neighbors_new_index = [unique_nodes_new_index[n] for n in unique_nodes_list ]

		embed_matrix = embedding[neighbors_new_index]
		
		embed_matrix = embed_matrix.cpu()

		_feats_1 = mask.mm(embed_matrix) 
		if self.cuda:
			_feats_1 = _feats_1.cuda()

		#difference 
		_feats_2 = self_feats - _feats_1
		return torch.cat((_feats_1, _feats_2), dim=1)
