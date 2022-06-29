import torch
import torch.nn as nn
from torch.nn import init
import math

class MODEL(nn.Module):

	def __init__(self, K, num_classes, embed_dim, agg, prior, cuda ):
		super(MODEL, self).__init__()

		"""
		Initialize the model
		:param K: the number of CONVOLUTION layers of the model
		:param num_classes: number of classes (2 in our paper)
		:param embed_dim: the output dimension of MLP layer
		:agg: the inter-relation aggregator that output the final embedding
		:lambad 1: the weight of MLP layer (ignore it)
		:prior:prior
		"""

		self.agg = agg
		self.cuda = cuda
		#self.lambda_1 = lambda_1

		self.K = K #how many layers
		self.prior = prior
		self.xent = nn.CrossEntropyLoss()
		self.embed_dim = embed_dim
		self.fun = nn.LeakyReLU(0.3)


		self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes)) #Default requires_grad = True
		self.weight_model = nn.Parameter(torch.FloatTensor((int(math.pow(2, K+1)-1) * self.embed_dim), 64))

		self.weight_model2 = nn.Parameter(torch.FloatTensor(64, num_classes))

		
		init.xavier_uniform_(self.weight_mlp)
		init.xavier_uniform_(self.weight_model)
		init.xavier_uniform_(self.weight_model2)

	def forward(self, nodes, train_flag = True):

		embedding = self.agg(nodes, train_flag)

		scores_model = embedding.mm(self.weight_model)
		scores_model = self.fun(scores_model)
		scores_model = scores_model.mm(self.weight_model2)
		#scores_model = self.fun(scores_model)

		scores_mlp = embedding[:, 0: self.embed_dim].mm(self.weight_mlp)
		scores_mlp = self.fun(scores_mlp)

		return scores_model, scores_mlp
		#dimension, the number of center nodes * 2
	
	def to_prob(self, nodes, train_flag = False):

		scores_model, scores_mlp = self.forward(nodes, train_flag)
		scores_model = torch.sigmoid(scores_model)
		return scores_model


	def loss(self, nodes, labels, train_flag = True):

		#the classification module

		if self.cuda:
			logits = (torch.from_numpy(self.prior +1e-8)).cuda()
		else:
			logits = (torch.from_numpy(self.prior +1e-8))

		scores_model, scores_mlp = self.forward(nodes, train_flag)

		scores_model = scores_model + torch.log(logits)
		scores_mlp = scores_mlp + torch.log(logits)

		loss_model = self.xent(scores_model, labels.squeeze())
		#loss_mlp = self.xent(scores_mlp, labels.squeeze())
		final_loss = loss_model #+ self.lambda_1 * loss_mlp
		return final_loss
