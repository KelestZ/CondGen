import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import warnings
warnings.filterwarnings("ignore")


from utils import *

class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'

class Encoder(nn.Module):
	def __init__(self, av_size, d_size, gc_size, z_size, rep_size):
		"""

		:param av_size: D_A
		:param d_size: D_X
		:param gc_size: D'
		:param z_size: z
		"""
		super(Encoder, self).__init__()
		# input parameters
		self.z_size = z_size
		self.attr_vec = None
		self.gc_size = gc_size
		self.z_size = z_size
		self.d_size = d_size
		self.av_size = av_size
		self.rep_size = rep_size

		self.gc = GraphConvolution(d_size + av_size, gc_size)
		self.gc_mu = GraphConvolution(gc_size, z_size)
		self.gc_logvar = GraphConvolution(gc_size, z_size)

		# self.mean = nn.Sequential( nn.Linear(gc_size, z_size))
		# self.logvar = nn.Sequential(nn.Linear(gc_size, z_size))

		self.mean = nn.Sequential(nn.Linear(self.gc_size, int(self.gc_size / 4)),
								  nn.BatchNorm1d(int(self.gc_size / 4)),
								  nn.ReLU(),
								  nn.Linear(int(self.gc_size / 4), self.z_size))

		self.logvar = nn.Sequential(nn.Linear(self.gc_size, int(self.gc_size / 4)),
									nn.BatchNorm1d(int(self.gc_size / 4)),
									nn.ReLU(),
									nn.Linear(int(self.gc_size / 4), self.z_size))

	def set_attr_vec(self, attr_vec):
		self.attr_vec = attr_vec

	def forward(self, adj):
		# print('adj size',adj.size())

		t0 = time.time()
		x = get_spectral_embedding(adj, d=self.d_size)
		# print('Encoder, before mean logvar', x.size())
		t1 = time.time()
		adj = normalize(adj)
		x = cat_attr(x, self.attr_vec)
		# print('Before gc', 'x,size', x.size(),'att size',self.attr_vec.size()  , 'adj.size', adj.size())
		x = F.relu(self.gc(x, adj))
		x = F.dropout(x, p=0.5)

		# print('After gc')
		# z_mean = self.gc_mu(x, adj)
		# z_logvar = self.gc_logvar(x, adj)

		# create graph embedding N*D' -> 1*D'
		# x = x.sum(0)
		z_mean = self.mean(x)
		z_logvar = self.logvar(x)

		# feature III here
		# z_mean = torch.mean(z_mean, 0)
		# z_mean = z_mean.repeat(z_logvar.shape[0], 1)

		return z_mean, z_logvar

class Decoder(nn.Module):
	def __init__(self, z_out_size, rep_size):
		"""
		:param z_out_size: = z_size + len(attr_vec)
		"""
		super(Decoder, self).__init__()
		self.z_out_size = z_out_size
		self.rep_size = rep_size
		'''
		self.decode = nn.Sequential(
			nn.Linear(z_out_size, 32),
			nn.ReLU(),
			nn.Linear(32, 8),
			nn.ReLU())
		'''
		self.decode = nn.Sequential(
			# nn.Linear(z_out_size, self.rep_size),
			# nn.BatchNorm1d(self.rep_size),
			# nn.ReLU(),
			nn.Linear(z_out_size, int(self.rep_size)),
			nn.BatchNorm1d(int(self.rep_size)),
			nn.ReLU(),
			nn.Linear(int(self.rep_size), int(self.rep_size / 4)),
			# nn.BatchNorm1d(int(self.rep_size/2)),
			# nn.ReLU()
		)  # nn.BatchNorm1d(int(self.rep_size/4)),

	def forward(self, z):
		x = self.decode(z)
		# x = z
		x = torch.mm(x, x.t())
		# x = F.sigmoid(x)
		return x

class Generator(nn.Module):
	def __init__(self, av_size, d_size, gc_size, z_size, z_out_size, rep_size):
		"""
		:param av_size: D_A
		:param d_size: D_X
		:param gc_size: D' = GCN(D_X + D_A)
		:param z_size: original z size
		:param z_out_size: z size + D_A (append attribute)
		"""

		super(Generator, self).__init__()
		self.attr_vec = None
		self.av_size = av_size
		self.d_zize = d_size
		self.z_size = z_size
		self.z_out_size = z_out_size
		self.rep_size = rep_size

		self.encoder = Encoder(av_size, d_size, gc_size, z_size, rep_size)
		self.decoder = Decoder(z_out_size, rep_size)

	def set_attr_vec(self, attr_vec):
		self.attr_vec = attr_vec
		self.encoder.set_attr_vec(attr_vec)

	def forward(self, adj, training=True):
		# print('Before encoder')
		mean, logvar = self.encoder(adj)
		# print('After encoder')
		if (training):
			std = logvar.mul(0.5).exp_()
			reparametrized_noise = torch.randn(mean.shape, requires_grad=True).cuda()
			reparametrized_noise = mean + std * reparametrized_noise
		else:
			reparametrized_noise = mean
		# print('mean',mean)
		# print('After variational inference')
		x = cat_attr(reparametrized_noise, self.attr_vec)
		# print('Before decoder')
		rec_x = self.decoder(x)
		return mean, logvar, rec_x

class Discriminator(nn.Module):
	def __init__(self, av_size, d_size, gc_size, rep_size):
		super(Discriminator, self).__init__()
		self.av_size = av_size
		self.attr_vec = None
		self.d_size = d_size
		self.gc_size = gc_size
		self.rep_size = rep_size
		self.gc = GraphConvolution(d_size + av_size, gc_size)
		self.gc2 = GraphConvolution(gc_size, 8)

		self.main = nn.Sequential(
			nn.Linear(gc_size, int(self.rep_size / 2)),
			nn.LeakyReLU(0.2),
			nn.Linear(int(self.rep_size / 2), 8),

			nn.LeakyReLU(0.2))

		self.sigmoid_output = nn.Sequential(
			nn.Linear(8, 1),
			nn.Sigmoid())

	def set_attr_vec(self, attr_vec):
		self.attr_vec = attr_vec

	def forward(self, adj):
		# get spectral embedding from adj, D = D_X
		x = get_spectral_embedding(adj, d=self.d_size)
		adj = normalize(adj)
		x = cat_attr(x, self.attr_vec)

		# GCN layer N*D -> N*D'
		x = F.relu(self.gc(x, adj))
		# x = F.relu(self.gc2(x, adj))
		# x = F.dropout(x, p=0.5)
		x = self.main(x)
		x = x.sum(0)
		x = self.sigmoid_output(x)

		return x

	def similarity(self, adj):
		# get spectral embedding from adj, D = D_X
		x = get_spectral_embedding(adj, d=self.d_size)

		# norm adj
		adj = normalize(adj)

		# concatenate attr mat, D = D_X + D_A
		x = cat_attr(x, self.attr_vec)

		# GCN layer N*D -> N*D'
		x = F.relu(self.gc(x, adj))
		# x = F.dropout(x, p=0.5)
		# x = F.relu(self.gc2(x, adj))
		# create graph embedding N*D' -> 1*D'
		x = self.main(x)
		x = x.mean(0)
		# skip the last sigmoid layer
		# x = self.main(x)

		return x