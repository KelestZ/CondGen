import torch
import torch.nn.functional as F
from sklearn.manifold import SpectralEmbedding
import warnings
from graph_stat import *

warnings.filterwarnings("ignore")
from pprint import pprint
import numpy as np

# import matplotlib.pyplot as plt
import networkx as nx
import copy


def show_graph(adj, base_adj=None, remove_isolated=True):
	if not isinstance(adj, np.ndarray):
		adj_ = adj.data.cpu().numpy()
	else:
		adj_ = copy.deepcopy(adj)

	adj_ -= np.diag(np.diag(adj_))

	gr = nx.from_numpy_array(adj_)
	assert ((adj_ == adj_.T).all())
	if remove_isolated:
		gr.remove_nodes_from(list(nx.isolates(gr)))
	nx.draw(gr, node_size=10)
	plt.title('gen')
	plt.show()

	d = compute_graph_statistics(adj_)
	pprint(d)

	if base_adj is not None:
		base_gr = nx.from_numpy_array(base_adj)
		nx.draw(base_gr, node_size=10)
		plt.title('base')
		plt.show()
		bd = compute_graph_statistics(base_adj)
		diff_d = {}
		for k in list(d.keys()):
			diff_d[k] = round(abs(d[k] - bd[k]), 4)
		print(diff_d.keys())
		print(diff_d.values())


def make_symmetric(m):
	m_ = torch.transpose(m)
	w = torch.max(m_, m_.T)
	return w


def make_adj(x, n):
	res = torch.zeros(n, n).cuda()
	i = 0
	for r in range(1, n):
		for c in range(r, n):
			res[r, c] = x[i]
			res[c, r] = res[r, c]
			i += 1
	return res


def cat_attr(x, attr_vec):
	if attr_vec is None:
		return x
	attr_mat = attr_vec.repeat(x.size()[0], 1)
	x = torch.cat([x, attr_mat], dim=1)
	return x


def get_spectral_embedding(adj, d):
	"""
	Given adj is N*N, return its feature mat N*D, D is fixed in model
	:param adj:
	:return:
	"""

	adj_ = adj.data.cpu().numpy()
	emb = SpectralEmbedding(n_components=d)
	res = emb.fit_transform(adj_)
	x = torch.from_numpy(res).float().cuda()
	return x


def normalize(adj):
	adj = adj.data.cpu().numpy()
	adj_ = adj + np.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
	degree_mat_sqrt = np.diag(np.power(rowsum, 0.5).flatten())
	adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_sqrt)
	return torch.from_numpy(adj_normalized).float().cuda()


def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	return sparse_to_tuple(adj_normalized)


def keep_topk_conns(adj, k=3):
	g = nx.from_numpy_array(adj)
	to_removes = [cp for cp in sorted(nx.connected_components(g), key=len)][:-k]
	for cp in to_removes:
		g.remove_nodes_from(cp)
	adj = nx.to_numpy_array(g)
	return adj


def remove_small_conns(adj, keep_min_conn=4):
	g = nx.from_numpy_array(adj)
	for cp in list(nx.algorithms.components.connected_components(g)):
		if len(cp) < keep_min_conn:
			g.remove_nodes_from(cp)
	adj = nx.to_numpy_array(g)
	return adj


def top_n_indexes(arr, n):
	idx = np.argpartition(arr, arr.size - n, axis=None)[-n:]
	width = arr.shape[1]
	return [divmod(i, width) for i in idx]


def topk_adj(adj, k):
	adj_ = adj.data.cpu().numpy()
	assert ((adj_ == adj_.T).all())
	adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)
	adj_ -= np.diag(np.diag(adj_))
	tri_adj = np.triu(adj_)
	inds = top_n_indexes(tri_adj, k // 2)
	res = torch.zeros(adj.shape)
	for ind in inds:
		i = ind[0]
		j = ind[1]
		res[i, j] = 1.0
		res[j, i] = 1.0
	return res.cuda()


def test_gen(model, n, attr_vec, z_size, twice_edge_num, bd=None):
	fixed_noise = torch.randn((n, z_size), requires_grad=True).cuda()
	if attr_vec is not None:
		fixed_noise = cat_attr(fixed_noise, attr_vec.cuda())
	a_ = model.decoder(fixed_noise)
	# print(F.sigmoid(a_))
	a_ = topk_adj(F.sigmoid(a_), twice_edge_num)
	# print(a_)
	if bd:
		show_graph(a_, bd)
	else:
		show_graph(a_)


def gen_adj(model, n, e, attr_vec, z_size):
	fixed_noise = torch.randn((n, z_size), requires_grad=True).cuda()
	fixed_noise = cat_attr(fixed_noise, attr_vec)
	rec_adj = model.decoder(fixed_noise)
	return topk_adj(rec_adj, e * 2)


def eval(adj, base_adj=None):
	if not isinstance(adj, np.ndarray):
		adj_ = adj.data.cpu().numpy()
	else:
		adj_ = copy.deepcopy(adj)

	adj_ -= np.diag(np.diag(adj_))
	gr = nx.from_numpy_array(adj_)
	assert ((adj_ == adj_.T).all())

	d = compute_graph_statistics(adj_)
	pprint(d)

	if base_adj is not None:
		# base_adj = base_adj.numpy()
		base_gr = nx.from_numpy_array(base_adj)
		bd = compute_graph_statistics(base_adj)
		diff_d = {}

		for k in list(d.keys()):
			diff_d[k] = round(abs(d[k] - bd[k]), 4)
	return diff_d

