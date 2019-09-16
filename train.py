import os
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

from graph_stat import *
from options import Options
from GVGAN import *
from utils import *
import pprint

warnings.filterwarnings("ignore")



def load_data(DATA_DIR):
	# script for loading NWE dblp
	# folder structure
	# - this.ipynb
	# - $DATA_DIR - *.txt

	mat_names = []  # e.g. GSE_2304
	adj_mats = []  # essential data, type: list(np.ndarray)
	attr_vecs = []  # essential data, type: list(np.ndarray)
	id_maps = []  # map index to gene name if you need

	for f in os.listdir(DATA_DIR):
		if not f.startswith(('nodes', 'links', 'attrs')):
			continue
		else:
			mat_names.append('_'.join(f.split('.')[0].split('_')[1:]))
	mat_names = sorted([it for it in set(mat_names)])
	print('Test length', len(mat_names))
	for mat_name in mat_names:
		node_file = 'nodes_' + mat_name + '.txt'
		link_file = 'links_' + mat_name + '.txt'
		attr_file = 'attrs_' + mat_name + '.txt'
		node_file_path = os.path.join(DATA_DIR, node_file)
		link_file_path = os.path.join(DATA_DIR, link_file)
		attr_file_path = os.path.join(DATA_DIR, attr_file)

		id_to_item = {}
		with open(node_file_path, 'r') as f:
			for i, line in enumerate(f):
				author = line.rstrip('\n')
				id_to_item[i] = author
		all_ids = set(id_to_item.keys())

		with open(attr_file_path, 'r') as f:
			attr_vec = np.loadtxt(f).T.flatten()
			attr_vecs.append(attr_vec)

		links = defaultdict(set)
		with open(link_file_path, 'r') as f:
			for line in f:
				cells = line.rstrip('\n').split(',')
				from_id = int(cells[0])
				to_id = int(cells[1])
				if from_id in all_ids and to_id in all_ids:
					links[from_id].add(to_id)

		N = len(all_ids)
		adj = np.zeros((N, N))
		for from_id in range(N):
			for to_id in links[from_id]:
				adj[from_id, to_id] = 1
				adj[to_id, from_id] = 1

		adj -= np.diag(np.diag(adj))
		id_map = [id_to_item[i] for i in range(N)]

		# Remove small component
		# adj = remove_small_conns(adj, keep_min_conn=4)

		# Keep large component
		adj = keep_topk_conns(adj, k=1)
		adj_mats.append(adj)
		id_maps.append(id_map)

		if int(np.sum(adj)) == 0:
			adj_mats.pop(-1)
			id_maps.pop(-1)
			mat_names.pop(-1)
			attr_vecs.pop(-1)

	train_adj_mats = adj_mats[:int(len(adj_mats) * .8)]
	test_adj_mats = adj_mats[int(len(adj_mats) * .8):]
	train_attr_vecs = attr_vecs[:int(len(attr_vecs) * .8)]
	test_attr_vecs = attr_vecs[int(len(attr_vecs) * .8):]
	return train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs


def train(train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs, opt=None):
	training_index = list(range(0, len(train_adj_mats)))

	max_epochs = opt.max_epochs
	for epoch in range(max_epochs):
		D_real_list, D_rec_enc_list, D_rec_noise_list, D_list, Encoder_list = [], [], [], [], []
		# g_loss_list, rec_loss_list, prior_loss_list = [], [], []
		g_loss_list, rec_loss_list, prior_loss_list, aa_loss_list = [], [], [], []
		random.shuffle(training_index)
		for i in training_index:

			ones_label = Variable(torch.ones(1)).cuda()
			zeros_label = Variable(torch.zeros(1)).cuda()
			# adj = Variable(train_adj_mats[i]).cuda()
			adj = Variable(torch.from_numpy(train_adj_mats[i]).float()).cuda()

			# if adj.shape[0] <= d_size + 2 :
			#    continue
			if adj.shape[0] <= opt.d_size + 2:
				continue
			if opt.av_size == 0:
				attr_vec = None
			else:
				# attr_vec = Variable(train_attr_vecs[i, :]).cuda()
				attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

			# edge_num = train_adj_mats[i].sum()
			G.set_attr_vec(attr_vec)
			D.set_attr_vec(attr_vec)

			norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
			pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
			# print('pos_weight', pos_weight)

			mean, logvar, rec_adj = G(adj)

			noisev = torch.randn(mean.shape, requires_grad=True).cuda()
			noisev = cat_attr(noisev, attr_vec)
			rec_noise = G.decoder(noisev)

			e = int(np.sum(train_adj_mats[i])) // 2

			c_adj = topk_adj(F.sigmoid(rec_adj), e * 2)
			c_noise = topk_adj(F.sigmoid(rec_noise), e * 2)

			# train discriminator
			output = D(adj)
			errD_real = criterion_bce(output, ones_label)
			D_real_list.append(output.data.mean())
			# output = D(rec_adj)
			output = D(c_adj)
			errD_rec_enc = criterion_bce(output, zeros_label)
			D_rec_enc_list.append(output.data.mean())
			# output = D(rec_noise)
			output = D(c_noise)

			errD_rec_noise = criterion_bce(output, zeros_label)
			D_rec_noise_list.append(output.data.mean())

			dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
			# print ("print (dis_img_loss)", dis_img_loss)
			D_list.append(dis_img_loss.data.mean())
			opt_dis.zero_grad()
			dis_img_loss.backward(retain_graph=True)
			opt_dis.step()

			# AA_loss b/w rec_adj and adj
			# aa_loss = loss_MSE(rec_adj, adj)

			loss_BCE_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
			loss_BCE_logits.cuda()

			aa_loss = loss_BCE_logits(rec_adj, adj)

			# print(c_adj,c_adj)
			# aa_loss = loss_BCE(c_adj, adj)

			# train decoder
			output = D(adj)
			errD_real = criterion_bce(output, ones_label)
			# output = D(rec_adj)
			output = D(c_adj)

			errD_rec_enc = criterion_bce(output, zeros_label)
			errG_rec_enc = criterion_bce(output, ones_label)
			# output = D(rec_noise)
			output = D(c_noise)

			errD_rec_noise = criterion_bce(output, zeros_label)
			errG_rec_noise = criterion_bce(output, ones_label)

			similarity_rec_enc = D.similarity(c_adj)
			similarity_data = D.similarity(adj)

			dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
			# print (dis_img_loss)
			# gen_img_loss = norm*(aa_loss + errG_rec_enc  + errG_rec_noise)- dis_img_loss #- dis_img_loss #aa_loss #+ errG_rec_enc  + errG_rec_noise # - dis_img_loss
			gen_img_loss = - dis_img_loss  # norm*(aa_loss) #

			g_loss_list.append(gen_img_loss.data.mean())
			rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
			rec_loss_list.append(rec_loss.data.mean())
			# err_dec =  gamma * rec_loss + gen_img_loss

			err_dec = opt.gamma * rec_loss + gen_img_loss
			opt_dec.zero_grad()
			err_dec.backward(retain_graph=True)
			opt_dec.step()

			# train encoder
			# fix me: sum version of prior loss
			pl = []
			for j in range(mean.size()[0]):
				prior_loss = 1 + logvar[j, :] - mean[j, :].pow(2) - logvar[j, :].exp()
				prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean[j, :].data)
				pl.append(prior_loss)
			prior_loss_list.append(sum(pl))
			err_enc = sum(pl) + gen_img_loss + opt.beta * (rec_loss)  # + beta2* norm* aa_loss
			opt_enc.zero_grad()
			err_enc.backward()
			opt_enc.step()
			Encoder_list.append(err_enc.data.mean())

		print('[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f, Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f'
			     %(epoch,
				 max_epochs,
				 torch.mean(torch.stack(D_real_list)),
				 torch.mean(torch.stack(D_rec_enc_list)),
				 torch.mean(torch.stack(D_rec_noise_list)),
				 torch.mean(torch.stack(D_list)),
				 torch.mean(torch.stack(g_loss_list)),
				 torch.mean(torch.stack(rec_loss_list)),
				 torch.mean(torch.stack(prior_loss_list))))

		print('Training set')
		for i in range(3):
			base_adj = train_adj_mats[i]

			if base_adj.shape[0] <= opt.d_size:
				continue
			print('Base Adj_size: ', base_adj.shape)
			attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

			# add a new line
			G.set_attr_vec(attr_vec)

			print('Show sample')
			sample_adj = gen_adj(G, base_adj.shape[0], int(np.sum(base_adj)) // 2, attr_vec, z_size=opt.z_size)


# show_graph(sample_adj, base_adj=base_adj, remove_isolated=True)

if __name__ == '__main__':

	print('=========== OPTIONS ===========')
	pprint.pprint(vars(opt))
	print(' ======== END OPTIONS ========\n\n')

	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

	train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data(
		DATA_DIR=opt.DATA_DIR)

	# output_dir = opt.output_dir
	train(train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs, opt=opt)
