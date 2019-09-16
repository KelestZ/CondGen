import argparse

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser("Training Parser")

	def initialize(self):
		parser = self.parser

		parser.add_argument('--av_size', type=int, default=10, help="set 0 if you do not need attr_vec")
		parser.add_argument('--z_size', type=int, default=6, help="noise vector")

		parser.add_argument('--rep_size', type=int, default=32, help="hidden vector")
		parser.add_argument('--d_size', type=int, default=2, help="d vector")
		parser.add_argument('--gc_size', type=int, default=16, help="gc vector")

		parser.add_argument('--adj_thresh', type=float, default=0.6, help='threshold of adj edges')
		parser.add_argument('--max_epochs', type=int, default=1, help='max epochs')

		parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
		parser.add_argument('--beta', type=int, default=5, help='beta')
		parser.add_argument('--beta2', type=float, default=0.1, help='beta2')
		parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
		parser.add_argument('--gamma', type=int, default=15, help='gamma')

		parser.add_argument('--gpu', type=str, default='0', help='gpu id')
		parser.add_argument('--DATA_DIR', type=str, default='./data/dblp/', help='output dir')

		parser.add_argument('--output_dir', type=str, default='./output/', help='output dir')


		opt = parser.parse_args()
		return opt