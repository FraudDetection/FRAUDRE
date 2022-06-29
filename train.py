import time
import argparse
from sklearn.model_selection import train_test_split

from model import MODEL
from layers import *
from utlis import *
#os.environ["CUDA_LAUNCH_BLOCKING"]="0"


"""
   Paper: FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance
   Source: https://github.com/FraudDetection/FRAUDRE
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='amazon', help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size 1024 for yelp, 256 for amazon.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. [0.1 for amazon and 0.001 for yelp]')
parser.add_argument('--lambda_1', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=64, help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int, default=71, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')

if(torch.cuda.is_available()):
	print("cuda is available")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if(args.cuda):
	print("runing with GPU")

print(f'run on {args.data}')

# load topology, feature, and label
homo, relation1, relation2, relation3, feat_data, labels = load_data(args.data)

# set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# train_test split
if args.data == 'yelp':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.80,
															random_state = 2, shuffle = True)

	# set prior
	num_1= len(np.where(y_train==1)[0])
	num_2= len(np.where(y_train==0)[0])
	p0 = (num_1/(num_1+num_2))
	p1 = 1- p0
	prior = np.array([p1, p0])

elif args.data == 'amazon':

	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify = labels[3305:],
															test_size = 0.90, random_state = 2, shuffle = True)

	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])
	#prior = np.array([0.9, 0.1])


# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data) 
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad = False)
if args.cuda:
	features.cuda()

# set input graph topology
adj_lists = [relation1, relation2, relation3]


# build model

# the first neural network layer (ego-feature embedding module)
mlp = MLP_(features, feat_data.shape[1], args.embed_dim, cuda = args.cuda)

#first convolution layer
intra1_1 = IntraAgg(cuda = args.cuda)
intra1_2 = IntraAgg(cuda = args.cuda)
intra1_3 = IntraAgg(cuda = args.cuda)
agg1 = InterAgg(lambda nodes: mlp(nodes), args.embed_dim, adj_lists, [intra1_1, intra1_2, intra1_3], cuda = args.cuda)


#second convolution layer
intra2_1 = IntraAgg(cuda = args.cuda)
intra2_2 = IntraAgg(cuda = args.cuda)
intra2_3 = IntraAgg(cuda = args.cuda)

#def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):
agg2 = InterAgg(lambda nodes: agg1(nodes), args.embed_dim*2, adj_lists, [intra2_1, intra2_2, intra2_3], cuda = args.cuda)
gnn_model = MODEL(2, 2, args.embed_dim, agg2, prior, cuda = args.cuda)
# gnn_model in one convolution layer
#gnn_model = MODEL(1, 2, args.embed_dim, agg1, prior, cuda = args.cuda)


if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_1)
performance_log = []

# train the model

overall_time = 0
for epoch in range(args.num_epochs):

	# gnn_model.train()
	# shuffle
	random.shuffle(idx_train)
	num_batches = int(len(idx_train) / args.batch_size) +1

	loss = 0.0
	epoch_time = 0

	#mini-batch training
	for batch in range(num_batches):

		print(f'Epoch: {epoch}, batch: {batch}')

		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(idx_train))

		batch_nodes = idx_train[i_start:i_end]

		batch_label = labels[np.array(batch_nodes)]

		optimizer.zero_grad()

		start_time = time.time()

		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))

		end_time = time.time()

		epoch_time += end_time - start_time

		loss.backward()
		optimizer.step()
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
	overall_time += epoch_time

	#testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:

		#gnn_model.eval()
		auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, gnn_model)
		performance_log.append([auc, precision, a_p, recall, f1])

print("The training time per epoch")
print(overall_time/args.num_epochs)
