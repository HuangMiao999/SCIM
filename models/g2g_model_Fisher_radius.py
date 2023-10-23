
#----------Graph Guassian embedding with Fisher distance as energy loss


import argparse
import random

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import umap
from sklearn.datasets import make_swiss_roll,make_s_curve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




class CompleteKPartiteGraph:
    """A complete k-partite graph
    """

    def __init__(self, partitions):
        """
        Parameters
        ----------
        partitions : [[int]]
            List of node partitions where each partition is list of node IDs
        """

        self.partitions = partitions
        self.counts = np.array([len(p) for p in partitions])
        self.total = self.counts.sum()

        assert len(self.partitions) >= 2
        assert np.all(self.counts > 0)

        # Enumerate all nodes so that we can easily look them up with an index
        # from 1..total
        self.nodes = np.array([node for partition in partitions for node in partition])

        # Precompute the partition count of each node
        self.n_i = np.array(
            [n for partition, n in zip(self.partitions, self.counts) for _ in partition]
        )

        # Precompute the start of each node's partition in self.nodes
        self.start_i = np.array(
            [
                end - n
                for partition, n, end in zip(
                    self.partitions, self.counts, self.counts.cumsum()
                )
                for node in partition
            ]
        )

        # Each node has edges to every other node except the ones in its own
        # level set
        self.out_degrees = np.full(self.total, self.total) - self.n_i

        # Sample the first nodes proportionally to their out-degree
        self.p = self.out_degrees / self.out_degrees.sum()

    def sample_edges(self, size=1):
        """Sample edges (j, k) from this graph uniformly and independently

        Returns
        -------
        ([j], [k])
        j will always be in a lower partition than k
        """

        # Sample the originating nodes for each edge
        j = np.random.choice(self.total, size=size, p=self.p, replace=True)

        # For each j sample one outgoing edge uniformly
        #
        # Se we want to sample from 1..n \ start[j]...(start[j] + count[j]). We
        # do this by sampling from 1..#degrees[j] and if we hit a node

        k = np.random.randint(self.out_degrees[j])
        filter = k >= self.start_i[j]
        k += filter.astype(int) * self.n_i[j]

        # Swap nodes such that the partition index of j is less than that of k
        # for each edge
        wrong_order = k < j
        tmp = k[wrong_order]
        k[wrong_order] = j[wrong_order]
        j[wrong_order] = tmp

        # Translate node indices back into user configured node IDs
        j = self.nodes[j]
        k = self.nodes[k]

        return j, k


class AttributedGraph:
    def __init__(self, A, X, z, K, mode='common'):
        """
        A: distances matrix
        X: attribute vector of nodes
        z: label of nodes
        K: the highest order of neighbours (see Gaussian Embeedding)
        """
        self.A = A
        self.X = torch.tensor(X)#torch.tensor(X.toarray())
        self.z = z
        self.level_sets = level_sets(A, K, mode)

        # Precompute the cardinality of each level set for every node
        self.level_counts = {
            node: np.array(list(map(len, level_sets)))
            for node, level_sets in self.level_sets.items()
        }

        # Precompute the weights of each node's expected value in the loss
        N = self.level_counts
        self.loss_weights = 0.5 * np.array(
            [N[i][1:].sum() ** 2 - (N[i][1:] ** 2).sum() for i in self.nodes()]
        )

        n = self.A.shape[0]
        self.neighborhoods = [None] * n
        for i in range(n):
            ls = self.level_sets[i]
            if len(ls) >= 3:
                self.neighborhoods[i] = CompleteKPartiteGraph(ls[1:])

    def nodes(self):
        return range(self.A.shape[0])

    def eligible_nodes(self):
        """Nodes that can be used to compute the loss"""
        N = self.level_counts

        # If a node only has first-degree neighbors, the loss is undefined
        return [i for i in self.nodes() if len(N[i]) >= 3]

    def sample_two_neighbors(self, node, size=1):
        """Sample to nodes from the neighborhood of different rank"""

        level_sets = self.level_sets[node]
        if len(level_sets) < 3:
            raise Exception(f"Node {node} has only one layer of neighbors")

        return self.neighborhoods[node].sample_edges(size)


class GraphDataset(IterableDataset):
    """A dataset that generates all necessary information for one training step

    Sampling the edges is actually the most expensive part of the whole training
    loop and by putting it in the dataset generator, we can parallelize it
    independently from the training loop.
    """

    def __init__(self, graph, nsamples, iterations):
        self.graph = graph
        self.nsamples = nsamples
        self.iterations = iterations

    def __iter__(self):
        graph = self.graph
        nsamples = self.nsamples

        eligible_nodes = list(graph.eligible_nodes())
        nrows = len(eligible_nodes) * nsamples

        weights = torch.empty(nrows)

        for _ in range(self.iterations):
            i_indices = torch.empty(nrows, dtype=torch.long)
            j_indices = torch.empty(nrows, dtype=torch.long)
            k_indices = torch.empty(nrows, dtype=torch.long)
            for index, i in enumerate(eligible_nodes):
                start = index * nsamples
                end = start + nsamples
                i_indices[start:end] = i

                js, ks = graph.sample_two_neighbors(i, size=nsamples)
                j_indices[start:end] = torch.tensor(js)
                k_indices[start:end] = torch.tensor(ks)

                weights[start:end] = graph.loss_weights[i]

            yield graph.X, i_indices, j_indices, k_indices, weights, nsamples


def gather_rows(input, index):
    """Gather the rows specificed by index from the input tensor"""
    return torch.gather(input, 0, index.unsqueeze(-1).expand((-1, input.shape[1])))


class Encoder(nn.Module):
    def __init__(self, D, L):
        """Construct the encoder

        Parameters
        ----------
        D : int
            Dimensionality of the node attributes
        L : int
            Dimensionality of the embedding

        """
        super().__init__()

        self.D = D
        self.L = L

        def xavier_init(layer):
            nn.init.xavier_normal_(layer.weight)
            # TODO: Initialize bias with xavier but pytorch cannot compute the
            # necessary fan-in for 1-dimensional parameters

        self.linear1 = nn.Linear(D, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear_mu = nn.Linear(128, L)
        self.linear_sigma = nn.Linear(128, L)

        xavier_init(self.linear1)
        xavier_init(self.linear2)
        xavier_init(self.linear_mu)
        xavier_init(self.linear_sigma)

    def forward(self, node):
        h = F.relu(self.linear1(node))
        h = F.relu(self.linear2(h))
        mu = self.linear_mu(h)
        sigma = F.elu(self.linear_sigma(h)) + 1 # ELU有负区间，保持方差正性

        return mu, sigma

    def compute_loss(self, X, i, j, k, w, nsamples):
        """Compute the energy-based loss from the paper
        """

        mu, sigma = self.forward(X)

        mu_i = gather_rows(mu, i)
        sigma_i = gather_rows(sigma, i)
        mu_j = gather_rows(mu, j)
        sigma_j = gather_rows(sigma, j)
        mu_k = gather_rows(mu, k)
        sigma_k = gather_rows(sigma, k)

        diff_ij = mu_i - mu_j
        ratio_ji = sigma_j / sigma_i
        ss_ij=torch.sqrt(sigma_i)+torch.sqrt(sigma_j)
        ds_ij=torch.sqrt(sigma_i)-torch.sqrt(sigma_j)
        
        closer = 2*((torch.log (
            (torch.sqrt(0.5*diff_ij**2+ss_ij**2)+torch.sqrt(0.5*diff_ij**2+ds_ij**2))/\
            (torch.sqrt(0.5*diff_ij**2+ss_ij**2)-torch.sqrt(0.5*diff_ij**2+ds_ij**2))
             ))**2).sum(axis=-1)

        diff_ik = mu_i - mu_k
        ratio_ki = sigma_k / sigma_i
        ss_ik=torch.sqrt(sigma_i)+torch.sqrt(sigma_k)
        ds_ik=torch.sqrt(sigma_i)-torch.sqrt(sigma_k)
        
        apart = 2*((torch.log (
            (torch.sqrt(0.5*diff_ik**2+ss_ik**2)+torch.sqrt(0.5*diff_ik**2+ds_ik**2))/\
            (torch.sqrt(0.5*diff_ik**2+ss_ik**2)-torch.sqrt(0.5*diff_ik**2+ds_ik**2)+1e-10)
             ))**2).sum(axis=-1)

        E = closer  + torch.exp(-torch.sqrt(apart)) 

        loss = E.dot(w) / nsamples

        return loss


def level_sets(A, K, mode="common"):
    """Enumerate the level sets for each node's neighborhood

    Parameters
    ----------
    A : np.array
        Distances matrix or Adjacency matrix
    K : int?
        Maximum path length to consider

        All nodes that are further apart go into the last level set.

    Returns
    -------
    { node: [i -> i-hop neighborhood] }
    """

    if A.shape[0] == 0 or A.shape[1] == 0:
        return {}


    # KNN modes
    if mode == "KNN":
        # Compute the shortest path length between any two nodes
        D = scipy.sparse.csgraph.shortest_path(
            A, method="D", unweighted=False, directed=False
        )

        # Cast to int so that the distances can be used as indices
        #
        # D has inf for any pair of nodes from different cmponents and np.isfinite
        # is really slow on individual numbers so we call it only once here
        D[np.logical_not(np.isfinite(D))] = -1.0
        D = D.astype(int)

        # Handle nodes farther than K as if they were unreachable
        if K is not None:
            D[D > K] = -1

        # Read the level sets off the distance matrix
        set_counts = D.max(axis=1)
        sets = {i: [[] for _ in range(1 + set_counts[i] + 1)] for i in range(D.shape[0])}
        for i in range(D.shape[0]):
            sets[i][0].append(i)

            for j in range(i):
                d = D[i, j]

                # If a node is unreachable, add it to the outermost level set. This
                # trick ensures that nodes from different connected components get
                # pushed apart and is essential to get good performance.
                if d < 0:
                    sets[i][-1].append(j)
                    sets[j][-1].append(i)
                else:
                    sets[i][d].append(j)
                    sets[j][d].append(i)
    

    # radius layers
    elif mode == 'r-mean':
    
        # Read the level sets off the distances matrix
        sets = {i: [[] for _ in range(1+K+1)] for i in range(A.shape[0])} 
        # K+2 layers 0:itself, 1-K:K-hop, K+1:further nodes
        

        # compute the radius
        radius = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            idx = np.argsort(A[i,:])[1:31] # hyper-parameter 30-NN means to construct radius
            radius[i] = np.mean(A[i,idx])


        # adding nodes into level sets

            for j in range(A.shape[0]):
                if j==i:
                    sets[i][0].append(i)
                else:
                    d = A[i, j]
                    n = int(d//radius[i])+1 # the node is in the n-th layers
                    if n <= K:
                        sets[i][n].append(j)
                    else:
                        sets[i][-1].append(j)
            
        # remove the empty layers to fit the codes in CompleteKPartiteGraph ''assert np.all(self.counts > 0)''
            for idx in range(sets[i].count([])):
                sets[i].remove([])
    
    elif mode == 'r-max':
    
        # Read the level sets off the distances matrix
        sets = {i: [[] for _ in range(1+K+1)] for i in range(A.shape[0])} 
        # K+2 layers 0:itself, 1-K:K-hop, K+1:further nodes
        

        # compute the radius
        radius = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            idx = np.argsort(A[i,:])[1:11] # hyper-parameter 30-NN means to construct radius
            radius[i] = np.max(A[i,idx]).item()


        # adding nodes into level sets

            for j in range(A.shape[0]):
                if j==i:
                    sets[i][0].append(i)
                else:
                    d = A[i, j]
                    n = int(d//radius[i])+1 # the node is in the n-th layers
                    if n <= K:
                        sets[i][n].append(j)
                    else:
                        sets[i][-1].append(j)
            
        # remove the empty layers to fit the codes in CompleteKPartiteGraph ''assert np.all(self.counts > 0)''
            for idx in range(sets[i].count([])):
                sets[i].remove([])
    
    
    elif mode == 'common':
    
        # Read the level sets off the distances matrix
        sets = {i: [[] for _ in range(1+K+1)] for i in range(A.shape[0])} 
        # K+2 layers 0:itself, 1-K:K-hop, K+1:further nodes
        

        # compute the radius
        radius = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            idx = np.argsort(A[i,:])[1:11] # hyper-parameter 10-NN max to construct radius
            radius[i] = np.max(A[i,idx])

        radius = np.max(radius)
        print(radius)

        # adding nodes into level sets
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if j==i:
                    sets[i][0].append(i)
                else:
                    d = A[i, j]
                    n = int(d//radius)+1 # the node is in the n-th layers
                    if n <= K:
                        sets[i][n].append(j)
                    else:
                        sets[i][-1].append(j)
            
        # remove the empty layers to fit the codes in CompleteKPartiteGraph ''assert np.all(self.counts > 0)''
            for idx in range(sets[i].count([])):
                sets[i].remove([])


    else:
        print("undefined mode!!!")
        raise NotImplementedError


    return sets


def train_test_split(n, train_ratio=0.5):
    nodes = list(range(n))
    split_index = int(n * train_ratio)

    random.shuffle(nodes)
    return nodes[:split_index], nodes[split_index:]


def reset_seeds(seed=None):
    if seed is None:
        seed = get_worker_info().seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def training(encoder, optimizer, train_data, epochs=200, nsamples=5, learning_rate=1e-3):
    seed = 0
    if seed is not None:
        reset_seeds(seed)
    iterations = epochs
    dataset = GraphDataset(train_data, nsamples, iterations)
    loader = DataLoader(
        dataset,
        batch_size=1,
    #     num_workers=n_workers,
        worker_init_fn=reset_seeds,
        collate_fn=lambda args: args,
    )

    # for epoch in range(1, epochs + 1):
    #     print(epoch)

    for batch_idx, data in enumerate(loader):
    #     print(batch_idx)
        encoder.train()
        optimizer.zero_grad()
        loss = encoder.compute_loss(data[0][0],data[0][1],data[0][2],data[0][3],data[0][4],data[0][5])
        if batch_idx% 10 == 9:
            print(batch_idx,loss)
        loss.backward()
        optimizer.step()
    return encoder
