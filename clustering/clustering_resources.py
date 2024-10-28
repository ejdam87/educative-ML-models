import sys, csv, datetime
import numpy
import matplotlib.pyplot as plt
import pdb

# datetime can be used for tracking running time
# tic = datetime.datetime.now()
# [running some function that takes time]
# tac = datetime.datetime.now()
# print("Function running time: %s" % (tac-tic))
# print("The function took %s seconds to complete" % (tac-tic).total_seconds())


# For repeatability of experiments using randomness
class RandomNumGen:

    generator = None
    seed = None

    @classmethod
    def set_seed(tcl, seed=None):
        tcl.seed = seed
        if seed is not None:
            tcl.generator = numpy.random.Generator(numpy.random.MT19937(numpy.random.SeedSequence(seed)))
        else:
            tcl.generator = None

    @classmethod
    def get_gen(tcl):
        if tcl.generator is None:
            return numpy.random.default_rng()
        return tcl.generator 


# Load a CSV file
def load_csv(filename, last_column_str=False, normalize=False, as_int=False):
    dataset = list()
    head = None
    classes = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for ri, row in enumerate(csv_reader):
            if not row:
                continue
            if ri == 0:
                head = row
            else:
                rr = [r.strip() for r in row]
                if last_column_str:
                    if rr[-1] not in classes:
                        classes[rr[-1]] = len(classes)
                    rr[-1] = classes[rr[-1]]
                dataset.append([float(r) for r in rr])
    dataset = numpy.array(dataset)
    if not last_column_str and len(numpy.unique(dataset[:,-1])) <= 10:
        classes = dict([("%s" % v, v) for v in numpy.unique(dataset[:,-1])])
    if normalize:
        dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    return dataset, head, classes

# Find the min and max values for each column
def dataset_minmax(dataset):
    return numpy.vstack([numpy.min(dataset, axis=0), numpy.max(dataset, axis=0)])

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax=None):
    if minmax is None:
        minmax = dataset_minmax(dataset)
    return (dataset - numpy.tile(minmax[0, :], (dataset.shape[0], 1))) / numpy.tile(minmax[1, :]-minmax[0, :], (dataset.shape[0], 1))

# Sample k random points from the domain 
def sample_domain(k, minmax=None, dataset=None):
    if dataset is not None:
        minmax = dataset_minmax(dataset)
    if minmax is None:
        return RandomNumGen.get_gen().random(k)
    d = RandomNumGen.get_gen().random((k, minmax.shape[1]))
    return numpy.tile(minmax[0, :], (k, 1)) + d*numpy.tile(minmax[1, :]-minmax[0, :], (k, 1))

# Compute distances between two sets of instances
def L2_distance(A, B):
    return numpy.vstack([numpy.sqrt(numpy.sum((A - numpy.tile(B[i,:], (A.shape[0], 1)))**2, axis=1)) for i in range(B.shape[0])]).T

def L1_distance(A, B):
    return numpy.vstack([numpy.sum(numpy.abs(A - numpy.tile(B[i,:], (A.shape[0], 1))), axis=1) for i in range(B.shape[0])]).T

# Calculate contingency matrix
def contingency_matrix(actual, predicted, weights=None):
    if weights is None:
        weights = numpy.ones(actual.shape[0], dtype=int)
    ac_int = actual.astype(int)
    prd_int = predicted.astype(int)
    nb_ac = numpy.maximum(2, numpy.max(ac_int)+1) + 1*numpy.any(ac_int == -1)
    nb_prd = numpy.maximum(2, numpy.max(prd_int)+1) + 1*numpy.any(prd_int == -1)
    counts = numpy.zeros((nb_prd, nb_ac, 2), dtype=type(weights[0]))
    for p,a,w in zip(prd_int, ac_int, weights):
        counts[p, a, 0] += 1
        counts[p, a, 1] += w
    return counts

### Evaluation of clusterings
##### utility function to turn a vector of cluster ids into lists of instances ids
def clabels_to_groups(clabels, exclude=[]):
    groups = []
    for l in sorted(set(clabels)):
        if l not in exclude:
            groups.append(numpy.where(clabels==l)[0])
    return groups

#### Internal validation criteria
def ss_distances_centroids(X, clabels, L1_dist=False):
    ssdc = 0
    groups = clabels_to_groups(clabels)
    for g in groups:
        if L1_dist:
            center = numpy.median(X[g,:], axis=0)
            gd = L1_distance(X[g,:], numpy.array([center]))
        else:
            center = numpy.mean(X[g,:], axis=0)
            gd = L2_distance(X[g,:], numpy.array([center]))
        ssdc += numpy.sum(gd**2)
    return ssdc

def intra_inter_distance_ratio(distM, clabels):
    Ct = numpy.tile(clabels, (len(clabels), 1))
    mask_sameclass = (Ct == Ct.T) & (Ct != -1) & (numpy.diag(numpy.ones(len(clabels))) != 1)
    mask_diffclass = (Ct != Ct.T) & (Ct != -1) & ( -1 != Ct.T)    
    return numpy.mean(distM[mask_sameclass])/numpy.mean(distM[mask_diffclass])

def clust_silhouette(distM, clabels):
    uniq_labels = numpy.array(sorted(set(clabels).difference([-1])), dtype=int)
    # uniq_labels = numpy.array(sorted(set(clabels)), dtype=int) # consider outliers as one separate cluster
    sumM = numpy.array([numpy.sum(distM[clabels==l, :], axis=0) for l in uniq_labels])
    countsM = numpy.tile(numpy.array([numpy.sum(clabels==l) for l in uniq_labels]), (len(clabels), 1)).T
    ## no discount for singleton clusters to avoid division by zero
    discountM = numpy.array([1*((clabels==l) & (numpy.sum(clabels==l) > 1)) for l in uniq_labels])
    avgM = numpy.divide(sumM, (countsM-discountM))

    silhouette_v = numpy.zeros(clabels.shape)
    for l in uniq_labels:
        Do = numpy.min(avgM[:, clabels==l][uniq_labels!=l, :], axis=0)
        Ds = avgM[:, clabels==l][uniq_labels==l, :][0]
        silhouette_v[clabels==l] = (Do - Ds)/numpy.max([Do, Ds], axis=0)
    # return numpy.mean(silhouette_v), silhouette_v # consider outliers as one separate cluster
    return numpy.mean(silhouette_v[clabels != -1]), silhouette_v

#### measures to quantify the degree of agreement between two clusterings, given the corresponding contingency matrix
def purity_CM(contingency_matrix):
    return numpy.sum(numpy.max(contingency_matrix, axis=0))/numpy.sum(contingency_matrix)

def gini_CM(contingency_matrix):
    marg_rows = numpy.sum(contingency_matrix, axis=0)
    return numpy.sum((1 - numpy.sum(numpy.divide(contingency_matrix, numpy.tile(marg_rows, (contingency_matrix.shape[0], 1)))**2, axis=0))* marg_rows/numpy.sum(contingency_matrix))

def entropy_CM(contingency_matrix):
    marg_rows = numpy.sum(contingency_matrix, axis=0)
    ps = numpy.divide(contingency_matrix, numpy.tile(marg_rows, (contingency_matrix.shape[0], 1)))
    ps[contingency_matrix==0] = 1
    return numpy.sum(numpy.sum(-ps*numpy.log2(ps), axis=0) * marg_rows/numpy.sum(contingency_matrix))
metrics_clust_cm = {"purity": purity_CM, "gini": gini_CM, "entropy": entropy_CM}

#### given two clusterings A and B as two vectors of cluster ids, compute the contingency matrix and measures to quantify agreement
def get_clust_CM_vals(clabelsA, clabelsB, weights=None, vks=None):
    if vks is None:
        vks = metrics_clust_cm.keys()
    cm = contingency_matrix(clabelsA, clabelsB)
    if weights is None:
        cm = cm[:, :, 0]
    else:
        cm = cm[:, :, 1]
    vals = {}
    for vk in vks:
        if vk in metrics_clust_cm:
            vals[vk] = metrics_clust_cm[vk](cm)
    return vals, cm
 
if __name__ == '__main__':
    
    ### For repeatability, set the seed of the seudo-random number generator to some large integer of your choice. In particular, set the value of SEED below to your student number.
    # SEED = 0
    # RandomNumGen.set_seed(SEED)
    data_params = {"filename": "Dry_Bean_Dataset_small.csv", "last_column_str": True}
    dataset, head, classes = load_csv(**data_params)

    ## Dropping the last column, which contains class labels, and normalizing the data
    Dn = normalize_dataset(dataset[:,:-1])
    ## sampling four points from the domain of the data
    centers = sample_domain(k=4, dataset=Dn)
    ## computing the Euclidean distance between each data point and each of the sampled centers,
    ### distPC[i,j] = L_2 dstance between i^th data point and j^th center
    distPC = L2_distance(Dn, centers)

    
    ## The following is an example demonstrate the use of evaluation functions
    ##  considering two dummy assignment vectors corresponding each to five clusters over the dataset
    clabelsA = numpy.floor(5*(numpy.arange(dataset.shape[0])[::-1]/dataset.shape[0]))
    clabelsB = numpy.floor(5*((numpy.arange(dataset.shape[0])/dataset.shape[0])**2))

    ### matrix of pairwise distances between points in the dataset
    distM = L2_distance(Dn, Dn)
    print("Clustering A:\t SSDC=%.4f, DR=%.4f, S=%.4f" % (ss_distances_centroids(Dn, clabelsA),
                                                        intra_inter_distance_ratio(distM, clabelsA),
                                                        clust_silhouette(distM, clabelsA)[0]))
    vals, cm = get_clust_CM_vals(clabelsA, clabelsB)
    print("contingency matrix:\n", cm) ## what do the rows and the columns respectively represent?
    print(", ".join(["%s=%.4f" % (k,v) for (k,v) in vals.items()]))

