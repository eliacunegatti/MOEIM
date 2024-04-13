
import pandas as pd
import operator as op

def inverse_ncr(combinations, r):
	"""
	"inverse" ncr function, given r and ncr, returns n
	:param ncr:
	:param r:
	:return:
	"""
	n = 1
	ncr_n = ncr(n, r)
	while ncr_n < combinations:
		n += 1
		ncr_n = ncr(n, r)
	return n


def to_csv(archiver, archiver_default, population_file) :
    """
	Saving MOEA results into .csv format for 3 obj functions.
	"""

    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    communities = []
    for item in archiver:
        nodes.append(str(item[0]))
        influence.append(round(item[1],2))
        n_nodes.append(item[2])
        communities.append(item[3])
    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["obj"] = communities
    df["nodes"] = nodes
    df.to_csv(population_file+"._normalized.csv", sep=",", index=False)

    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    communities = []
    for item in archiver_default:
        nodes.append(str(item[0]))
        influence.append(round(item[1],2))
        n_nodes.append(item[2])
        communities.append(item[3])
    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["communities"] = communities
    df["nodes"] = nodes
    df.to_csv(population_file+"._default.csv", sep=",", index=False)

def to_csv2(archiver, archiver_default, population_file, no_nodes):
    """
	Saving MOEA results into .csv format for 2 obj functions.
	""" 
    
    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    for item in archiver:
        nodes.append(str(item[0]))
        influence.append(round(item[1],2))
        n_nodes.append(len(item[0])/no_nodes)
    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["nodes"] = nodes
	
    df.to_csv(population_file+"_normalized.csv", sep=",", index=False)	
    print(f'Save into {population_file}')
    
    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    for item in archiver_default:
        nodes.append(str(item[0]))
        influence.append(round(item[1],2))
        n_nodes.append(int(item[2]))
    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["nodes"] = nodes
    df.to_csv(population_file+"_default.csv", sep=",", index=False)	
    print(f'Save into {population_file}')

def to_csv_4(archiver, population_file):
    """
	Saving MOEA results into .csv format for 3 obj functions.
	"""

    df = pd.DataFrame()
    nodes = []
    influence = []
    n_nodes = []
    communities = []
    time = []
    for item in archiver:
        nodes.append(str(item[0]))
        influence.append(round(item[1],2))
        n_nodes.append(item[2])
        communities.append(item[3])
        time.append(item[4])
    df["n_nodes"] = n_nodes
    df["influence"] = influence
    df["communities"] = communities
    df["time"] = time
    df["nodes"] = nodes
    
    df.to_csv(population_file+".csv", sep=",", index=False)


def plot_all_trend(all, archive,no_nodes, k,m, max_time, population_file):
    values = []
    arch = [sorted(list(set(x.candidate)), reverse=True) for x in archive]
    tag = []
    for item in arch:
        if str(item) in all.keys():
            values.append(all[str(item)])
            tag.append(item)
        else:
            print('item not found',item)
    assert len(values) == len(arch)
    archive_normalized = [[tag[idx],a[0], a[1], a[2], a[3], a[4], a[5]] for idx, a in enumerate(values)] 
    archive_default = [[tag[idx],a[0]*no_nodes,k*(1-a[1])+1, 1-a[2], 1-a[3],m*(1 -a[4]), max_time*a[5]] for idx, a in enumerate(values)] 

	#generate a dataframe from list of list 
    df = pd.DataFrame(archive_normalized, columns = ['seed_set', 'spread', 'seed', 'communities', 'fairness', 'budget', 'time'])	
    df.to_csv(population_file+"_all_archive_normalized.csv", sep=",", index=False)

    df = pd.DataFrame(archive_default, columns = ['seed_set', 'spread', 'seed', 'communities', 'fairness', 'budget', 'time'])	
    df.to_csv(population_file+"_all_archive_default.csv", sep=",", index=False)