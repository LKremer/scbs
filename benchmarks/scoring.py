import argparse
import pandas as pd
import numpy as np
import math
import umap

def score_dc_s(files, bins, metadata):
	metadata = pd.read_csv(metadata, index_col = 0)
	score_dict = {}

	for file in files:
		DSC_list = []
		DC_list = []
		print('Calculating score for file ' + file)
		data = pd.read_csv(file, index_col = 0)
		data = data.merge(metadata[['Neuron type', 'Sample']] , how = 'left', right_on = metadata['Sample'], left_on = data.index)
		for i in range(20):
			UMAP1 = 'UMAP' + str((i*2)+1)
			UMAP2 = 'UMAP' + str((i*2)+2)
			data_subs = data.loc[:,[UMAP1, UMAP2, 'Neuron type']]
			data_subs.loc[:,'CD'] = np.nan
			xmeandf = data_subs.groupby('Neuron type')[UMAP1].mean().to_frame(name = 'x')
			ymeandf = data_subs.groupby('Neuron type')[UMAP2].mean().to_frame(name = 'y')
			centroids  = xmeandf.merge(ymeandf, on = 'Neuron type')
			types = centroids.index.unique()
			for idx, value in enumerate(types):
				subset = data_subs[data_subs['Neuron type'] == value]
				for index, row in subset.iterrows():
					dist = []
					for index_c, row_c in centroids.iterrows():
						dist.append((np.sqrt(((row[UMAP1]-row_c['x'])**2) + ((row[UMAP2]-row_c['y'])**2))))
					minimum = dist.index(min(dist))
					if minimum == idx:
						data_subs.at[index, 'CD']=1
					else:
						data_subs.at[index, 'CD']=0
			Ci = data_subs.groupby('Neuron type')['CD'].sum() / data_subs.groupby('Neuron type')['CD'].count()
			DSC = Ci.sum()/len(Ci)
			DSC_list.append(DSC)
    		#DC score
			H, x_edges, y_edges = np.histogram2d(data_subs[UMAP1], data_subs[UMAP2], range = [[-10, 30], [-20, 30]], bins = [bins, bins])
			grouped = dict(tuple(data_subs.groupby('Neuron type')))
			region_array = np.zeros((bins,bins))
			for i in grouped:
				cat_data, edge_x, edge_y  = np.histogram2d(grouped[i][UMAP1], grouped[i][UMAP2],range = [[-10, 30], [-20, 30]], bins = [bins, bins])
				with np.errstate(invalid = 'ignore'):
					fractions = np.divide(cat_data, H)
				with np.errstate(divide = 'ignore'):
					region_array += np.multiply(np.nan_to_num(np.log2(fractions), nan = 0, posinf = 0, neginf = 0),np.nan_to_num(cat_data))
			entropy = np.multiply(region_array, -1)

			nr_cells = len(data_subs)
			nr_celltypes = len(types)
			Z = (nr_cells*np.log2(nr_celltypes))
			DC = 1-(entropy.sum()/Z)
			DC_list.append(DC)
		mean_list = (np.array(DC_list) +np.array(DSC_list))/2  
		score_dict[file] = [mean_list, DSC_list, DC_list]
	return score_dict


def score_rel_distance(files, bins, metadata):
	metadata = pd.read_csv(metadata, index_col = 0)
	score_dict = {}
	for file in files:
    #file = mypath+file
		data = pd.read_csv(file, index_col = 0)
		data = data.merge(metadata[['Neuron type', 'Sample']] , how = 'left', right_on = metadata['Sample'], left_on = data.index)
		total_list = []
		print('Calculating score for file ' + file)
		pcs = data.iloc[:,41:57]
		centroids = pcs.groupby('Neuron type').mean()
		types = centroids.index.unique()
		Ci_array = np.ones(len(types))
		for idx1, value1 in enumerate(types):
			subset = pcs[pcs['Neuron type'] == value1].iloc[:,:-1].to_numpy()
			cells = len(subset)
			dist_array = np.zeros(shape = (cells, len(types)))  
			for idx2, value2 in enumerate(types):
				centroid = centroids[centroids.index == value2].to_numpy()
				distances = np.power(np.subtract(subset, centroid), 2)
				summed_sqrt = np.sqrt(np.sum(distances, axis = 1))
				dist_array[:,idx2] = summed_sqrt
			min_distance = np.argmin(dist_array, axis = 1)
			index = np.argpartition(dist_array, 2, axis = 1)
			order_array = np.take_along_axis(dist_array,index, axis = 1)
			false_centroids = np.where(order_array[:,0]!= dist_array[:,idx1])
			order_array[:,1][false_centroids] = order_array[:,0][false_centroids]
			total = 1-(dist_array[:,idx1] / (dist_array[:,idx1] + order_array[:,1]))
			total_list.append(total)    
		RDS = np.concatenate(total_list, axis=0).sum()/len(data)
		score_dict[file] = [RDS]
	return score_dict

def nearest_neighbor_count(files, bins, metadata, cutoff, k):
	metadata = pd.read_csv(metadata, index_col = 0)
	# for easier subsetting: create list with PC1 - PC15 (column names of dataframe)
	PCs = []
	for i in range(15):
		PCs.append(str('PC' + str(i+1)))
	score_dict = {}
	for file in files:
		percentage_count_list = []
		data = pd.read_csv(file, index_col = 0) # read in data
    
		# merge data with metadata information about neuron types
		data = data.merge(metadata[['Neuron type', 'Sample']] , how = 'left', right_on = metadata['Sample'], left_on = data.index)
		types = data['Neuron type'].unique() #list of Neuron types 
		# calculate the nearest neighbor information from the PC coordinates (15-dimensional)
		nearest_neighbors = umap.umap_.nearest_neighbors(data[PCs], k+1, metric = 'euclidean', metric_kwds = [], angular = False, random_state = None)
		#take our first entry as first entry is always own cell
		indices = nearest_neighbors[0][:,1:]
		distances = nearest_neighbors[1][:,1:]

		for type in types:
			# Get list of indices from dataset that all have same Neuron type
			all_type_indices = data.index[data['Neuron type'] == type].tolist()
			# Get subset of nearest neighbor indices that all have same Neuron type
			neighbor_type_indices = indices[all_type_indices]
			# Count per cell how many of the nearest neighbors are from own type (=in all_type_indices list) and devide by k (nr. of nearest neighbors)
			percentage = np.divide(np.isin(neighbor_type_indices, all_type_indices).sum(axis = 1),k)
			# count cells that have >= the percentage cutoff of nearest neighbors in their own class
			percentage_count = sum(percentage >= cutoff)
		
			percentage_count_list.append(percentage_count)
		relative_percentage_count = np.hstack((percentage_count_list)).sum()/len(data)
		score_dict[file] = relative_percentage_count
	return score_dict


def process(score_dict, scoring):
	print('Scores were calculated & are processed into the end format.')
	df=pd.DataFrame.from_dict(score_dict,orient='index')
	if scoring.lower() == 'dc+s':
		df.loc[:,'mean'] = np.mean(df[0].tolist(), axis=1)
		df.loc[:,'std'] = np.std(df[0].tolist(), axis=1)
		df.loc[:,'mean_DSC'] = np.mean(df[1].tolist(), axis=1)
		df.loc[:,'std_DSC'] = np.std(df[1].tolist(), axis=1)
		df.loc[:,'mean_DC'] = np.mean(df[2].tolist(), axis=1)
		df.loc[:,'std_DC'] = np.std(df[2].tolist(), axis=1)
		dft = df.iloc[:,3:]
	else:
		dft = df
	regions = []
	value = []
	scaling = []
	data_list = []
	imputing = []
	for i in dft.index:
		i = i.replace('m_value', 'm value(methylation fraction)')
		i = i.replace('shrunken_residual', 'shrunken residual')
		i = i.replace('meth_frac', 'methylation fraction')
		i = i.replace('feature', 'scaled by feature')
		i = i.replace('cell', 'scaled by cell')
		i = i.replace('none', 'non-scaled')
		i = i.replace('100kb', '100 kb Regions')
		i = i.replace('var', 'Variable Methylation Regions') 
		data_list.append(i.split('/')[-1].split('_')[-2])
		value.append(i.split('/')[-1].split('_')[-4])
		scaling.append(i.split('/')[-1].split('_')[-3])
		regions.append(i.split('/')[-1].split('_')[0])
		if (i.split('/')[-1].split('_')[-1]) == '20PCAUMAP.csv':
			imputing.append('PCA')
		else:
			imputing.append('Funk SVD')
	dft.loc[:,'regions'] = regions
	dft.loc[:,'value'] = value
	dft.loc[:,'scaling'] = scaling
	dft.loc[:,'data'] = data_list
	dft.loc[:,'imputing'] = imputing

	return dft


def main(files, bins, save, metadata, scoring, k, cutoff):
    #Bins for DC score
	print('Started reading in all files.')
	x_bins = bins
	y_bins = bins
	#First check from which script file resulted
	files = [f.name for f in files if f.name.endswith('.csv')]
	score_dict = {}
	#Score all files (each file should consist of 20 UMAP scores, so the dictionaty should contain as many scores per file)
	if scoring.lower() == 'dc+s':
		score_dict = score_dc_s(files, bins, metadata)
	elif scoring.lower() == 'rel':
		score_dict = score_rel_distance(files, bins, metadata)
	elif scoring.lower() == 'nnc':
		score_dict = nearest_neighbor_count(files, bins, metadata, cutoff, k)
	else:
		raise Exception("Specify which scoring type should be used. Check --help for possible options.")
	#print(score_dict)
	# Assign Dict to condition and calculate mean value for each
	score_df = process(score_dict, scoring)
	score_df.to_csv(save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('-f', '--files', help='Add path to files.', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-b', '--bins', help='Number of bins grid should be devided in for DC score', type = int, 
                        default = 20)
    parser.add_argument('-s', '--save', help='Path & Name of output df')
    parser.add_argument('-m', '--metadata', help='Path to metadata', default='/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/metadata/metadata_format_mouse.csv')
    parser.add_argument('-sc', '--scoring', help = 'Which scoring type should be used? DC+S Score is the average of DC and DSC scores. \'rel\' refers to the relative distance score. \'nnc\' refers to the nearest neighbor count score.',
    					choices = ('DC+S', 'rel', 'nnc'), default = 'rel')
    parser.add_argument('-k', '--k', help='Nr of neatest neighbors for nearest neighbor count score', type = int, default = 20)
    parser.add_argument('-c', '--cutoff', help='Cutoff for percentage cutoff for nearest neighbors score', type = float, default = 0.9)	
    args = parser.parse_args()
    main(**vars(args))
