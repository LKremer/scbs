import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funk_svd import SVD
import umap
import matplotlib.patches as mpatches
import seaborn as sns



def read_files(data_path, metadata_path, CpG_cutoff, percentage, value, SVD_tuple, region_number, scaling):
    global metadata
    metadata = pd.read_csv(metadata_path, index_col = 0)
    #global data
    data = pd.read_csv(data_path, sep = ',', dtype={"chromosome": "string", 'start':'string', 'end':'string'})
    data['region'] = (data['chromosome']+':'+ data['start']+'-'+data['end']).apply(str)
    cell_nr = len(data.groupby('cell_name'))
    data_filtered = filter_data(data, CpG_cutoff, cell_nr, percentage, value, SVD_tuple, region_number, scaling)
    return data_filtered


def filter_data(data, CpG_cutoff, cell_nr, percentage, value, SVD_tuple, region_number, scaling):
    data_filtered = data[(data['n_sites'] >= CpG_cutoff) & (data['n_cells'] >= (cell_nr*percentage))]
    top_regions = data_filtered.groupby('region').first().nlargest(region_number, 'n_cells').index.tolist()
    data_filtered = data_filtered[data_filtered['region'].isin(top_regions)]
    if value == 'm_value':
        data_filtered['m_value'] = np.log2((data_filtered['meth_frac']+0.01)/((1-data_filtered['meth_frac'])+0.01))
    if scaling == True:
        data_filtered['mean'] = data_filtered.groupby('cell_name')[value].transform('mean')
        data_filtered['stdev'] = data_filtered.groupby('cell_name')[value].transform('std')
        data_filtered[value] = (data_filtered[value]-data_filtered['mean'])/data_filtered['stdev']
    data_filtered = data_filtered[['region','cell_name',value]].rename(columns = {'region':'i_id', 'cell_name':'u_id', value:'rating'})
    return data_filtered


def SVD_function(data_filtered, SVD_tuple):
    max_data = data_filtered['rating'].max()
    min_data = data_filtered['rating'].min()
    svd = SVD(lr = SVD_tuple[0], n_epochs=SVD_tuple[1], n_factors=SVD_tuple[2], early_stopping=True,shuffle=False, min_rating=min_data, max_rating=max_data)
    svd.fit(X=data_filtered, X_val = None)
    #global svd_matrix
    svd_matrix = pd.DataFrame(svd.pu_)
    svd_matrix['Sample'] = svd.user_mapping_.keys()
    svd_matrix = pd.merge(svd_matrix,metadata[['Sample','Neuron type']],on='Sample', how='left')
    return svd_matrix

def UMAP(svd_matrix, species, savename):
    reducer = umap.UMAP(n_neighbors = 20, min_dist = 0.4)
    #global embedding_list
    #global embedding
    embedding_list = pd.DataFrame(columns = [0, 1, 'Sample', 'Neuron type', 'Run'])
    for i in range(20):
        embedding = pd.DataFrame(reducer.fit_transform(svd_matrix.iloc[:,0:15]))
        embedding[['Sample', 'Neuron type']] = svd_matrix[['Sample', 'Neuron type']]
        embedding['Run'] = str(i)
        embedding_list = embedding_list.append(embedding)
    embedding_list.to_csv(str(savename + '.csv'))
    if species == 'human':
        excit_list = ['hDL-1', 'hDL-2', 'hDL-3','hL2/3','hL4', 'hL5-1', 'hL5-2', 'hL5-3', 'hL5-4','hL6-1', 'hL6-2', 'hL6-3']
        inhib_list = ['hNdnf', 'hNos', 'hVip-1', 'hVip-2', 'hPv-1','hPv-2', 'hSst-1', 'hSst-2', 'hSst-3']
    elif species == 'mouse':
        excit_list = ['mDL-1', 'mDL-2', 'mDL-3','mL2/3','mL4', 'mL5-1', 'mL5-2', 'mL6-1', 'mL6-2', 'mIn-1']
        inhib_list = ['mNdnf-1', 'mNdnf-2', 'mVip', 'mPv', 'mSst-1', 'mSst-2']
    excit = embedding[embedding['Neuron type'].isin(excit_list)]
    inhib = embedding[embedding['Neuron type'].isin(inhib_list)]
    
    fig, ax = plt.subplots(1, sharex = True, sharey = True, figsize = (7,7))
    sns.scatterplot(data = excit, x = (excit[0]), y = (excit[1]), hue = 'Neuron type', palette = ('Dark2'), hue_order = excit_list)
    sns.scatterplot(data = inhib, x = (inhib[0]), y = (inhib[1]), hue = 'Neuron type', palette = ('Set3'), hue_order = inhib_list)
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.setp(ax, xlabel='UMAP 1')
    plt.setp(ax, ylabel='UMAP 2')
    plt.savefig(str(savename +'.svg'), format="svg", bbox_extra_artists=(lgd,), bbox_inches='tight')



def main(file, regions, value, scaling, metadata, species, save):
    metadata_path = metadata
    data_path = file
    value = value
    if scaling == 1:
        scaling = True
    elif scaling == 0:
        scaling = False
    CpG_cutoff = 4
    percentage = 0.1
    region_number = regions
    SVD_tuple = (0.0005, 1000, 15)
    species = species
    # '100kb' or 'scan'
    #region = '100kb'
    save_path = save
    save_name = (data_path.split('/')[-1][:-4])+str(CpG_cutoff) + '_' + str(percentage).replace('.','-') + '_'+ str(region_number)+ '_'+value+'_'+str(scaling)+'_'+'20UMAPembedding'
    save = save_path + save_name
    print(save)
    print(species)

    data_filtered = read_files(data_path, metadata_path, CpG_cutoff, percentage, value, SVD_tuple, region_number, scaling)
    svd_matrix = SVD_function(data_filtered, SVD_tuple)
    UMAP(svd_matrix, species, save)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This is an argparse demo program')
    parser.add_argument('-f', '--file', help='Add path to file. Should be output of matrix script in scbs package & and end with .csv', 
                        default = '/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/mouse/regions/100kb/matrix_100kb_may3_test.csv')
    parser.add_argument('-r', '--regions', help='Number of Regions that should be kept for evaluation', type = int, 
                        default = 10000)
    parser.add_argument('-v', '--value', choices = ('shrunken_residual', 'm_value', 'meth_frac'), help='Value that should be analayzed. Possibilities: \'m_value\', \'meth_frac\',\'shrunken_residual\'',
                        default = 'm_value')
    parser.add_argument('-s', '--scaling', choices = (1, 0), help='Integer to specify whether values should be centered and scaled. Possibilities: 1, O', type = int,
                        default=0)
    parser.add_argument('-m', '--metadata', help='Add path to metadata file. Should end with .csv',
                        default='/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/metadata/metadata_format_mouse.csv')
    parser.add_argument('-sp', '--species', choices = ('human', 'mouse'), help='Specify species (mouse/human)',
                        default='mouse')
    parser.add_argument('-sa', '--save', help='Specify path to save svg and umap.csv',
                        default='/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/mouse/regions/100kb/')
    args = parser.parse_args()
    main(**vars(args))

