import argparse
import pandas as pd
import numpy as np
from funk_svd import SVD
import umap


def read_files(data_path, CpG_cutoff, percentage, value, SVD_tuple, region_list, scaling, savepath, region_type):
    print('Reading in and processing file ' + data_path)
    data = pd.read_csv(data_path, sep = ',', dtype={"chromosome": "string", 'start':'string', 'end':'string'})
    data['region'] = (data['chromosome']+':'+ data['start']+'-'+data['end']).apply(str)
    cell_nr = len(data.groupby('cell_name'))
    #Basic Quality Filter: 
    # 1. We require to have regions that have at least the number of CpGs as the CpG_cutoff
    # 2. One region needs to be measured in at least a certain percentage of total cells (cell_nr*percentage)
    data = data[(data['n_sites'] >= CpG_cutoff) & (data['n_cells'] >= (cell_nr*percentage))]
    filter_data(data, value, SVD_tuple, region_list, scaling, savepath, region_type)


def filter_data(data, value, SVD_tuple, region_list, scaling, savepath, region_type):
    #We only look at the top X regions (as defined in region_list) and filter out the rest
    top_regions = data.groupby('region').first().nlargest(region_list[0], 'n_cells').index.tolist()
    data_filtered = data[data['region'].isin(top_regions)]
    if value == 'm_value':
        data_filtered['m_value'] = np.log2((data_filtered['meth_frac']+0.01)/((1-data_filtered['meth_frac'])+0.01))
    #If defined, data is either scaled and centered per feature (=region) or cell.
    if scaling == 'feature':
        data_filtered['mean'] = data_filtered.groupby('cell_name')[value].transform('mean')
        data_filtered['stdev'] = data_filtered.groupby('cell_name')[value].transform('std')
        data_filtered[value] = (data_filtered[value]-data_filtered['mean'])/data_filtered['stdev']
    elif scaling == 'cell':
        data_filtered['mean'] = data_filtered.groupby('region')[value].transform('mean')
        data_filtered['stdev'] = data_filtered.groupby('region')[value].transform('std')
        data_filtered[value] = (data_filtered[value]-data_filtered['mean'])/data_filtered['stdev']
    # Columns need to be renamed for funk SVD input
    data_filtered = data_filtered[['region','cell_name',value]].rename(columns = {'region':'i_id', 'cell_name':'u_id', value:'rating'})
    # Define name with which output csv will be saved
    save_name = str(region_list[0])+ '_'+value+'_'+str(scaling)+'_'+region_type+'_'+'20UMAPembedding'
    save = savepath + save_name
    # Run funk SVD
    svd_matrix, cell_names = SVD_function(data_filtered, SVD_tuple)
    # Run UMAP (in UMAP function, output is saved)
    UMAP(svd_matrix, cell_names, save, SVD_tuple)
    # If there are several region numbers in the list, the filtered input table is further shortened
    # and funk SVD/UMAP are run again. 
    if len(region_list) > 1:
        for i in region_list[1:]:
            top_regions = top_regions[:i]
            data_filtered= data_filtered[data_filtered['i_id'].isin(top_regions)]
            save_name = str(i)+ '_'+value+'_'+str(scaling)+'_'+region_type+'_'+'20UMAPembedding'
            save = savepath + save_name
            svd_matrix, cell_names = SVD_function(data_filtered, SVD_tuple)
            UMAP(svd_matrix, cell_names, save, SVD_tuple)



def SVD_function(data_filtered, SVD_tuple):
    max_data = data_filtered['rating'].max()
    min_data = data_filtered['rating'].min()
    #define SVD parameters
    svd = SVD(lr = SVD_tuple[0], n_epochs=SVD_tuple[1], n_factors=SVD_tuple[2], early_stopping=True,shuffle=False, min_rating=min_data, max_rating=max_data)
    # run SVD
    svd.fit(X=data_filtered, X_val = None)
    svd_matrix = pd.DataFrame(svd.pu_)
    cell_names = svd.user_mapping_.keys()
    return svd_matrix, cell_names

def UMAP(svd_matrix, cell_names, savename, SVD_tuple):
    print('Running UMAPs...')
    # define UMAP parameters
    reducer = umap.UMAP(n_neighbors = 20, min_dist = 0.4)
    # Run UMAP once
    X_svd_reduced = reducer.fit_transform(svd_matrix)
    # UMAP is run 19 more times (because it differs every time) Every result is added to output table.
    for i in range(20-1):
        new = reducer.fit_transform(svd_matrix)
        X_svd_reduced = np.concatenate((X_svd_reduced, new), axis=1)
    data=pd.DataFrame(np.concatenate((X_svd_reduced, svd_matrix), axis=1))
    # Columns are named appropriatley. Both UMAP coordinates from each run and PCs from funk SVD are saved. 
    data.columns = ["UMAP" + str(i + 1) for i in range(40)] + ["PC" + str(i + 1) for i in range((SVD_tuple[2]))]
    data.index = cell_names
    data.index.name = 'cell_name'
    data.to_csv(str(savename + '.csv'))


def main(file, regions, region_type, value, scaling, save):
    data_path = file
    value = value
    CpG_cutoff = 4
    percentage = 0.1
    region_list = [f for f in regions]
    region_list.sort(reverse = True)
    print(region_list)
    SVD_tuple = (0.0005, 1000, 15)
    save_path = save
    print('Files will be saved in this folder: ' + save_path)
    read_files(data_path, CpG_cutoff, percentage, value, SVD_tuple, region_list, scaling, save_path, region_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('-f', '--file', help='Add path to file. Should be output of matrix script in scbs package & and end with .csv', 
                        default = '/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/mouse/regions/100kb/matrix_100kb_may3_test.csv')
    parser.add_argument('-r', '--regions', nargs = '+', help='List of number of Regions that should be kept for evaluation. For example: -r 1000 5000 8000', type = int, 
                        default = [25000, 10000, 5000, 500, 100])
    parser.add_argument('-rt', '--region_type', help = 'Specify which region type is used. Possibilities: \'100kb\', \'var\'', choices = ('100kb', 'var'))
    parser.add_argument('-v', '--value', choices = ('shrunken_residual', 'm_value', 'meth_frac'), help='Value that should be analayzed. Possibilities: \'m_value\', \'meth_frac\',\'shrunken_residual\'',
                        default = 'm_value')
    parser.add_argument('-s', '--scaling', choices = ('none', 'feature', 'cell'), help='Integer to specify whether values should be centered and scaled. Possibilities: \'none\', \'feature\', \'cell\'',
                        default='feature')
    parser.add_argument('-sa', '--save', help='Specify path to save output csv',
                        default='/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/mouse/regions/100kb/')
    args = parser.parse_args()
    main(**vars(args))

