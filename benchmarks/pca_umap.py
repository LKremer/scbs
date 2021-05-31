import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from umap import UMAP

def read_files(data_path, CpG_cutoff, percentage, value, region_list, scaling, savepath, region_type, iterations, pc):
    print('Reading in and processing file ' + data_path)
    data = pd.read_csv(data_path, sep = ',', dtype={"chromosome": "string", 'start':'string', 'end':'string'})
    data['region'] = (data['chromosome']+':'+ data['start']+'-'+data['end']).apply(str)
    cell_nr = len(data.groupby('cell_name'))
    #Basic Quality Filter: 
    # 1. We require to have regions that have at least the number of CpGs as the CpG_cutoff
    # 2. One region needs to be measured in at least a certain percentage of total cells (cell_nr*percentage)
    data = data[(data['n_sites'] >= CpG_cutoff) & (data['n_cells'] >= (cell_nr*percentage))]
    filter_data(data, value, region_list, scaling, savepath, region_type, iterations, pc)



def filter_data(data, value, region_list, scaling, savepath, region_type, iterations, pc):
    #We only look at the top X regions (as defined in region_list) and filter out the rest    
    top_regions = data.groupby('region').first().nlargest(region_list[0], 'n_cells').index.tolist()
    data_filtered = data[data['region'].isin(top_regions)]
    if value == 'm_value':
        data_filtered['m_value'] = np.log2((data_filtered['meth_frac']+0.01)/((1-data_filtered['meth_frac'])+0.01))
    #If defined, data is either scaled and centered per cell. Later, it will automatically also be scaled per feature (=region)    
    if scaling == 'cell':
        data_filtered['mean'] = data_filtered.groupby('region')[value].transform('mean')
        data_filtered['stdev'] = data_filtered.groupby('region')[value].transform('std')
        data_filtered[value] = (data_filtered[value]-data_filtered['mean'])/data_filtered['stdev']

    data_filtered = data_filtered[['region','cell_name',value]]
    # Define name with which output csv will be saved
    save_name = str(region_list[0])+ '_'+value+'_'+str(scaling)+'_'+region_type+'_'+'20PCAUMAP'
    save = savepath + save_name
    data = reduce(data_filtered, value_column = value, n_pc = pc, n_iterations = iterations, n_neighbors = 20, min_dist = 0.4)
    data.to_csv(str(save+'.csv'))
    # If there are several region numbers in the list, the filtered input table is further shortened
    # and iterative PCA/UMAP are run again. 
    if len(region_list) > 1:
        for i in region_list[1:]:
            top_regions = top_regions[:i]
            data_filtered= data_filtered[data_filtered['region'].isin(top_regions)]
            save_name = str(i)+ '_'+value+'_'+str(scaling)+'_'+region_type+'_'+'20PCAUMAP'
            save = savepath + save_name
            data = reduce(data_filtered, value_column = value, n_pc = pc, n_iterations = iterations, n_neighbors = 20, min_dist = 0.4)
            data.to_csv(str(save+'.csv'))



def imputing_pca(
    X, n_components=10, n_iterations=10, scale_features=True, center_features=True
):
    # center and scale features
    X = scale(X, axis=0, with_mean=center_features, with_std=scale_features)
    # for each set of predicted values, we calculated how similar it is to the values
    # we predicted in the previous iteration, so that we can roughly see when our
    # prediction converges
    dist = np.full(n_iterations, fill_value=np.nan)
    # varexpl = np.full(n_iterations, fill_value=np.nan)
    nan_positions = np.isnan(X)
    X[nan_positions] = 0  # zero is our first guess for all missing values
    # start iterative imputation
    for i in range(n_iterations):
        print(f"PCA iteration {i + 1}...")
        previous_guess = X[nan_positions]  # what we imputed in the previous iteration
        # PCA on the imputed matrix
        pca = PCA(n_components=n_components)
        pca.fit(X)
        # impute missing values with PCA
        new_guess = (pca.inverse_transform(pca.transform(X)))[nan_positions]
        X[nan_positions] = new_guess
        # compare our new imputed values to the ones from the previous round
        dist[i] = np.mean((previous_guess - new_guess) ** 2)
        # varexpl[i] = np.sum(pca.explained_variance_ratio_)
    pca.prediction_dist_iter = dist  # distance between predicted values
    # pca.total_var_exp_iter = varexpl
    pca.X_imputed = X
    return pca


def reduce(
    matrix,  # path to a matrix produced by scbs matrix OR 
    value_column="shrunken_residual",  # the name of the column containing the methylation values
    n_pc=10,  # number of principal components to compute
    n_iterations=10,  # number of iterations for PCA imputation
    n_neighbors=20,  # a umap parameter
    min_dist=0.1,  # a umap parameter
):
    """
    Takes the output of 'scbs matrix' and reduces it to fewer dimensions, first by
    PCA and then by UMAP.
    """
    if isinstance(matrix, str):
        df = pd.read_csv(matrix, header=0)
    elif isinstance(matrix, pd.core.frame.DataFrame):
        df = matrix
    else:
        raise Exception("'matrix' must be either a path or a pandas DataFrame.")
    # make a proper matrix (cell x region)
    print("Converting long matrix to wide matrix...")
    df_wide = (
        df.loc[:, ["cell_name", "region", value_column]]
        .pivot(index="cell_name", columns="region", values=value_column)
    )
    X = np.array(df_wide)
    Xdim_old = X.shape
    na_frac_cell = np.sum(np.isnan(X), axis=1) / X.shape[1]
    cell_names = df_wide.index

    # run our modified PCA
    print("Running PCA...")
    pca = imputing_pca(X, n_components=n_pc, n_iterations=n_iterations)
    X_pca_reduced = pca.transform(pca.X_imputed)
    # define UMAP parameters
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    # Run UMAP once
    X_umap_reduced = reducer.fit_transform(X_pca_reduced)
    # UMAP is run 19 more times (because it differs every time) Every result is added to output table.
    for i in range(20-1):
        new = reducer.fit_transform(X_pca_reduced)
        X_umap_reduced = np.concatenate((X_umap_reduced, new), axis=1)
    data=pd.DataFrame(np.concatenate((X_umap_reduced, X_pca_reduced), axis=1))
    # Columns are named appropriatley. Both UMAP coordinates from each run and PCs from funk SVD are saved. 
    data.columns = ["UMAP" + str(i + 1) for i in range(40)] + [
            "PC" + str(i + 1) for i in range((X_pca_reduced.shape[1]))]
    data.index = cell_names
    
    return data 



def main(file, regions, region_type, value, scaling, save, iterations, pc):
    value = value
    CpG_cutoff = 4
    percentage = 0.1
    savepath = save 
    region_list = [f for f in regions]
    region_list.sort(reverse = True)
    read_files(file, CpG_cutoff, percentage, value, region_list, scaling, savepath, region_type, iterations, pc)
    #data = reduce(data_filtered, value, center_cells = False, min_obs_region = 0, min_obs_cell = 0, n_pc = 15, n_iterations = 15, n_neighbors = 20, min_dist = 0.4)


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
    parser.add_argument('-s', '--scaling', choices = ('none', 'cell'), help='Integer to specify if cells should be centered and scaled per cell in addition to per feature. Possibilities: \'none\', \'cell\'',
                        default='none')
    parser.add_argument('-sa', '--save', help='Specify path to save svg and umap.csv',
                        default='/mnt/o_drive/Leonie_K/20200407_neuro_subtypes/mouse/regions/100kb/')
    parser.add_argument('-i', '--iterations', help='Specify nr of PCA iterations', type = int,
                        default='15')
    parser.add_argument('-p', '--pc', help='Specify nr. of principal components to compute', type = int,
                        default='15')

    args = parser.parse_args()
    main(**vars(args))
