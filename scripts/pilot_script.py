from ast import Try
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ete3 import Tree
from math import floor
import utils as utl

def split_test_train(df, outer_col, inner_col, bins, train_split):

    df_sorted = df[df['selected'] == 'Test'].sort_values(outer_col)
    df_false_rows = df[df['selected'] == 'False']
    
    bins = min(len(df_sorted), bins)
    bin_size = len(df_sorted) // bins
    # number of rows to be selected from each bin
    k = max(1, int(bin_size * train_split))
    modified_indices = []

    for i in range(bins):
        start_index = i * bin_size
        end_index = start_index + bin_size if i != bins - 1 else None
        bin_df = df_sorted[start_index:end_index].sort_values(inner_col)       
        selected_rows = bin_df.iloc[::int(len(bin_df)/k)]  # select rows at a constant interval of k
        
        selected_indices = selected_rows.index
        modified_indices.extend(selected_indices)  # store indices of modified rows
        df_sorted.loc[selected_indices, 'selected'] = 'Train'  # update 'Train' rows

    # create a new DataFrame by merging the modified rows with the original rows
    modified_df = df_sorted.copy()
    modified_df = pd.concat([df_sorted, df_false_rows], ignore_index=True)

    return modified_df

def main(args):

    trees_dir = os.path.join(args.in_dir, 'Trees')
    model = args.model
    min_taxa = args.min_taxa
    max_polytomy = args.max_polytomy
    train_split = args.train_split
    bins = args.bins
    sim_num = args.simulation_num
    memcpu_ratio = args.memcpu_ratio
    req_multiplier = args.speed_multiplier
        
    print('input directory:', trees_dir)
    
    if args.reuse is not None:
        
        # load DataFrame from csv
        out_dir = os.path.join(args.out_dir, 'Results_' + args.reuse)
        taxa_df = pd.read_csv(os.path.join(out_dir, 'data_summary.csv'))
      
    else:
    
        # create the DataFrame
        out_dir = os.path.join(args.out_dir, 'Results_' + utl.get_time())
        taxa_list = []
    
        for root, dirs, files in os.walk(trees_dir):
        
            if utl.get_nth_parent(root, 1) not in ['genus', 'family']:
                continue

            for file in files:
                if file.endswith('.newick'):
                    tree_file = os.path.join(root, file)
                    counts_file = os.path.join(root, 'counts.fasta')
                    try:
                        utl.remove_root_polytomies(tree_file, counts_file)
                    except:
                        continue # skip bad tree files
    
                    tree = Tree(tree_file)
                    counts = utl.get_counts(counts_file)
                    taxa = utl.count_taxa(tree)
                    polytomy_ratio = utl.count_polytomy(tree, taxa)
                    variation = np.std(counts)
                    unique = np.unique(counts)
                    name = os.path.basename(root)
                    tree_length = utl.get_tree_length(tree)
                    
                    if 'family' in root.lower():
                        entry_type = 'family'
                    elif 'genus' in root.lower():
                        entry_type = 'genus'
                    else:
                        entry_type = ''
    
                    taxa_list.append( {
                        'name': name,
                        'type': entry_type,
                        'taxa': taxa,
                        'poly': polytomy_ratio,
                        'min': np.min(counts),
                        'max': np.max(counts),
                        'std': variation,
                        'mean': np.mean(counts),
                        'median': np.median(counts),
                        'anomaly': utl.anomaly_score(counts),
                        'symmetry': utl.symmetry_score(tree),
                        'colless': utl.colless_index(tree),
                        'diversity': len(unique)/taxa,
                        'count_stats': utl.get_stats(counts), # skewness, kurtosis, MAD, signal
                        'branch_stats': utl.get_stats(utl.get_branch_lengths(tree)),
                        'scaling': len(unique)/tree_length,
                        'MP': utl.fitch(tree_file, counts_file=counts_file, mlar_tree=False),
                        'path': root,
                        'selected': 'Test' if (polytomy_ratio <= max_polytomy and taxa >= min_taxa and variation > 0) else 'False'
                    } )
    
        # turn the dictionary into a DataFrame
        taxa_df = pd.DataFrame(taxa_list)
        taxa_df.reset_index(drop=True, inplace=True)
    
        # decide on the train set
        taxa_df = split_test_train(taxa_df, 'taxa', 'MP', bins, train_split)
    
        # save the DataFrame to a csv file
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        data_path = os.path.join(out_dir, 'data_summary.csv')
        taxa_df.to_csv(data_path, index=False)
    
        plt.hist(taxa_df['taxa'], bins=10, edgecolor='black') # set bins LATER
        plt.xlabel('Number of taxa')
        plt.ylabel('Frequency')
        plt.title('Distribution of taxa number')
        plt.show()

    # loop over the DataFrame based on the 'selected' column
    for index, row in taxa_df.iterrows():

        if str(row['selected'])[0] == 'T':

            req = floor(1+int(row['taxa'])/100)*req_multiplier
            
            utl.do_pilot_cmd(path=row['path'], name=row['name'], sim_num=sim_num, model=model, mode='Homogenous',\
               data=data_path, ncpu=req, mem=int(req*memcpu_ratio))
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='runs homogenenous and heterogenous inference on empirical data')
    parser.add_argument('--in_dir', '-i', type=str, default=os.getcwd(), help='path to trees with chromosome numbers')
    parser.add_argument('--out_dir', '-o', type=str, default=os.getcwd(), help='csv output directory')
    parser.add_argument('--model', '-m', type=str, default='g+C_l+C_du+E_de+C_b+C', help='adequate model')
    parser.add_argument('--min_taxa', '-n', type=int, default=50, help='minimum number of taxa threshold')
    parser.add_argument('--max_polytomy', '-p', type=float, default=0.1, help='maximum polytomy portion threshold')
    parser.add_argument('--train_split', '-t', type=float, default=0.5, help='portion of trees for training set')
    parser.add_argument('--bins', '-b', type=int, default=5, help='interval for selecting trees from the distribution')
    parser.add_argument('--simulation_num', '-s', type=int, default=10, help='number of simulations')
    parser.add_argument('--memcpu_ratio', type=float, default=1.5, help='ratio between mem in gb and #cores to ask for the heterogenous model; def 1.5')
    parser.add_argument('--speed_multiplier', type=int, default=6, help='speed multiplier for the heterogenous model; def 6')
    parser.add_argument('--reuse', type=str, default=None, help='used to load df from dir with provided date in name')

    main(parser.parse_args())
