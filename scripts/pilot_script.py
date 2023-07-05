import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
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

def taxa_to_list(trees_dir, min_taxa, max_polytomy):

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

                skewness, kurtosis, MAD = utl.get_stats(counts) 
                # branch_stats = utl.get_stats(utl.get_branch_lengths(tree))

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
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'MAD': MAD,
                    'scaling': len(unique)/tree_length,
                    'MP': utl.fitch(tree_file, counts_file=counts_file, mlar_tree=False),
                    'path': root,
                    'selected': 'Test' if (polytomy_ratio <= max_polytomy and taxa >= min_taxa and variation > 0) else 'False'
                } )
    return taxa_list

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

    out_dir = os.path.join(args.out_dir, 'Results_' + (args.reuse if args.reuse else utl.get_time()))
    data_path = os.path.join(out_dir, 'data_summary.csv')

    if args.reuse:
        # load DataFrame from csv
        taxa_df = pd.read_csv(data_path)
    else:
        # create the DataFrame
        taxa_df = pd.DataFrame(taxa_to_list(trees_dir, model, min_taxa, max_polytomy, train_split, bins))
        taxa_df.reset_index(drop=True, inplace=True)
        
        # determine the training set
        taxa_df = split_test_train(taxa_df, 'taxa', 'MP', bins, train_split)

        # save the DataFrame to a csv file
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        taxa_df.to_csv(data_path, index=False)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax = sns.histplot(data=taxa_df, x='taxa', bins=bins, hue='selected', multiple='stack', palette=colors)
    ax.set_yscale('log')
    plt.xlabel('Number of taxa')
    plt.ylabel('Frequency')
    plt.title('Distribution of taxa number')
    plt.savefig(os.path.join(out_dir, 'taxa_distribution.png'), dpi=400)
    plt.clf()

    # loop over the DataFrame based on the 'selected' column
    counter = 0
    for index, row in taxa_df.iterrows():
        
        if str(row['selected'])[0] == 'T':
        
            if counter >= args.iters:
                break
            counter += 1

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
    parser.add_argument('--train_split', '-t', type=float, default=0.6, help='portion of trees for training set')
    parser.add_argument('--bins', '-b', type=int, default=15, help='interval for selecting trees from the distribution')
    parser.add_argument('--simulation_num', '-s', type=int, default=100, help='number of simulations')
    parser.add_argument('--memcpu_ratio', type=float, default=2, help='ratio between mem in gb and #cores to ask for the heterogenous model; def 1.5')
    parser.add_argument('--speed_multiplier', type=int, default=6, help='speed multiplier for the heterogenous model; def 6')
    parser.add_argument('--reuse', type=str, default=None, help='used to load df from dir with provided date in name')
    parser.add_argument('--iters', type=int, default=10, help='limits the number of jobs sent at once')

    main(parser.parse_args())