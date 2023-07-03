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
    out_dir = os.path.join(args.out_dir, 'Results_' + utl.get_time())
    model = args.model
    min_taxa = args.min_taxa
    max_polytomy = args.max_polytomy
    train_split = args.train_split
    bins = args.bins
    sim_num = args.simulation_num
    memcpu_ratio = args.memcpu_ratio
    req_multiplier = args.speed_multiplier

    # create an empty DataFrame
    taxa_df = pd.DataFrame(columns=["name", "type", "taxa", "poly", "min", "max", "std", "mean", "median", "unique", "scaling", "MP", "path", "selected"])
    taxa_list = []
    
    print('input directory:', trees_dir)

    for root, dirs, files in os.walk(trees_dir):
        for file in files:
            if file.endswith('.newick'):
                tree_file = os.path.join(root, file)
                counts_file = os.path.join(root, 'counts.fasta')
                utl.remove_root_polytomies(tree_file, counts_file)

                tree = Tree(tree_file)
                counts = utl.get_counts(counts_file)
                taxa = utl.count_taxa(tree)
                polytomy_ratio = utl.count_polytomy(tree, taxa)
                variation = np.std(counts)
                unique = np.unique(counts)
                name = os.path.basename(root)
                
                if "family" in root.lower():
                    entry_type = "family"
                elif "genus" in root.lower():
                    entry_type = "genus"
                else:
                    entry_type = ""

                taxa_list.append( {
                    "name": name,
                    "type": entry_type,
                    "taxa": taxa,
                    "poly": polytomy_ratio,
                    "min": np.min(counts),
                    "max": np.max(counts),
                    "std": variation,
                    "mean": np.mean(counts),
                    "median": np.median(counts),
                    "unique": np.unique(counts),
                    "scaling": len(unique)/utl.get_tree_length(tree),
                    "MP": utl.fitch(tree_file, counts_file=counts_file, mlar_tree=False),
                    "path": root,
                    "selected": "Test" if (polytomy_ratio <= max_polytomy and taxa >= min_taxa and variation > 0) else "False"
                } )

                '''
                mlar_tree = utl.get_mlar_tree(tree, counts_file)

                total_length, average_length = utl.get_tree_lengths(tree)

                _, _, transitions, total_nodes = utl.count_transitions(mlar_tree)
                cum_diff = utl.cumulative_diff(mlar_tree)

                variability = np.std(counts) / np.mean(counts)
                variance = np.var(counts)
                diversity = np.unique(counts) / taxa
                proportional_changes = transitions / total_nodes
                ratiiiooo = cum_diff / total_length

                '''

    # concatenate the dictionaries into a DataFrame
    taxa_df = pd.concat([taxa_df, pd.DataFrame(taxa_list)])
    taxa_df.reset_index(drop=True, inplace=True)

    # decide on the train set
    taxa_df = split_test_train(taxa_df, 'taxa', 'MP', bins, train_split)

    # save the DataFrame to a csv file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    taxa_df.to_csv(os.path.join(out_dir, 'data_summary.csv'), index=False)

    plt.hist(taxa_df['taxa'], bins=10, edgecolor='black') # set bins LATER
    plt.xlabel('Number of taxa')
    plt.ylabel('Frequency')
    plt.title('Distribution of taxa number')
    plt.show()

    # loop over the DataFrame based on the 'selected' column
    for index, row in taxa_df.iterrows():

        if str(row['selected'])[0] == 'T':

            req = floor(1+int(row['taxa'])/100)*req_multiplier
            
            cmd = utl.get_cmd(path=row['path'], name=row['name'], sim_num=sim_num, model=model, mode='Homogenous')

            utl.do_job(path=row['path'], name=row['name'], ncpu=req, mem=int(req*memcpu_ratio), cmd=cmd)
            
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

    main(parser.parse_args())
