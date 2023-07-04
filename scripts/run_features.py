import os
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ete3 import Tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import utils as utl

def select_n_clades(tree, n, m):
    # breadth-first search (BFS) algorithm to select n clades from a phylogenetic tree
    selected_clades = []
    queue = [tree]

    while queue and len(selected_clades) < n:
        node = queue.pop(0)
        
        if not node.is_leaf() and len(node.get_leaves()) > m:
            selected_clades.append(node)
        
        queue.extend(node.children)
    
    return selected_clades

def plot_learning_curve(model, num_rounds):
    eval_results = model.eval_results()
    train_error = eval_results['train']['rmse']
    val_error = eval_results['test']['rmse']
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_rounds + 1), train_error, label='Training Error')
    plt.plot(range(1, num_rounds + 1), val_error, label='Validation Error')
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

def plot_scatter_plot(test_target, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(test_target, predictions)
    plt.xlabel('Actual Threshold')
    plt.ylabel('Predicted Threshold')
    plt.title('Scatter Plot of Predicted vs Actual Values')
    plt.show()

def plot_histogram(predictions):
    plt.figure(figsize=(8, 6))
    sns.histplot(predictions, bins=20)
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted Thresholds')
    plt.show()

def plot_confusion_matrix(test_target, predictions):
    # convert values to binary
    threshold_bin = (test_target >= 0.5).astype(int)
    predictions_bin = (predictions >= 0.5).astype(int)
    cm = confusion_matrix(threshold_bin, predictions_bin)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

def main(args):

    sim_num = args.num_of_simulations
    p_cutoff = args.p_cutoff

    new_features = []
    taxa_df = pd.read_csv(args.data_path)
    string = r'AICc of the best model = ([\d.]+)'
    # [r'AICc of the best model = ([\d.]+)', r'Final optimized likelihood is: ([\d.]+)']

    for path in taxa_df['path']:

        emp_dir, sim_dir, inf_dir = utl.concat_paths(path, 'Homogenous')
        mlar_tree = Tree(os.path.join(emp_dir, 'MLAncestralReconstruction.tree'))
        emp_counts = utl.get_counts(os.path.join(emp_dir, 'counts.fasta'))

        # get features using functions from utils.py
        gain_r, loss_r, dupl_r, demi_r = utl.count_transitions(mlar_tree)
        cum_diff, correlation, reg_coef, mean_interaction = utl.chromosome_branch_correlation(mlar_tree)
        cum_diff = cum_diff / utl.get_branch_lengths(mlar_tree)

        empirical_aicc = utl.extract_score(emp_dir, string)

        # loop over sim_num folders in sim_dir
        for i in range(sim_num):
            sim_counts = utl.get_counts(os.path.join(os.path.join(sim_dir, i), 'counts.fasta'))

        # loop over sim_num folders in inf_dir
        homogenous_aiccs = []
        heterogenous_aiccs = []
        for i in range(sim_num):
            homogenous_aiccs.append( utl.extract_score(os.path.join(inf_dir, f'{i}_Homogenous'), string) )
            heterogenous_aiccs.append( utl.extract_score(os.path.join(inf_dir, f'{i}_Heterogenous'), string) )

        # avergae a few randomly sampled AICcs
        choice = random.sample(range(sim_num), 5)
        avg_homogenous_aicc = np.mean(homogenous_aiccs[choice])
        avg_heterogenous_aicc = np.mean(heterogenous_aiccs[choice])

        # top cuttoff % deltaLL or deltaAICc
        dAICc_cutoff = np.percentile(np.array(heterogenous_aiccs)-np.array(homogenous_aiccs), 100-p_cutoff)

        new_features.append( {
            'path': path,
            'entropy': utl.get_entropy(mlar_tree),
            'gain_ratio': gain_r,
            'loss_ratio': loss_r,
            'dupl_ratio': dupl_r,
            'demi_ratio': demi_r,
            'norm_cum_diff': cum_diff,
            'correlation': correlation,
            'mean_interaction': mean_interaction,
            'empirical_AICc': empirical_aicc,
            'avg_homogenous_AICc': avg_homogenous_aicc,
            'avg_heterogenous_AICc': avg_heterogenous_aicc,
            'dAICc_cutoff': dAICc_cutoff
        } )

    new_df = pd.DataFrame(new_features)
    new_df.reset_index(drop=True, inplace=True)
    taxa_df = taxa_df.merge(new_df, on='path', how='outer')

    # normalize the AICc target columns in the df using MinMaxScaler
    scaler = MinMaxScaler()
    taxa_df['nrm_homogenous_AICc'] = scaler.fit_transform(taxa_df[['avg_homogenous_AICc']])
    taxa_df['nrm_heterogenous_AICc'] = scaler.fit_transform(taxa_df[['avg_heterogenous_AICc']])

    # save the DataFrame back to a csv file
    taxa_df.to_csv(args.data_path, index=False)

    # split the data into train and test sets
    feature_list = ['taxa', 'min', 'max', 'std', 'anomaly', 'symmetry', 'colless', 'diversity', 'scaling', 'MP', 'entropy',
                        'gain_ratio', 'loss_ratio', 'dupl_ratio', 'demi_ratio', 'norm_cum_diff', 'correlation',
                        'mean_interaction', 'empirical_AICc', 'nrm_homogenous_AICc', 'nrm_heterogenous_AICc']
    target_name = 'dAICc_cutoff'
    train_features, test_features, train_target, test_target = split_data(taxa_df, feature_names=feature_list, target_name='dAICc_cutoff')

    num_rounds = 100
    num_features = 10
    model = train_xgboost_model(train_features, train_target, num_rounds, num_features)

    predictions = evaluate_model(model, test_features, test_target)

    plot_learning_curve(model, num_rounds)
    plot_scatter_plot(test_target, predictions)
    plot_histogram(predictions)
    plot_confusion_matrix(test_target, predictions)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate additional features for the model')
    #parser.add_argument('--emp_dir', '-e', type=str, help='path to empirical results')
    #parser.add_argument('--sim_dir', '-s', type=str, help='path to simulation results')
    #parser.add_argument('--inf_dir', '-i', type=str, help='path to inference results')
    parser.add_argument('--data_path', '-d', type=str, help='path to csv file summary')
    parser.add_argument('--num_of_simulations', '-n', type=int, help='number of simulations')
    parser.add_argument('--p_cutoff', '-p', type=float, default=5, help='percent cutoff applied to deltaLL')

    main(parser.parse_args())





'''

def normalize_aicc(aicc_values):
    """
    Normalize AICc values to the range [0, 1]
    :param aicc_values: List or array of AICc values
    :return: Normalized AICc values
    """
    min_aicc = min(aicc_values)
    max_aicc = max(aicc_values)

    normalized_aicc = (aicc_values - min_aicc) / (max_aicc - min_aicc)

    return normalized_aicc


def count_node_comparisons(tree):
    #node.get_sisters())
    pass



def calculate_delta_log_likelihood(tree, simulated_trees):
    delta_ll_stats = []
    # Perform calculations for each simulation
    for simulated_tree in simulated_trees:
        delta_ll = simulated_tree.log_likelihood() - tree.log_likelihood()
        delta_ll_stats.append(delta_ll)
    return delta_ll_stats


def calculate_event_sum_over_branch_length(tree):
    event_sum = 0
    branch_length_sum = 0.0
    clade_count = 0
    for node in tree.traverse("preorder"):
        if not node.is_leaf():
            branch_length_sum += node.dist
            event_sum += count_events(node)  # Custom function to count events for a given node
            clade_count += 1
            if clade_count == 10:
                break
    if branch_length_sum > 0:
        event_sum_over_length = event_sum / branch_length_sum
    else:
        event_sum_over_length = 0.0
    return event_sum_over_length




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, aicc_cutoff_labels, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use the trained model to predict AICc cutoff
predictions = model.predict(X_test)

# Calculate the AICc based on the predictions
def calculate_aicc_cutoff(predictions, y_true, num_features):
    n = len(y_true)
    residual_error = mean_squared_error(y_true, predictions) * n
    num_params = num_features + 1  # Number of features + 1 (intercept term)
    aicc = n * np.log(residual_error / n) + 2 * num_params + (2 * num_params * (num_params + 1)) / (n - num_params - 1)
    return aicc

# Calculate the AICc using the predictions
aicc_cutoff = calculate_aicc_cutoff(predictions, y_test, len(features[0]))

# Print the AICc cutoff
print("AICc Cutoff:", aicc_cutoff)



'''