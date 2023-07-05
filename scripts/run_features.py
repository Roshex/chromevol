import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from ete3 import Tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
import utils as utl

def features_to_list(paths, sim_num, p_cutoff, string = r'AICc of the best model = ([\d.]+)'):
    # for LL use: string = r'Final optimized likelihood is: ([\d.]+)'
    new_features = []
    for path in paths:

        emp_dir, sim_dir, inf_dir = utl.concat_paths(path, 'Homogenous')
        mlar_tree = Tree(os.path.join(emp_dir, 'MLAncestralReconstruction.tree'))
        emp_counts = utl.get_counts(os.path.join(emp_dir, 'counts.fasta'))

        # get 10 subtrees
        subtrees = utl.get_m_subtrees(mlar_tree, m=10, n=utl.count_taxa(mlar_tree)//15)

        # calculate correlation features
        correlation, reg_coef, mean_interaction, ncd = utl.chromosome_branch_correlation(mlar_tree)
        std_ncd = np.std([utl.chromosome_branch_correlation(t)[3] for t in subtrees])

        # calculate rate features
        event_matrix, transitions = utl.read_event_matrix(emp_dir)
        trans_std = [utl.count_events(t, event_matrix) for t in subtrees]
        trans_std = {key: np.std([d[key] for d in trans_std]) for key in ['DYS', 'POL']}

        # extract empirical AICc
        empirical_aicc = utl.extract_score(emp_dir, string)

        # loop over sim_num folders in sim_dir
        for i in range(sim_num):
            sim_counts = utl.get_counts(os.path.join(os.path.join(sim_dir, i), 'counts.fasta'))

        # loop over sim_num folders in inf_dir
        homogenous_aiccs = heterogenous_aiccs = []
        for i in range(sim_num):
            homogenous_aiccs.append( utl.extract_score(os.path.join(inf_dir, f'{i}_Homogenous'), string) )
            heterogenous_aiccs.append( utl.extract_score(os.path.join(inf_dir, f'{i}_Heterogenous'), string) )

        # avergae a few randomly sampled AICcs
        choice = utl.choose_sims(sim_dir, sim_num, k=5, random=True)
        repr_dAICc = np.mean(heterogenous_aiccs[choice]-homogenous_aiccs[choice])

        # top cuttoff % deltaLL or deltaAICc
        dAICc_cutoff = np.percentile(np.array(heterogenous_aiccs)-np.array(homogenous_aiccs), 100-p_cutoff)

        new_features.append( {
            'path': path,
            'order_signal': utl.order_signal(mlar_tree),
            'comparisons': utl.count_comparisons(mlar_tree, utl.get_min_clade(emp_dir)),
            'entropy': np.std([utl.get_entropy(t) for t in subtrees]),
            'correlation': correlation,
            'mean_interaction': mean_interaction,
            'reg_coef': reg_coef,
            'norm_cum_diff': ncd,
            'std_cum_diff': std_ncd,
            'dys_transitions': transitions['DYS'],
            'poly_transitions': transitions['POL'],
            'dys_std': trans_std['DYS'],
            'poly_std': trans_std['POL'],
            'sim_feat1': None,
            'sim_feat2': None,
            'sim_feat3': None,
            'sim_feat4': None,
            'empirical_AICc': empirical_aicc,
            'repr_dAICc': repr_dAICc,
            'dAICc_cutoff': dAICc_cutoff
        } )
    return new_features

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

def plot_params_evolution(best_params_list, mse_list):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, param in enumerate(['learning_rate', 'max_depth', 'lambda', 'gamma', 'n_estimators']):
        param_values = [params[param] for params in best_params_list]
        axes[i].plot(param_values, mse_list)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('mse')

    plt.show()

def main(args):

    taxa_df = pd.read_csv(args.data_path)
    new_features = features_to_list(taxa_df['path'], args.num_of_simulations, args.p_cutoff)

    new_df = pd.DataFrame(new_features)
    new_df.reset_index(drop=True, inplace=True)
    taxa_df = taxa_df.merge(new_df, on='path', how='outer')

    '''

    Tasks:

    1. how to normalize the AICc values? (min-max, z-score, etc.)
    2. *AICc* or LL - LL is more sensitive to the number of taxa, so worse for the model
    3. email Anat about bug
    4. ask Keren about Astraceae
    5. should we do PCA on all the stat features to reduce their number?
    6. what to do about sim_feat1-4?

    '''

    # all 3 AICs cols should be norm by the same scaler!!! method - open for discussion
    # normalize the AICc target columns in the df using MinMaxScaler
    scaler = MinMaxScaler()
    taxa_df['norm_empirical_AICc'] = scaler.fit_transform(taxa_df[['empirical_AICc']])
    taxa_df['norm_repr_dAICc'] = scaler.fit_transform(taxa_df[['repr_dAICc']])
    taxa_df['norm_dAICc_cutoff'] = scaler.fit_transform(taxa_df[['dAICc_cutoff']])

    # save the DataFrame back to a csv file
    taxa_df.to_csv(args.data_path, index=False)

    # split the data into train and test sets
    feature_list = ['taxa', 'median', 'std', 'anomaly', 'symmetry', 'colless', 'diversity', 'skewness', 'kurdosis',
                    'scaling', 'MP', 'comparisons', 'order_signal', 'entropy', 'correlation', 'norm_cum_diff', 'std_cum_diff',
                    'dys_transitions', 'pol_transitions', 'pol_std', 'dys_std', 'norm_empirical_AICc', 'norm_repr_dAICc']
    target_name = 'norm_dAICc_cutoff'

    # train_features, test_features, train_target, test_target
    X_train, X_val, y_train, y_val = utl.split_data(taxa_df, feature_names=feature_list, target_name=target_name)

    model = xgb.XGBRegressor()

    iter_num = 3
    k = 15
    rfe = False
    space_frac = 0.6
    best_params = None

    selected_features = []
    best_params_list = []
    mse_list = []

    for _ in range(iter_num):

        # feature selection
        X_train_selected, selector = utl.select_features(model, X_train, y_train, k=k, use_rfe=rfe)
        # hyperparameter tuning
        best_params, _ = utl.tune_hyperparams(model, X_train_selected, y_train, \
            space_frac=space_frac, best_params=best_params, bayesian_opt=True)
        # model evaluation
        mse = utl.evaluate_model(X_train, X_val, y_train, y_val, selector, best_params)
    
        # store results for this iteration
        selected_features.append(X_train.columns[selector.get_support()])
        best_params_list.append(best_params)
        mse_list.append(mse)

        # reduce constants for the next iteration
        space_frac *= 0.8
        k -= 5
        if k < 10:
            rfe = True

    # final model selection
    best_iteration = np.argmax(mse_list)
    final_features = selected_features[best_iteration]
    final_best_params = best_params_list[best_iteration]
    print(f'Best features: {final_features}', f'Best parameters: {final_best_params}', sep='\n')

    # retrain final model using all training data and selected features
    final_model = xgb.XGBRegressor(**final_best_params)
    final_model.fit(X_train_selected, y_train, num_boost_rounds=1000, early_stopping_rounds=10, \
       eval_set=[(X_train_selected, y_train), (X_val_selected, y_val)], verbose=False)

    # final evaluation on the validation set
    X_val_selected = selector.transform(X_val)
    y_pred = final_model.predict(X_val_selected)
    final_mse = mean_squared_error(y_val, y_pred)
    print(f'Final RMSE: {final_mse**0.5}')

    plot_learning_curve(final_model, 1000)
    plot_scatter_plot(y_val, y_pred)
    plot_histogram(y_pred)
    plot_confusion_matrix(y_val, y_pred)
    plot_params_evolution(best_params_list, mse_list)
    xgb.plot_tree(final_model, rankdir='LR')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate additional features for the model')
    parser.add_argument('--data_path', '-d', type=str, help='path to csv file summary')
    parser.add_argument('--num_of_simulations', '-n', type=int, help='number of simulations')
    parser.add_argument('--p_cutoff', '-p', type=float, default=5, help='percent cutoff applied to deltaLL')

    main(parser.parse_args())
