'''
import torch
import torch.nn as nn
import torch.optim as optim

# Define your dataset
class PhyloChromoDataset(torch.utils.data.Dataset):
    def __init__(self, features, aicc_cutoff_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.aicc_cutoff_labels = torch.tensor(aicc_cutoff_labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.aicc_cutoff_labels[idx]

# Define your neural network model
class AICcModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AICcModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set hyperparameters
input_size = 7  # Number of features
hidden_size = 20
output_size = 1
learning_rate = 0.01
num_epochs = 100

# Prepare the dataset
features = [
    [max_chromo_count, min_chromo_count, mean_chromo_count, chromo_count_std, mean_branch_length, clustering_coefficient, mean_simulation_score],
    # Add more feature vectors as needed
]

aicc_cutoff_labels = [
    aicc_cutoff1,
    aicc_cutoff2,
    # Add more AICc cutoff labels as needed
]

dataset = PhyloChromoDataset(features, aicc_cutoff_labels)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = AICcModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_features, batch_labels in data_loader:
        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Use the trained model to predict AICc cutoff
test_features = [
    [test_max_chromo_count, test_min_chromo_count, test_mean_chromo_count, test_chromo_count_std, test_mean_branch_length, test_clustering_coefficient, test_mean_simulation_score],
    # Add more test feature vectors as needed
]

test_features = torch.tensor(test_features, dtype=torch.float32)
predictions = model(test_features)

# Convert predictions to AICc cutoff values
aic_cutoffs = predictions.detach().numpy()

# Calculate the AICc cutoff based on the predictions
def calculate_aicc_cutoff(predictions):
    # Perform the necessary calculations here to determine the AICc cutoff
    # ...

    aicc_cutoff = ...  # Replace with your AICc cutoff calculation
    
    return aicc_cutoff

# Calculate the AICc cutoff using the predictions
aicc_cutoff = calculate_aicc_cutoff(aic_cutoffs)

# Print the AICc cutoff
print("AICc Cutoff:", aicc_cutoff)

######
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare the dataset
features = [
    [max_chromo_count, min_chromo_count, mean_chromo_count, chromo_count_std, mean_branch_length, clustering_coefficient, mean_simulation_score],
    # Add more feature vectors as needed
]

aicc_cutoff_labels = [
    aicc_cutoff1,
    aicc_cutoff2,
    # Add more AICc cutoff labels as needed
]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, aicc_cutoff_labels, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use the trained model to predict AICc cutoff
predictions = model.predict(X_test)

# Calculate the AICc cutoff based on the predictions
def calculate_aicc_cutoff(predictions):
    # Perform the necessary calculations here to determine the AICc cutoff
    # ...

    aicc_cutoff = ...  # Replace with your AICc cutoff calculation
    
    return aicc_cutoff

# Calculate the AICc cutoff using the predictions
aicc_cutoff = calculate_aicc_cutoff(predictions)

# Print the AICc cutoff
print("AICc Cutoff:", aicc_cutoff)
'''

######
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the dataset
features = [
    [max_chromo_count, min_chromo_count, mean_chromo_count, chromo_count_std, mean_branch_length, clustering_coefficient, mean_simulation_score],
    # Add more feature vectors as needed
]

aicc_cutoff_labels = [
    aicc_cutoff1,
    aicc_cutoff2,
    # Add more AICc cutoff labels as needed
]

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



import os
import re
import glob
import numpy as np

def get_sig_vals(res_dir, rng, heterogenous=False):

    sig_vals=[]
    mode = '_Heterogenous' if heterogenous else '_Homogenous'
    for _ in rng:
        sim_dir = os.path.join(res_dir, str(_)+mode)
        res_file = os.path.join(sim_dir, 'chromEvol.res')
        with open(res_file, 'r') as f:
            content = f.read()

        # match = re.search(r'AIC of the best model = ([\d.]+)', content)
        match = re.search(r'Final optimized likelihood is: ([\d.]+)', content)
        sig_val = float(match.group(1))

        sig_vals.append(sig_val)

    return np.array(sig_vals)

def main(args):

    train_split = args.train_split
    interval = args.interval



        #for file in files:
         #   if file.endswith('.tree'):


    training_set, testing_set = create_training_testing_sets(out_dir, train_split, interval)

    for tree_file in training_set:
        func1(tree_file)
    for tree_file in testing_set:
        func2(tree_file)

    # Results_date...
    dLs = get_sig_vals(res_dir, rng, heterogenous=True) - get_sig_vals(res_dir, rng)

    # Calculate the top x% of the resulting list
    p_cutoff = 0.05  # Example top x% value
    empirical_delta = np.percentile(dLs, 100 - (p_cutoff * 100))




    # Print the extracted values
    print('Likelihood:', likelihood)
    print('AICc:', aic)

if __name__ == '__main__':



    main(parser.parse_args())





#########
#########
#########

'''
def get_tree_lengths(tree):
    total_length = 0.0
    num_branches = 0
    for node in tree.traverse():
        if node.up:
            total_length += node.dist
            num_branches += 1
    if num_branches > 0:
        average_length = total_length / num_branches
    else:
        average_length = 0.0
    return total_length, average_length




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
    import math

    pass
    #node.get_sisters())


def calculate_entropy(mlar_tree):
    entropy = 0.0
    total_internal_nodes = 0
    node_counts = {}
    for node in mlar_tree.traverse():
        if not node.is_leaf():
            total_internal_nodes += 1
            child_counts = [node_count for child in node.get_children() if child.is_leaf() for node_count in child.name.split("-")]
            for count in child_counts:
                if count not in node_counts:
                    node_counts[count] = 0
                node_counts[count] += 1

    for count in node_counts.values():
        probability = count / total_internal_nodes
        entropy -= probability * math.log2(probability)

    return entropy


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

def calculate_max_chromosome_number(tree):
    max_chromosome_number = 0
    for node in tree.traverse("preorder"):
        if node.is_leaf():
            chromosome_number = int(node.name.split("-")[1])  # Assuming chromosome number is present in node name
            if chromosome_number > max_chromosome_number:
                max_chromosome_number = chromosome_number
    return max_chromosome_number


def count_transitions(mlar_tree):
    gains = 0
    losses = 0
    transitions = 0
    total_nodes = 0
    for node in mlar_tree.traverse():
        if not node.is_leaf() and node.up:
            parent_chrom = int(node.up.name.split("-")[1])  # Assuming chromosome number is present in node name
            child_chrom = int(node.name.split("-")[1])  # Assuming chromosome number is present in node name
            if parent_chrom < child_chrom:
                gains += child_chrom - parent_chrom
            elif parent_chrom > child_chrom:
                losses += parent_chrom - child_chrom
            if parent_chrom != child_chrom:
                transitions += 1
            total_nodes += 1
    return gains, losses, transitions, total_nodes

import statsmodels.api as sm

def calculate_phylogenetic_signal(fasta):


    chromosome_numbers=get_counts(fasta)

    # Fit the phylogenetic signal model
    phylo_signal = sm.OLS(chromosome_numbers, sm.add_constant(range(len(chromosome_numbers)))).fit()
    return phylo_signal.rsquared


import statsmodels.api as sm
import numpy as np

def calculate_chromosome_branch_correlation(mlar_tree):
    chromosome_numbers = []
    branch_lengths = []
    for node in mlar_tree.traverse():
        if node.is_leaf():
            chromosome_number = int(node.name.split("-")[1]) # maybe take the delta of the chromosome number
            chromosome_numbers.append(chromosome_number)
            branch_length = node.dist
            branch_lengths.append(branch_length)

    correlation = np.corrcoef(chromosome_numbers, branch_lengths)[0, 1]

    model = sm.OLS(branch_lengths, sm.add_constant(chromosome_numbers))
    results = model.fit()
    regression_coefficients = results.params[1]

    mean_interaction = np.mean(np.multiply(chromosome_numbers, branch_lengths))

    return correlation, regression_coefficients, mean_interaction


def count_cumulative_diff(mlar_tree):
    cumulative_diff = 0
    for node in mlar_tree.traverse("preorder"):
        if not node.is_root():
            parent_chromosome_number = int(node.up.name.split("-")[1])
            current_chromosome_number = int(node.name.split("-")[1])
            cumulative_diff += abs(parent_chromosome_number - current_chromosome_number)
    return cumulative_diff


def calculate_branch_length_skewness(tree):
    branch_lengths = [node.dist for node in tree.iter_descendants()]
    return pd.Series(branch_lengths).skew()

def calculate_features(df, tree_file):
    # Load phylogenetic tree using ete3
    tree = Tree(tree_file)

    # Calculate features
    features = {
        'Entropy': calculate_entropy(tree),
        'BranchLengthSkewness': calculate_branch_length_skewness(tree)
    }

    # Add features to DataFrame
    for feature, value in features.items():
        df[feature] = value

    # Normalize the AICc target
    scaler = MinMaxScaler()
    df['AICc_normalized'] = scaler.fit_transform(df[['AICc']])

    return df, features
'''



'''
def process_wait(procs):
import time
import subprocess

    # remove Nones from the list
    procs = [p for p in procs if p is not None]

    if len(procs) == 0:
        print('No processes to wait for.')
        return

    for process in procs:
        # Wait for the process to finish
        #process.wait()

        # Get the stdout output of the process
        #output = process.stdout.read()

        # Start the job using Popen
        # job_process = subprocess.Popen(['qsub', job_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(process)

        job_output, job_error = process.communicate()

        # check if the job submission was successful
        if process.returncode != 0:
            print(f"Error submitting job: {job_error}")
            return

        # extract the job ID
        job_id = job_output.strip().split('.')[0]

        # wait for the job to complete
        while True:
            # check the status of the job using qstat
            status_process = subprocess.run(['qstat', job_id])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # check if the job ID is not found (indicating job completion)
            if 'Unknown Job Id' in status_process.stderr:
                break

            time.sleep(60)
'''


