from ete3 import Tree

# Example phylogenetic tree in Newick format
newick_tree = "(A:0.1,(((B:0.2,E:0.1),D:0.2),(C:0.3,D:0.4):0.5):0.6);"

# Load the phylogenetic tree
tree = Tree(newick_tree)

# 1. Phylogenetic Diversity
def compute_phylogenetic_diversity(node):
    if node.is_leaf():
        return node.dist
    branch_lengths = [child.dist for child in node.children]
    return node.dist + sum(compute_phylogenetic_diversity(child) for child in node.children) + sum(branch_lengths) / len(branch_lengths)

phylo_diversity = compute_phylogenetic_diversity(tree)
print("Phylogenetic Diversity:", phylo_diversity)

# 1. Phylogenetic Diversity (Faith's Phylogenetic Diversity)
def compute_faiths_phylogenetic_diversity(node):
    if node.is_leaf():
        return 1
    branch_lengths = [child.dist for child in node.children]
    subtree_faiths_pd = sum(compute_faiths_phylogenetic_diversity(child) for child in node.children)
    return subtree_faiths_pd + sum(branch_lengths) / len(branch_lengths)

faiths_pd = compute_faiths_phylogenetic_diversity(tree)
print("Faith's Phylogenetic Diversity:", faiths_pd)

# 3. Phylogenetic Balance (Colless index)
def compute_colless_index(node):
    if node.is_leaf():
        return 0
    left_children = node.children[0].get_leaf_names()
    right_children = node.children[1].get_leaf_names()
    colless_index = abs(len(left_children) - len(right_children))
    return colless_index + compute_colless_index(node.children[0]) + compute_colless_index(node.children[1])

colless_index = compute_colless_index(tree)
print("Phylogenetic Balance (Colless index):", colless_index)

'''

Phylogenetic Diversity: Measure of the diversity or evolutionary distance among taxa in the phylogenetic tree used in the simulation.
Number of Taxa: The total number of taxa or species in the phylogenetic tree.
Phylogenetic Balance: The balance or evenness of branching patterns in the phylogenetic tree.
Genome Size: The size of the genome for each taxon in the simulation.
Reproductive Strategy: Categorical feature indicating the reproductive strategy (e.g., sexual reproduction, asexual reproduction) observed in the taxa.
Habitat Type: Categorical feature indicating the habitat type (e.g., terrestrial, aquatic) of the taxa.
Life History Traits: Additional categorical or numerical features capturing specific life history traits or ecological characteristics of the taxa (e.g., generation time, dispersal ability).


'''



