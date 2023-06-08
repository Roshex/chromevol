from ete3 import Tree
import re
import os
import random
import datetime
from subprocess import Popen
from defs import PYTHON_MODULE_COMMAND, CHROMEVOL_EMP_EXE, QUEUE

TOTAL_TO_REQ_SIM_RATIO = 1.5
MAX_CHR_NUM = 200

def get_time():
    return datetime.datetime.now().strftime('%y-%m-%d-%H-%M')

class paramio:

    def __init__(self, results_path, fname, counts_path, tree_path, *, dAICc_threshold=None):
        self.fname = fname
        self.dataFile = counts_path
        self.treeFile = tree_path
        self.resultsPathDir = results_path
        if dAICc_threshold is not None:
            self.deltaAICcThreshold = dAICc_threshold

    ### empirical parameters ###

    def _set_fixed_params(self):
        self.maxChrNum = '-10'
        self.minChrNum = '-1'
        self.tolParamOptimization = '0.1'
        self.seed = '1'
        self.optimizationMethod = 'Brent'
        self.maxParsimonyBound = 'true'
        self.baseNumOptimizationMethod = 'Ranges'
        self.optimizePointsNum = '10,2,1'
        self.optimizeIterNum = '0,1,3'

    def _set_heterogeneous_params(self, number_of_models, taxa_num=0, min_clade_size=5, nodes_file_path=None, max_number_of_models=None):
        self.optimizeIterNumNextRounds = '0,1'
        self.optimizePointsNumNextRounds = '2,1'
        self.backwardPhase = 'false'
        self.heterogeneousModel = 'true'
        self.parallelization = 'true'
        self.numOfModels = str(number_of_models)
        if max_number_of_models is not None:
            self.maxNumOfModels = str(max_number_of_models)
        self.minCladeSize = min( max(min_clade_size, int(number_of_models*taxa_num/50)), int(taxa_num*0.1) if (taxa_num>10) else 1 )
        if nodes_file_path is not None:
            self.nodeIdsFilePath = nodes_file_path

    def _set_homogeneous_params(self):
        self.heterogeneousModel = 'false'
        self.parallelization = 'false'
        self.numOfModels = '1'
        self.maxNumOfModels = '1'

    def _set_model_params(self, model, num_of_models):
        '''
        Functions: L (linear), E (exponent), C (constant), I (ignore)
        Rates: g (gain), l (loss), du (dupl), de (demi), b (base number)
        An example of a model: g+L_l+L_du+C_de+C_b+C
        '''
        dic_rates_functions = get_functions(model)
        dic_rates = {'gainFunc':'gain_', 'lossFunc':'loss_', 'duplFunc':'dupl_', 'demiDuplFunc':'demiPloidyR_',
                     'baseNumRFunc':'baseNumR_'}
        for key, value in dic_rates_functions.items():
            setattr(self, key, value)
        dic_params = {}
        cat_counter = 1
        for i in range(num_of_models):
            if dic_rates_functions['baseNumRFunc'] != 'IGNORE':
                dic_params['baseNum_' + str(i+1)] = str(cat_counter) + ';4'
                cat_counter += 1
            for func in dic_rates:
                rate = dic_rates[func]
                function = dic_rates_functions[func]
                if function == 'IGNORE':
                    continue
                default_parameters = get_default_parameters(function)
                dic_params[rate + str(i+1)] = str(cat_counter) + ';' + default_parameters
                cat_counter += 1
        for key, value in dic_params.items():
            setattr(self, key, value)

    def set_empirical(self, number_of_models=1, model='g+L_l+L_du+C_de+C_b+C', heterogenous=False, *, param_dict=None, taxa_num=0, min_clade_size=5, nodes_file_path=None, max_number_of_models=None):
        self._set_fixed_params()
        if heterogenous:
            self._set_heterogeneous_params(number_of_models, taxa_num, min_clade_size, nodes_file_path, max_number_of_models)
        else:
            self._set_homogeneous_params()
            number_of_models = 1
        if param_dict is not None:
            for key, value in param_dict.items():
                setattr(self, key, value)
        else:
            self._set_model_params(model, number_of_models)
        return self

    ### read parameters from res file 

    def _read_res_params(self, chromevol_res_path, freq_file_path, nodes_file_path=None, heterogeneous=False):
        
        res_file = open(chromevol_res_path, 'r')
        res_content = res_file.read()
        res_file.close()
        
        self.minChrNum = str(get_min_chromosome_number(res_content))
        self.maxChrNum = str(MAX_CHR_NUM)
        self.maxChrInferred = str(get_max_chromosome_number(res_content))
        self.branchMul = str(get_tree_scaling_factor(res_content))
        self.fixedFrequenciesFilePath = freq_file_path
        if heterogeneous:
            self.heterogeneousModel = 'true'
            self.nodeIdsFilePath = str(nodes_file_path)
            num_of_models = 2
        else:
            self.heterogeneousModel = 'false'
            self.nodeIdsFilePath = 'none'
            num_of_models = 1
        self.numOfModels = str(num_of_models)
        self.maxNumOfModels = str(num_of_models)

        return res_content

    def _read_res_model(self, res_content):

        for key, value in get_functions_settings(res_content).items():
            setattr(self, key, value)

        pattern_section = re.compile('Best chosen model[\s]+[#]+[\s]+(.*?)(\*|$)', re.DOTALL)
        pattern_params = re.compile('Chromosome\.([\D]+)([\d]*)_([\d]+)[\s]+=[\s]+([\S]+)')
        section = pattern_section.findall(res_content)[0][0]
        params_index_models = pattern_params.findall(section)
        rate_parameter_dict = {}
        # structure is { model: [ {param:[(index, value)]} ] }
        # where category is the param's index in the list of param dictionaries
        for rate_param, index_str, model_str, value_str in  params_index_models:

            model_entries = rate_parameter_dict.setdefault(model_str, [])
            existing_entry = next((e for e in model_entries if rate_param in e), None)
            if existing_entry:
                existing_entry[rate_param].extend( [(index_str, value_str)] )
            else:
                model_entries.append({ rate_param: [(index_str, value_str)] })

        return rate_parameter_dict

    def _write_rate_params(self, rate_parameter_dict, multiplier=None, manipulated_rates=None):

        func_conversion_dic = {'dupl':'duplFunc', 'demiPloidyR':'demiDuplFunc',
                               'gain':'gainFunc', 'loss':'lossFunc', 'baseNumR':'baseNumRFunc'}
        multiplier = [1] + [multiplier] if multiplier else [1]
        counter = 0
        for k in multiplier:
            for model, param_list in rate_parameter_dict.items():
                for category, param_dict in enumerate(param_list):
                    for key, val in param_dict.items():
                        # index is ignored for now
                        param = key + '_' + str(int(model)+counter)
                        setting = str(category) + ';' + val[0][1]
                        if manipulated_rates is not None and key in manipulated_rates:
                            func = vars(self)[func_conversion_dic[key]]
                            if (func == 'LINEAR') or (func == 'CONST') or ((func == 'EXP') and (val[0][0] == 0)):
                                # val[0][0] is the index, not sure why it's compared to 0
                                setting = str(category) + ';' + str(float(val[0][1])*k)
                        setattr(self, param, setting)
            counter += 1

    ### simulation parameters ###

    def set_simulated(self, chromevol_res_dir, num_of_simulations=120, *, heterogeneous=False, nodes_file_path=None, multiplier=None, manipulated_rates=None): 

        chromevol_res_path = os.path.join(chromevol_res_dir, 'chromEvol.res')
        freq_file_path = os.path.join(chromevol_res_dir, 'root_freq')
        if not os.path.exists(freq_file_path):
            create_freq_file(chromevol_res_path, freq_file_path)
        content = self._read_res_params(chromevol_res_path, freq_file_path, nodes_file_path, heterogeneous)

        self.seed = str(random.randint(1, 100000))
        self.simulateData = 'true'
        self.numOfSimulatedData = str(int(num_of_simulations * TOTAL_TO_REQ_SIM_RATIO))
        self.fracAllowedFailedSimulations = 0.1
        self.numOfRequiredSimulatedData = str(num_of_simulations)

        rate_parameter_dict = self._read_res_model(content)

        self._write_rate_params(rate_parameter_dict, multiplier, manipulated_rates)
        
        # note that the sequence at which parameters are set is important in this function!
        ml_tree = os.path.join(chromevol_res_dir, 'MLAncestralReconstruction.tree')
        max_base_num = 0
        if hasattr(self, 'baseNumRFunc') and self.baseNumRFunc != 'IGNORE':
            max_base_num = max([int(_.split(';')[1]) for var, _ in vars(self).items() if var.startswith('baseNum_')])
        # dataFile is counts_path
        self.maxBaseNumTransition = max(test_max_on_tree(self.dataFile, ml_tree), max_base_num+1)
        #
        # Ask Anat about the counts bug, and the logic of test_max_on_tree returning 0 conditional !!!
        # Also, ask if the multiplier syntax output makes sense, what happens if model number and mult overlap,
        # and that's the verdict on index?
        #

        return self
    
    ### write parameters to file ###

    def output(self, path=None):
        path = self.resultsPathDir if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)
        param_path = os.path.join(path, self.fname + '.params')
        del self.fname
        with open(param_path, 'w') as file:
            for name, value in vars(self).items():
                file.write(f'_{name} = {value}\n')



#################################################################
# Function I/O                                                  #
#################################################################


def get_functions_settings(res_content):
    '''
    extracts the functions set for each transition rate
    :param res_content: chromEvol results file content (chromEvol.res(=)
    :return: list of functions settings
    '''
    max_num_of_functions = 5
    dict_of_functions = {}
    pattern_section = re.compile('Assigned functions for each rate parameter:[\s]+(.*)?#', re.DOTALL)
    pattern_functions = re.compile('([\S]+):[\s]+([\S]+)')
    section = pattern_section.findall(res_content)[0]
    functions_raw = pattern_functions.findall(section)
    dict_functions = {'baseNumR': 'baseNumRFunc', 'dupl': 'duplFunc',
                      'demi': 'demiDuplFunc', 'gain': 'gainFunc', 'loss': 'lossFunc'}
    for i in range(min(max_num_of_functions, len(functions_raw))):
        transition_type, function = functions_raw[i]
        if transition_type in dict_functions:
            dict_of_functions[dict_functions[transition_type]] = function
    return dict_of_functions


def get_functions(model):
    dict_func_as_params = {}
    pattern = re.compile('([a-z]+)\+([A-Z]+)')
    dict_functions_names = {'g':'gainFunc', 'l':'lossFunc', 'du':'duplFunc', 'de':'demiDuplFunc', 'b':'baseNumRFunc'}
    dict_fucntions_defs = {'L':'LINEAR', 'C':'CONST', 'E':'EXP', 'I':'IGNORE'}
    rate_and_func = pattern.findall(model)
    for abb_rate, abb_func in rate_and_func:
        dict_func_as_params[dict_functions_names[abb_rate]] = dict_fucntions_defs[abb_func]
    return dict_func_as_params


def get_default_parameters(func):
    if func == 'CONST':
        param = '2'
    elif func == 'LINEAR':
        param = '2,0.1'
    elif func == 'EXP':
        param = '2,0.01'
    else:
        raise Exception('get_default_parameters(): Not implemented yet!!')
    return param


def write_general_params(parameters):
    content = ''
    for param in parameters:
        content += param + ' = '+ str(parameters[param]) + '\n'
    return content


def write_rate_parameters(rate_parameters):
    content = ''
    category = 1
    for model in rate_parameters:
        for param in rate_parameters[model]:
            content += '_'+param+'_'+ str(model)+' = '+ str(category)+';'
            for i in range(len(rate_parameters[model][param])):
                value = rate_parameters[model][param][i][1]
                if i != len(rate_parameters[model][param])-1:
                    content += str(value)+','
                else:
                    content += str(value)+'\n'
            category += 1
    return content



#################################################################
# File Management                                               #
#################################################################


def create_job_file(path, job_name, mem=4, ncpu=1, exe=CHROMEVOL_EMP_EXE, queue=QUEUE, on=True):

    if not os.path.exists(path):
        os.makedirs(path)

    module_python = PYTHON_MODULE_COMMAND
    param_path = os.path.join(path, f'{job_name}.params')
    cmd = f'{module_python}{exe} param={param_path} > {os.path.join(path, "out.txt")} 2> {os.path.join(path, "err.txt")}\n'
    job_path = os.path.join(path, f'{job_name}.sh')
    text = f'''\
#!/bin/bash
#PBS -S /bin/bash
#PBS -r y
#PBS -q {queue}
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N {job_name}
#PBS -e {path}.ER
#PBS -o {path}.OU
#PBS -l select=ncpus={ncpu}:mem={mem}gb
cd {path}
{cmd}
'''
    with open(job_path, 'w') as file:
        file.write(text)
    if on:
        Popen(['qsub', job_path])


def create_nodes_split_file(nodes_file_path, sampling_frac, tree_path):
    tree = Tree(tree_path, format=1)
    dict_size_node = search_node_sizes(tree)
    subclade_size = find_the_closest_size(dict_size_node, sampling_frac, len(tree))
    if subclade_size < 2:
        raise ValueError(f'Sampling fraction is too small; subclade_size={subclade_size} must be bigger than 2!')
    nodes = dict_size_node[subclade_size]
    node = random.sample(nodes, 1)[0]
    mrcas_defs = [get_farthest_leaves(tree), get_farthest_leaves(node)]
    num_of_models = 2
    node_splits = '\n'.join([f'Model #{i} = ({mrcas_defs[i-1][0]},{mrcas_defs[i-1][1]});' for i in range(1, num_of_models+1)])
    with open(nodes_file_path, 'w') as nodes_file:
        nodes_file.write(node_splits)


def create_freq_file(res_file_path, freq_file_path):
    '''
    create root frequency file
    :param res_file_path: the file path of the res chromEvol file
    :param freq_file_path: root frequencies file to be written to
    :return:
    '''
    # make folder Results_g+L_l+L_du+E_de+C_b+C
    res_file = open(res_file_path, 'r')
    content = res_file.read()
    res_file.close()
    pattern_section = re.compile('Best chosen model[\s]+[#]+[\s]+(.*?)(\*|$)', re.DOTALL)
    section_best_model = pattern_section.findall(content)[0][0]
    min_chr_num = get_min_chromosome_number(content)
    pattern_freqs = re.compile('F\[([\d]+)\][\s]+=[\s]([\S]+)')
    root_frequencies = pattern_freqs.findall(section_best_model)
    inidices_with_freqs = [(int(state)-min_chr_num, float(freq)) for state,freq in root_frequencies]
    freq_file = open(freq_file_path, 'w')
    for i in range(inidices_with_freqs[0][0]):
        freq_file.write('0\n')
    for i in range(len(inidices_with_freqs)):
        freq_file.write(str(inidices_with_freqs[i][1])+'\n')

    freq_file.close()


def create_rescaled_tree(manipulated_rates, multiplier, nodes_file_path, original_tree_path, new_tree_path, expectation_file_path):
    tree = Tree(original_tree_path, format=1)
    original_tree_length = get_tree_length(tree)
    if manipulated_rates == 'all':
        factor = 1/multiplier
    else:
        factor = 1/get_scaling_factor_from_expectations(expectation_file_path, manipulated_rates, multiplier)
    mrca_node = get_mrca(nodes_file_path, tree)
    rescale_tree(mrca_node, factor)
    factor_to_original_length = original_tree_length/get_tree_length(tree)
    rescale_tree(tree, factor_to_original_length)
    print('tree length', str(get_tree_length(tree)))
    tree.write(format=1, outfile=new_tree_path)


def get_min_chromosome_number(res_content, file_handler=None):
    '''
    get min chromosome number from results file
    :param res_content: the content of the results file (str)
    :param file_handler: file handler of the results file
    :return: min chromosome number
    '''
    if res_content is None:
        res_content = file_handler.read()
    pattern_min_chr_num = re.compile('Min[\s]+allowed[\s]+chromosome[\s]+number[\s]+=[\s]+([\d]+)')
    min_chr_num_sec = pattern_min_chr_num.findall(res_content)
    min_chr_num = int(min_chr_num_sec[0])
    return min_chr_num


def get_max_chromosome_number(res_content, file_handler=None):
    '''
    get max chromosome number from results file
    :param res_content: the content of the results file (str)
    :param file_handler: file handler of the results file
    :return: max chromosome number
    '''
    if res_content is None:
        res_content = file_handler.read()
    pattern_max_chr_num = re.compile('Max allowed chromosome number[\s]+=[\s]+([\d]+)')
    max_chr_num = int(pattern_max_chr_num.findall(res_content)[0])
    return max_chr_num



#################################################################
# Tree Management                                               #
#################################################################


def get_farthest_leaves(tree):
    sons = tree.get_children()
    farthest_leaves = []
    for i in range(len(sons)):
        son = sons[i]
        if son.is_leaf():
            farthest_leaves.append(son.name)
        else:
            son_leaves = son.get_leaves()
            farthest_leaves.append(son_leaves[0].name)

    return farthest_leaves


def find_the_closest_size(clade_sizes, sampling_frac, full_tree_size):
    min_tree_size = 400
    required_size = int(sampling_frac*full_tree_size)
    for clade_size in clade_sizes:
        if abs(required_size-clade_size) < abs(min_tree_size-required_size):
            min_tree_size = clade_size
    return min_tree_size


def get_tree_length(tree):
    sum_of_branch_lengths = 0
    for node in tree.traverse():
        sum_of_branch_lengths += node.dist
    return sum_of_branch_lengths


def rescale_tree(tree, factor):
    for node in tree.traverse():
        node.dist *= factor
    return


def read_counts_data(src_counts_path):
    file = open(src_counts_path, 'r')
    content = file.read()
    file.close()
    pattern = re.compile('>([\S]+)[\s]+([\S]+)', re.MULTILINE)
    return pattern.findall(content)


def test_max_on_tree(counts_file, tree_file):
    '''
    calculates the maximal transition on the inferred tree
    :param counts_file: counts input file
    :param tree_file: ml inferred tree
    :return: if the maximal transitions if equal or larger than the original range -
                return 0
            otherwise
                return a list of the new base number and the max base transition
    '''
    counts = get_counts(counts_file)
    counts_range = max(counts)-min(counts)
    max_base_on_tree = get_max_transition(tree_file)
    if max_base_on_tree >= counts_range:
        return 0
    max_base = max(max_base_on_tree, 3)
    return max_base


def get_max_transition(tree_file):
    '''
    searches for the largest transition that was made on the phylogeny itself, between internal node and tip, or between internal nodes. Used in base_num_models.py
    :param tree_file: phylogeny file
    :return: a number representing the maximal transition on the tree
    '''
    t = Tree(tree_file, format=1)
    max_transition = 0
    for node in t.traverse():
        if node.name == '':
            continue
        if not node.is_leaf():
            num1 = regex_internal(node.name)
            for child in node.get_children():
                if child.is_leaf():  # if the child is a tip - parse number from tip label
                    num2 = regex_tip(child.name)
                else:  # if the child is an internal node - take number
                    num2 = regex_internal(child.name)
                tmp_score = abs(num1 - num2)
                if max_transition < tmp_score:
                    max_transition = tmp_score
    return max_transition


def regex_tip(str):
    '''
    extracts the number from a tip label in a phylogeny
    :param str: tip label
    :return: count of tip
    '''
    tmp = re.search('(\d+)', str)
    if tmp:  # there is a number at the tip, and not X
        num = int(tmp.group(1))
    return num


def regex_internal(str):
    '''
    extracts the number from the internal node's name from a phylogeny, in a NX-XX format. Used in get_max_transition(tree_file)
    :param str: internal node's label
    :return: count of node
    '''
    tmp = re.search('N\d+\-(\d+)', str)
    if tmp:
        num = int(tmp.group(1))
    return num


def get_counts(counts_file):
    file = open(counts_file)
    content = file.read()
    file.close()
    pattern = re.compile('>.+?\n(\d+)', re.MULTILINE)
    counts_str = pattern.findall(content)
    counts = [int(count) for count in counts_str]
    return counts


def search_node_sizes(node):
    'Finds all node sizes'
    sizes = {}
    for n in node.traverse():
        size = len(n)
        if not (size in sizes):
            sizes[size] = [n]
        else:
            sizes[size].append(n)
    return sizes


def get_tree_scaling_factor(content):
    '''
    extracts the tree scaling factor that should be used for the simulations
    :param content: chromEvol results file (chromEvol.res(=)
    :return: the tree scaling factor
    '''
    pattern = re.compile('Tree scaling factor is:[\s]+([\S]+)')
    scaling_factor = float(pattern.findall(content)[0])
    return scaling_factor


def get_mrca(nodes_file_path, tree):
    nodes_file = open(nodes_file_path, 'r')
    content = nodes_file.read()
    nodes_file.close()
    pattern = re.compile('Model\s+#\d+\s+=\s+\(([^,]+),([^)]+)\);')
    partitions = pattern.findall(content)
    species1, species2 = partitions[-1]
    ancestor = tree.get_common_ancestor(species1, species2)
    return ancestor


def get_descendent_leaves(node):
    leaves = node.get_leaves()
    leaves_names = [n.name for n in leaves]
    return set(leaves_names)


def get_leaves_under_shifts_tree_node(tree_path):
    tree = Tree(tree_path, format=1)
    pattern = re.compile('([\S]+)-([\d]+)$')
    leaves_names = []
    for node in tree.traverse():
        name_parts = pattern.findall(node.name)[0]
        name, model = name_parts[0], int(name_parts[1])
        if (model == 2) and (node.is_leaf()):
            leaves_names.append(name)
    return set(leaves_names)


def get_scaling_factor_from_expectations(expectation_file_path, manipulated_rates, multiplier):
    if manipulated_rates == 'poly':
        rates = ['DUPLICATION', 'DEMI-DUPLICATION', 'BASE-NUMBER']
    elif manipulated_rates == 'dys':
        rates = ['GAIN', 'LOSS']
    else:
        raise Exception('get_scaling_factor_from_expectations()' + ' such rate '+manipulated_rates + ' does not exists!')
    expectation_file = open(expectation_file_path, 'r')
    content = expectation_file.read()
    expectation_file.close()
    pattern_section = re.compile('#TOTAL EXPECTATIONS:(.*)', re.DOTALL)
    section = pattern_section.findall(content)[0]
    pattern_exp = re.compile('([\S]+):[\s]+([\S]+)')
    expectations = pattern_exp.findall(section)
    weight = 0
    sum_of_weights = 0
    for rate_type, expecation in expectations:
        if rate_type == 'TOMAX':
            continue
        if rate_type in rates:
            weight += float(expecation)
        sum_of_weights += float(expecation)
    factor = (weight/sum_of_weights) * multiplier
    return factor

