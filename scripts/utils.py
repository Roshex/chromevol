import re
import os
import time
import math
import random
import datetime
import numpy as np
import xgboost as xgb
import scipy.stats as stt
import statsmodels.api as sm
from os.path import join as opj
from ete3 import Tree
from subprocess import Popen
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from defs import *


TOTAL_TO_REQ_SIM_RATIO = 1.5
MAX_CHR_NUM = 200


def get_time():
    return datetime.datetime.now().strftime('%y-%m-%d-%H-%M')


def get_res_path(path, standalone):
    return path if not standalone else opj(path, 'Results_' + get_time())


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
        
        self.fixedFrequenciesFilePath = freq_file_path
        self.minChrNum = str(get_min_chromosome_number(res_content))
        self.maxChrInferred = str(get_max_chromosome_number(res_content))
        self.maxChrNum = str(MAX_CHR_NUM)
        self.branchMul = str(get_tree_scaling_factor(res_content))
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
        rate_param_dict = {}
        # structure is { model: [ {param:[(index, value)]} ] }
        # where category is the param's index in the list of param dictionaries
        for param, index, model, value in  params_index_models:
            # ChromEvol expects 'demiPloidyR' instead of 'demi' in params file!
            if param == 'demi':
                param = 'demiPloidyR'

            model_entries = rate_param_dict.setdefault(model, [])
            existing_entry = next((e for e in model_entries if param in e), None)
            if existing_entry:
                existing_entry[param].extend( [(index, value)] )
            else:
                model_entries.append({ param: [(index, value)] })

        return rate_param_dict

    def _write_rate_params(self, rate_param_dict, multiplier=None, manipulated_rates=None):

        func_conversion_dic = {'dupl':'duplFunc', 'demiPloidyR':'demiDuplFunc',
                               'gain':'gainFunc', 'loss':'lossFunc', 'baseNumR':'baseNumRFunc'}
        multiplier = [1] + [multiplier] if multiplier else [1]
        counter = 0
        for k in multiplier:
            for model, param_list in rate_param_dict.items():
                for category, param_dict in enumerate(param_list):
                    for key, val in param_dict.items():
                        if k == 1 or (manipulated_rates is not None and key in manipulated_rates):
                            param = key + '_' + str(int(model)+counter)
                            # first index == '' means baseNum -> a single int value
                            setting = val[0][1] if val[0][0] == '' else ','.join([str(float(val[_][1])*k) for _ in range(len(val))])
                            setattr(self, param, str(category+1) + ';' + setting)
            counter += 1

    ### simulation parameters ###

    def set_simulated(self, chromevol_res_dir, num_of_simulations=120, *, heterogeneous=False, nodes_file_path=None, multiplier=None, manipulated_rates=None): 

        chromevol_res_path = opj(chromevol_res_dir, 'chromEvol.res')
        freq_file_path = opj(chromevol_res_dir, 'root_freq')
        if not os.path.exists(freq_file_path):
            create_freq_file(chromevol_res_path, freq_file_path)
        content = self._read_res_params(chromevol_res_path, freq_file_path, nodes_file_path, heterogeneous)

        self.seed = str(random.randint(1, 100000))
        self.simulateData = 'true'
        self.numOfSimulatedData = str(int(num_of_simulations * TOTAL_TO_REQ_SIM_RATIO))
        self.fracAllowedFailedSimulations = 0.1
        self.numOfRequiredSimulatedData = str(num_of_simulations)

        rate_param_dict = self._read_res_model(content)
        self._write_rate_params(rate_param_dict, multiplier, manipulated_rates)
        
        # note that the sequence at which parameters are set is important in this function!
        ml_tree = opj(chromevol_res_dir, 'MLAncestralReconstruction.tree')
        max_base_num = 0
        if hasattr(self, 'baseNumRFunc') and self.baseNumRFunc != 'IGNORE':
            max_base_num = max([int(_.split(';')[1]) for var, _ in vars(self).items() if var.startswith('baseNum_')])
        # important: in sim mode dataFile is NOT counts_path, but final output dir!
        counts_path = opj(chromevol_res_dir, 'counts.fasta')
        self.maxBaseNumTransition = max(test_max_on_tree(counts_path, ml_tree), max_base_num+1)
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
        param_path = opj(path, self.fname + '.params')
        del self.fname
        with open(param_path, 'w') as file:
            for name, value in vars(self).items():
                file.write(f'_{name} = {value}\n')



#################################################################
# File Management                                               #
#################################################################


def get_nth_parent(root, n):
    for _ in range(n):
        root = os.path.dirname(root)
    return os.path.basename(root)


def write_expected_outputs(path, output_dirs, name=None, flag=None):
    cats = { 'paths': output_dirs, 'name': name, 'flag': flag }
    path = opj(path, 'expected_outputs.txt')
    with open(path, 'w') as f:
        for cat, val in cats.items():
            if val is not None:
                f.write(f'>{cat}\n')
                val = '\n'.join(val) + '\n' if isinstance(val, list) else f'{val}\n'
                f.write(val)


def parse_expected_outputs(path):
    outputs = {'paths': []}
    current_cat = 'paths'
    path = opj(path, 'expected_outputs.txt')
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_cat = line[1:]
                    outputs[current_cat] = []
                    continue
                outputs[current_cat].append(line)
    except (FileNotFoundError, IOError):
        return None
    return outputs


def check_output(path, name='out.txt', flag='Total running time is:', config=False, notif1='checking', notif2='Outputs found'):

    if config:
        print(f'{notif1}: expected_outputs.txt')
        outputs = parse_expected_outputs(path)
        if outputs is None:
            print('file does not exist yet')
            return False
        paths = outputs['paths']
        name = outputs['name'] if 'name' in outputs else name
        flag = outputs['flag'] if 'flag' in outputs else flag
    else:
        paths = [path]

    for p in paths:
        p = opj(p, name)
        print(f'{notif1}: {p}')
        if os.path.isfile(p):
            with open(p, 'r') as f:
                content = f.read()
                if flag not in content:
                    print('file is yet to be completed')
                    return False
        else:
            print('file does not exist yet')
            return False
    print(f'{notif2}: {len(paths)}')
    return True


def wait_for_output(path, **kwargs):
    while not check_output(path, **kwargs):
        # wait half a minute
        time.sleep(30)
    

def concat_paths(path, mode='Homogenous'):
    emp_out = opj(path, mode)
    sim_out = opj(emp_out, 'Sims/')
    inf_out = opj(sim_out, 'Infer_on_sims')
    return emp_out, sim_out, inf_out


def do_pilot_cmd(path, name, sim_num, model, mode, data, **kwargs):
    cwd = os.getcwd()
    emp_out, sim_out, inf_out = concat_paths(path, mode)
    scripts = [
        {'call': f'{opj(cwd, "infer_empirical.py")} -i {path} -o {path} -m {model} -d {mode}', 'output': emp_out, 'config': False},
        {'call': f'{opj(cwd, "simulate_models.py")} -i {emp_out} -o {sim_out} -n {sim_num}', 'output': sim_out, 'config': False},
        {'call': f'{opj(cwd, "infer_on_sims.py")} -i {sim_out} -t {opj(emp_out, "tree.newick")} -m {model} -o {inf_out} -r {sim_num}', 'output': inf_out, 'config': True},
        #{'call': f'{opj(cwd, "get_features.py")} -e {path} -s {sim_out} -i {inf_out} -d {data} -n {sim_num}', 'output': cwd, 'config': False},
        {'call': f'{opj(cwd, "hello.py")}', 'output': cwd, 'config': False}
    ]

    cmd = ''
    for script in scripts:
        if not check_output(script['output'], config=script['config']):
            cmd += f'python {script["call"]}\n'
    
    send_job(path, name, cmd, **kwargs)


def do_chevol_cmd(path, name, exe=CHROMEVOL_EMP_EXE, standalone=False, **kwargs):

    param_path = opj(path, f'{name}.params')
    cmd = f'{exe} param={param_path} > {opj(path, "out.txt")} 2> {opj(path, "err.txt")}\n'
        
    if standalone:
        send_job(path, name, cmd, **kwargs)
    else:
        if check_output(path):
            return # already done
        else:
            print(f'running: {param_path}')
            os.system(cmd)
            return wait_for_output(path, notif1='waiting', notif2='Done waiting')


def send_job(path, name, cmd, mem=4, ncpu=1, queue=QUEUE):

    if not os.path.exists(path):
        os.makedirs(path)

    module = MODULE_COMMAND
    env = ACTIVATE_ENV
    job_path = opj(path, f'{name}.sh')
    err_path = opj(path, f'{name}.err')
    out_path = opj(path, f'{name}.out')
    text = f'''\
#!/bin/bash
#PBS -S /bin/bash
#PBS -r y
#PBS -q {queue}
#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH
#PBS -N {name}
#PBS -e {err_path}
#PBS -o {out_path}
#PBS -l select=ncpus={ncpu}:mem={mem}gb
source ~/.bashrc
cd {path}
module load {module}
{env}
{cmd}
'''
    with open(job_path, 'w') as file:
        file.write(text)
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


def extract_score(res_dir, string):
    # read chromEvol.res file and regex it
    match = None
    with open(os.path.join(res_dir, 'chromEvol.res'), 'r') as f:
        text = f.read()
        match = re.search(string, text)
    return match.group(1) if match else 'none' # convert to float?



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


def count_taxa(tree):
    return len(tree.get_leaf_names())


def count_polytomy(tree, n):
    internal_nodes = len(list(tree.traverse())) - n
    max_internal_nodes = n-1
    polytomy_ratio = 1 - (internal_nodes / max_internal_nodes)
    return polytomy_ratio


def remove_root_polytomies(tree_file, counts_file, mx=2):

    # remove root leaves if over 2
    tree = Tree(tree_file)
    flag = 0
    polytomy_children_names = []
    root = tree.get_tree_root()
    for child in root.get_children():
        if child.is_leaf() and flag >= mx:
            polytomy_children_names.append(child.name)
            root.remove_child(child)
        flag += 1
    tree.write(format=5, outfile=tree_file)

    # remove detached children from counts file
    with open(counts_file, 'r') as input_counts_file:
        lines = input_counts_file.readlines()
    with open(counts_file, 'w') as output_counts:
        for line in lines:
            if line.startswith('>'):
                name = line[1:].strip()
                if name not in polytomy_children_names:
                    output_counts.write(line)
            else:
                if name not in polytomy_children_names:
                    output_counts.write(line)


def create_counts_hash(counts_file):
    """
    puts all counts from a counts file in a dictionary: taxa_name: count. Taxa with an X count are skipped.
    If there are counts that are X the function returns two hashes and a set of the taxa to prune.
    :param counts_file in FASTA format
    :return:(2) dictionary of counts
    """
    d = {}
    with open(counts_file, "r") as counts_handler:
        for line in counts_handler:
            line = line.strip()
            if line.startswith('>'):  # taxon name
                name = line[1:]
            else:
                if line != "x":
                    num = int(line)
                    d[name] = num
    return d


def fitch(tree_file, counts_file=False, mlar_tree=True):
    """
    calculates the maximum parsimony score following Fitch algorithm, where the states are chromosome counts.
    :param tree_file: MLAncestralReconstruction or original tree file
    :param counts: counts file (needed if simulated, or if original tree is used)
    :param mlar_tree: True if MLAncestralReconstruction tree is used, False if original tree is used
    :return: number of unions (parsimony score)
    """
    try:
        t = Tree(tree_file, format=1)
        score = 0
        if counts_file:
            d = create_counts_hash(counts_file)

        for node in t.traverse("postorder"):
            if not node.is_leaf():  # internal node
                lst = []  # list version
                for child in node.get_children():
                    if child.is_leaf():  # if the child is a tip - parse number from tip label or counts file
                        if mlar_tree:
                            if counts_file:  # dictionary exists if the tree is simulated --> take the number from it
                                name = re.search("(.*)\-\d+", child.name)
                                if name:
                                    num = {int(d.get(name.group(1)))}
                            else:  # calculation on original counts
                                 num = {regex_tip(child.name)}
                        else:
                            num = {int(d.get(child.name))} 
                    else:  # if the child is an internal node - take number
                        num = child.name
                    lst.append(num)
                intersect = set.intersection(*lst)
                union = set.union(*lst)
                if len(intersect) == 0:
                    result = union
                    score += 1
                else:
                    result = intersect
                node.name = result
    except Exception as ex:
        print(f"Exception: {ex}. Unable to calculate parsimony score")
        score = None
    return score


def symmetry_score(tree, topology_only=False):
    max_leaf_distances = []
    internal_nodes = 0

    for node in tree.traverse():
        if not node.is_leaf() and not node.is_root():
            max_distance = 0
            for leaf in node.iter_leaves():
                distance = node.get_distance(leaf, topology_only=topology_only)
                if distance > max_distance:
                    max_distance = distance
            max_leaf_distances.append(max_distance)
            internal_nodes += 1

    if internal_nodes > 0:
        average_max_distance = sum(max_leaf_distances) / internal_nodes
        symmetry_score = 1.0 / average_max_distance
        return symmetry_score
    else:
        return 0.0


def colless_index(node):
    if node.is_leaf():
        return 0
    colless_index_sum = 0
    child_pairs = combinations(node.children, 2)
    for pair in child_pairs:
        left = pair[0].get_leaf_names()
        right = pair[1].get_leaf_names()
        colless_index_sum += abs(len(left) - len(right))
    return colless_index_sum + sum(colless_index(child) for child in node.children)


def anomaly_score(counts):
    # calculate the interquartile range (IQR)
    q1 = np.percentile(counts, 25)
    q3 = np.percentile(counts, 75)
    iqr = q3 - q1

    # find anomalies outside the upper and lower limits
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    anomalies = [chromosome for chromosome in counts if chromosome > upper_limit or chromosome < lower_limit]

    return sum(anomalies)/sum(counts)


def get_branch_lengths(tree):
    return [node.dist for node in tree.iter_descendants()]


def get_stats(v):
    # skewness, kurtosis, MAD, signal
    return stt.skew(v), stt.kurtosis(v), np.mean(np.abs(v - np.median(v))), sm.OLS(v, sm.add_constant(range(len(v)))).fit().rsquared



#################################################################
# Labeled Tree Features                                         #
#################################################################


def extract_chrom(node):
    return int(re.search(".*\-(\d+)", node.name).group(1))


def count_transitions(mlar_tree):
    gains = 0
    losses = 0
    duples = 0
    demis = 0
    internal_nodes = 0
    for node in mlar_tree.traverse():
        if not node.is_leaf() and node.up:
            parent_chrom = extract_chrom(node.up)
            child_chrom = extract_chrom(node)
            if parent_chrom - child_chrom == 1:
                gains += 1
            elif parent_chrom - child_chrom == -1:
                losses += 1
            elif 2*parent_chrom == child_chrom:
                duples += 1
            else:
                demis += 1
            internal_nodes += 1
    return gains/internal_nodes, losses/internal_nodes, duples/internal_nodes, demis/internal_nodes


def chromosome_branch_correlation(mlar_tree):
    chromosome_traits = []
    branch_lengths = []
    cumulative_diff = 0
    for node in mlar_tree.traverse():
        if node.up:
            parent_num = extract_chrom(node.up)
            difference = extract_chrom(node) - parent_num
            chromosome_trait = (difference) / parent_num
            chromosome_traits.append(chromosome_trait)
            branch_length = node.dist
            branch_lengths.append(branch_length)
            cumulative_diff += abs(difference)

    correlation = np.corrcoef(chromosome_traits, branch_lengths)[0, 1]

    model = sm.OLS(branch_lengths, sm.add_constant(chromosome_traits))
    results = model.fit()
    regression_coefficients = results.params[1]

    mean_interaction = np.mean(np.multiply(chromosome_traits, branch_lengths))

    return cumulative_diff, correlation, regression_coefficients, mean_interaction


def get_entropy(mlar_tree):
    entropy = 0.0
    total_internal_nodes = 0
    node_counts = {}
    for node in mlar_tree.traverse():
        if not node.is_leaf():
            total_internal_nodes += 1
            child_counts = [extract_chrom(child) for child in node.get_leaves()]
            for count in child_counts:
                if count not in node_counts:
                    node_counts[count] = 0
                node_counts[count] += 1

    for count in node_counts.values():
        probability = count / total_internal_nodes
        entropy -= probability * math.log2(probability)

    return entropy



#################################################################
# Machine Learning                                              #
#################################################################


def split_data(data, feature_names, target_name):

    target = data[target_name].values
    features = data[feature_names].values
    train_mask = data['selected'] == 'Train'
    test_mask = data['selected'] == 'Test'
    train_target, test_target = target[train_mask], target[test_mask]
    train_features, test_features = features[train_mask], features[test_mask]

    return train_features, test_features, train_target, test_target

def feature_selection(train_features, train_target, num_rounds, num_features):

    dtrain = xgb.DMatrix(train_features, label=train_target)
    params = {'objective': 'reg:squarederror'}
    model = xgb.train(params, dtrain, num_rounds)

    scores = model.get_score(importance_type='weight')
    sorted_features = sorted(scores, key=scores.get, reverse=True)
    selected_features = sorted_features[:num_features]

    print('Selected features and their scores:')
    for feature in selected_features:
        print(f"{feature}: {scores[feature]}")

    scores = np.array([scores[feature] for feature in selected_features])
    return scores

def train_xgboost_model(train_features, train_target, num_rounds, num_features):

    scores = feature_selection(train_features, train_target, num_rounds, num_features)
    selected_feature_indices = np.argsort(scores)[-num_features:]
    selected_train_features = train_features[:, selected_feature_indices]

    # convert the data to DMatrix format & set parameters
    dtrain = xgb.DMatrix(selected_train_features, label=train_target)
    params = {
        'objective': 'reg:squarederror',  # regression problem
        'eval_metric': 'rmse'  # Root Mean Squared Error metric
    }
    # return XGBoost model
    return xgb.train(params, dtrain, num_rounds, verbose_eval=False)

def evaluate_model(model, test_features, test_target):
    # convert the test features to DMatrix format & make predictions
    dtest = xgb.DMatrix(test_features)
    predictions = model.predict(dtest)

    # evaluate the model
    rmse = mean_squared_error(test_target, predictions) ** 0.5
    print(f"Root Mean Squared Error (RMSE) on test set: {rmse}")

    # perform cross-validation
    cv_scores = -cross_val_score(model, test_features, test_target, cv=5, scoring='neg_root_mean_squared_error')
    mean_cv_rmse = cv_scores.mean()
    print(f"Cross-validated RMSE: {mean_cv_rmse}")

    return predictions
