import os
import argparse
from utils import paramio, get_time, create_job_file, create_nodes_split_file, create_rescaled_tree
from defs import *

def main(args):

    xstr = lambda s: '' if s is None else str(s)
    lstd = lambda l: [l] if not isinstance(l, list) else l

    in_dir = args.in_dir
    num_of_simulations = args.num_of_simulations
    model = args.model
    out_dir = os.path.join(args.out_dir, 'Results_' + get_time())
    sampling_fractions = lstd(args.sampling_fractions)
    multipliers = lstd(args.multipliers)
    manipulated_rates = lstd(args.manipulated_rates)
    hetero_res_dir = args.empirical_hetero_params
    q_on = args.q_on

    other_kwargs = {}
    other_dir = out_dir
    other_modes_flag = False
    if [1] == multipliers: # homogenous simulation mode
        sampling_fractions = [None]
        multipliers = manipulated_rates = [None]
    else: # other simulation modes
        other_modes_flag = True
        if (0 in multipliers) and (hetero_res_dir is not None): # empirical params simulation mode
            other_dir = hetero_res_dir
            multipliers = manipulated_rates = [None]
        else: # multiplier simulation mode 
            pass

    manipulations_on = {'poly':['demiPloidyR', 'dupl', 'baseNumR'], 'dys':['gain', 'loss'],
                        'all':['demiPloidyR', 'dupl', 'baseNumR', 'gain', 'loss'], None:None}#
    tree_path = os.path.join(in_dir, 'tree.newick')
    counts_path = os.path.join(in_dir, 'counts.fasta')
    expectation_file_path = os.path.join(other_dir, 'expectations_second_round.txt')#

    for sampling_frac in sampling_fractions:

        sampling_frac_dir = os.path.join(other_dir, xstr(sampling_frac))
        if not os.path.exists(sampling_frac_dir):
            os.makedirs(sampling_frac_dir)

        if other_modes_flag:
            nodes_file_path = os.path.join(sampling_frac_dir, 'nodes.txt')
            if not os.path.exists(nodes_file_path):
                create_nodes_split_file(nodes_file_path, sampling_frac, tree_path)
            other_kwargs = {'nodes_file_path': nodes_file_path, 'heterogeneous': True}
      
        for multiplier in multipliers:

            multiplier_dir_path = os.path.join(sampling_frac_dir, xstr(multiplier))
            if not os.path.exists(multiplier_dir_path):
                os.makedirs(multiplier_dir_path)

            for manipulation_model in manipulated_rates:

                mult_manupulation_dir = os.path.join(multiplier_dir_path, xstr(manipulation_model))
                if not os.path.exists(mult_manupulation_dir):
                    os.makedirs(mult_manupulation_dir)

                new_tree_path = tree_path
                if xstr(manipulation_model) != '':
                    new_tree_path = os.path.join(mult_manupulation_dir, 'tree.newick')
                    create_rescaled_tree(manipulation_model, multiplier, nodes_file_path, tree_path, new_tree_path, expectation_file_path)

                job_name = 'sim_'
                job_name += 'homogenous' if not other_modes_flag else 'heterogenous' + xstr(sampling_frac)
                if other_modes_flag:
                    job_name += '_emp' if hetero_res_dir is not None else '_' + str(multiplier) + '_' + str(manipulation_model)
                
                other_kwargs['multiplier'] = multiplier
                other_kwargs['manipulated_rates'] = manipulations_on[manipulation_model]
                paramio(mult_manupulation_dir, job_name, counts_path, new_tree_path) \
                    .set_simulated(in_dir, num_of_simulations, **other_kwargs).output()

                create_job_file(mult_manupulation_dir, job_name, mem=10, ncpu=1, exe=CHROMEVOL_SIM_EXE, on=q_on)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simulates homogeneous and heterogeneous models')
    parser.add_argument('--in_dir', '-i', type=str,
        help='input directory; must include: tree.newick, counts.fasta, chromEvol.res, MLAncestralReconstruction.tree, [het] expectations_second_round.txt')
    parser.add_argument('--num_of_simulations', '-n', type=int, help='number of simulations')
    parser.add_argument('--model', '-m', type=str, help='adequate model')
    parser.add_argument('--out_dir', '-o', type=str, help='simulations directory')
    parser.add_argument('--sampling_fractions', '-s', type=float, nargs='+', help='clade size fractions')
    parser.add_argument('--multipliers', '-k', type=float, nargs = '+', help='rate multipliers')
    parser.add_argument('--manipulated_rates', '-r', type=str, nargs='+', help='manipulated rates')
    parser.add_argument('--empirical_hetero_params', '-e', default=None, help='chrom res file to get empirical heterogeneous model data')
    parser.add_argument('--q_on', '-q', action=argparse.BooleanOptionalAction, default=True, help='use False to only generate param & job files')
    
    main(parser.parse_args())
