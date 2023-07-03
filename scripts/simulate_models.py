import os
import argparse
import utils as utl
from defs import CHROMEVOL_SIM_EXE

def main(args):

    xstr = lambda s: '' if s is None else str(s)
    lstd = lambda l: [l] if not isinstance(l, list) else l

    in_dir = args.in_dir
    num_of_simulations = args.num_of_simulations
    multipliers = lstd(args.multipliers)
    sampling_fractions = lstd(args.sampling_fractions)
    manipulated_rates = lstd(args.manipulated_rates)
    hetero_res_dir = args.empirical_hetero_params
    standalone = args.standalone
    out_dir = utl.get_res_path(args.out_dir, standalone)

    other_kwargs = {}
    other_dir = out_dir
    other_modes_flag = False
    if [1.0] == multipliers: # homogenous simulation mode
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
                        'all':['demiPloidyR', 'dupl', 'baseNumR', 'gain', 'loss'], None:None}
    tree_path = os.path.join(in_dir, 'tree.newick')
    expectation_file_path = os.path.join(other_dir, 'expectations_second_round.txt')

    for sampling_frac in sampling_fractions:

        sampling_frac_dir = os.path.join(other_dir, xstr(sampling_frac))
        if not os.path.exists(sampling_frac_dir):
            os.makedirs(sampling_frac_dir)

        if other_modes_flag:
            nodes_file_path = os.path.join(sampling_frac_dir, 'nodes.txt')
            if not os.path.exists(nodes_file_path):
                utl.create_nodes_split_file(nodes_file_path, sampling_frac, tree_path)
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
                    utl.create_rescaled_tree(manipulation_model, multiplier, nodes_file_path, tree_path, new_tree_path, expectation_file_path)

                job_name = 'sim_'
                job_name += 'homogenous' if not other_modes_flag else 'heterogenous' + xstr(sampling_frac)
                if other_modes_flag:
                    job_name += '_emp' if hetero_res_dir is not None else '_' + str(multiplier) + '_' + str(manipulation_model)
                
                other_kwargs['multiplier'] = multiplier
                other_kwargs['manipulated_rates'] = manipulations_on[manipulation_model]
                # important: in sim mode dataFile is NOT counts_path, but final output dir!
                utl.paramio(mult_manupulation_dir, job_name, mult_manupulation_dir, new_tree_path) \
                    .set_simulated(in_dir, num_of_simulations, **other_kwargs).output()

                utl.do_job(mult_manupulation_dir, job_name, mem=10, ncpu=1, exe=CHROMEVOL_SIM_EXE, standalone=standalone)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simulates homogeneous and heterogeneous models')
    parser.add_argument('--in_dir', '-i', type=str,
        help='input directory; must include: tree.newick, counts.fasta, chromEvol.res, MLAncestralReconstruction.tree, [het] expectations_second_round.txt')
    parser.add_argument('--num_of_simulations', '-n', type=int, help='number of simulations')
    parser.add_argument('--out_dir', '-o', type=str, help='simulations directory')
    parser.add_argument('--multipliers', '-k', type=float, nargs='+', default=1.0, help='rate multipliers, default 1.0')
    parser.add_argument('--sampling_fractions', '-s', type=float, nargs='+', help='clade size fractions, default None')
    parser.add_argument('--manipulated_rates', '-r', type=str, nargs='+', help='manipulated rates, default None')
    parser.add_argument('--empirical_hetero_params', '-e', default=None, help='chrom res file to get empirical heterogeneous model data')
    parser.add_argument('--standalone', action='store_true', default=False, help='use standalone flag to generate job file')
    
    main(parser.parse_args())

# standalone example:
# python simulate_models.py -n 20 -i "/groups/itay_mayrose/ronenshtein/chrevodata/test2/Trees/family/phrymaceae/Homogenous/" -k 1
# -o "/groups/itay_mayrose/ronenshtein/chrevodata/test2/Trees/family/phrymaceae/Homogenous/" --standalone 