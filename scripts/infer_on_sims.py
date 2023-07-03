import os
import random
import argparse
from math import floor
from utils import paramio, get_res_path, do_job

def main(args):

    sims_dir = args.sims_dir
    tree_path = args.tree_path
    model = args.model
    simulation_range = list(map(int, args.simulation_range.split(':')))
    randomize = args.randomize
    seed = args.seed
    min_clade = args.min_clade
    memcpu_ratio = args.memcpu_ratio
    req_multiplier = args.speed_multiplier
    standalone = args.standalone
    out_dir = get_res_path(args.out_dir, standalone)

    rng = range( * simulation_range[:min(len(simulation_range), 3)] )
    if randomize <= 0 or randomize >= len(rng):
        pass
    else:
        if seed is not None:
            random.seed(seed)
    rng = random.sample(rng, randomize)
 
    req = 0 # flag

    for _ in rng:
        sim_dir = os.path.join(sims_dir, str(_))
        counts_path = os.path.join(sim_dir, 'counts.fasta')

        # get number of taxa in the tree, to approx cpu & memory requirements
        if not req:
            with open(counts_path, 'r') as file:
                taxa = sum(1 for _ in file) // 2
            req = floor(1+taxa/100)*req_multiplier

        hom_str = str(_)+'_Homogenous'
        het_str = str(_)+'_Heterogenous'

        hom_dir = os.path.join(out_dir, hom_str)
        het_dir = os.path.join(out_dir, het_str)

        paramio(hom_dir, 'emp_'+hom_str, counts_path, tree_path).set_empirical(model=model).output()
        paramio(het_dir, 'emp_'+het_str, counts_path, tree_path).set_empirical(model=model, heterogenous=True, \
            taxa_num=taxa, min_clade_size=min_clade, max_number_of_models=10).output()

        do_job(hom_dir, 'emp_'+hom_str, standalone=standalone)
        do_job(het_dir, 'emp_'+het_str, ncpu=req, mem=int(req*memcpu_ratio), standalone=standalone)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='runs homogenenous and heterogenous inference on simulated chromosome data')
    parser.add_argument('--sims_dir', '-i', type=str, help='simulations directory')
    parser.add_argument('--tree_path', '-t', type=str, help='path to original tree; newick format')
    parser.add_argument('--out_dir', '-o', type=str, help='analyses results directory')
    parser.add_argument('--model', '-m', type=str, help='adequate model')
    parser.add_argument('--simulation_range', '-r', type=str, help='range from which to pick, required arg')
    parser.add_argument('--randomize', '-z', default=0, type=int, help='range for selecting random sims; def 0')
    parser.add_argument('--seed', '-s', type=int, default=None, help='seed for randomization')
    parser.add_argument('--min_clade', type=float, default=0, help='min clade size override for the heterogenous model')
    parser.add_argument('--memcpu_ratio', type=float, default=1.5, help='ratio between mem in gb and #cores to ask for the heterogenous model; def 1.5')
    parser.add_argument('--speed_multiplier', type=int, default=6, help='speed multiplier for the heterogenous model; def 6')
    parser.add_argument('--standalone', action='store_true', default=False, help='use standalone flag to generate job file')

    main(parser.parse_args())