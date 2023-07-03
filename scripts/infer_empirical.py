import os
import shutil
import argparse
from ete3 import Tree
from math import floor
import utils as utl

def main(args):

    in_dir = args.in_dir
    model = args.model
    mode = args.mode.capitalize()
    memcpu_ratio = args.memcpu_ratio
    req_multiplier = args.speed_multiplier
    standalone = args.standalone
    out_dir = utl.get_res_path(args.out_dir, standalone)

    for item in os.listdir(in_dir):
        if item.endswith('.newick'):
            name = os.path.splitext(item)[0]
            tree_file = os.path.join(in_dir, item)
            counts_file = os.path.join(in_dir, 'counts.fasta')
            
            taxa = utl.count_taxa(Tree(tree_file))
            req = floor(1+taxa/100)*req_multiplier

            if mode == 'Homogenous' or mode == 'Both':
                inf_dir = os.path.join(out_dir, 'Homogenous')
                utl.paramio(inf_dir, 'emp_'+name, counts_file, tree_file).set_empirical(model=model).output()
                shutil.copy(tree_file, inf_dir) # must be done after paramio (otherwise, no such folder)
                shutil.copy(counts_file, inf_dir)
                utl.do_job(inf_dir, 'emp_'+name, standalone=standalone)

            if mode == 'Heterogenous' or mode == 'Both':
                inf_dir = os.path.join(out_dir, 'Heterogenous')
                utl.paramio(inf_dir, 'emp_'+name, counts_file, tree_file).set_empirical(model=model, heterogenous=True, \
                    taxa_num=taxa, max_number_of_models=10).output()
                shutil.copy(tree_file, inf_dir) # must be done after paramio (otherwise, no such folder)
                shutil.copy(counts_file, inf_dir)
                utl.do_job(inf_dir, 'emp_'+name, ncpu=req, mem=int(req*memcpu_ratio), standalone=standalone)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='runs homogenenous and heterogenous inference on empirical data')
    parser.add_argument('--in_dir', '-i', type=str, help='path to trees with chromosome numbers')
    parser.add_argument('--out_dir', '-o', type=str, help='analyses results directory')
    parser.add_argument('--model', '-m', type=str, help='adequate model')
    parser.add_argument('--mode', '-d', type=str, default='both', help='homogenenous, heterogenous, or both')
    parser.add_argument('--memcpu_ratio', type=float, default=1.5, help='ratio between mem in gb and #cores to ask for the heterogenous model; def 1.5')
    parser.add_argument('--speed_multiplier', type=int, default=6, help='speed multiplier for the heterogenous model; def 6')
    parser.add_argument('--standalone', action='store_true', default=False, help='use standalone flag to generate job file')

    main(parser.parse_args())
