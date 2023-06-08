'''
Hi Keren,
Now you can pull the updates from bpp-phyl.
You have to add the following parameters:
_numOfSimulatedData: the maximum number of datasets to simulate (i.e., the total number of trials). The default number is 1.
_fracAllowedFailedSimulations: fraction of simulations that are allowed to reach max chromosome number. The default is 0.01. For example if _numOfSimulatedData = 1000, ChromEvol will stop simulating after 10 failures, and throw an exception. For your usage, I think you can set it to 1, because you care only about successes.
_numOfRequiredSimulatedData: the number of successful simulations you want to achieve. The default value is 1. If for example _numOfRequiredSimulatedData = 100, _numOfSimulatedData = 1000, you have 1000 possible trials, but the simulations will end when reaching 100 successful simulations. However, the total amount of simulations can be larger than 100, because it might also contain the failed simulations. So, for example, if you had 5 failures before reaching 100 successful simulations, you will have 105 simulations in your directory. Among these simulations you should remove the unsuccessful ones.
Also, note that now _dataFile should be defined as the folder where you intend to store your simulations. ChromEvol will create a folder for each simulation inside this directory (from 0 to the n-1, where n is the of the generated simulations). You can also use my python script to remove all the unsuccessful simulations. This script also reorders the simulations folders, so that if for example, the successful simulations were 0,3,5, now they will be renamed to 0,1,2. The script is attached below. You should use it as following:
python retain_successful_simulations.py -n <number of required simulations> -m <max chromosome number> -d <the directory of all the simulations>
Hope it is helpful :)
'''

import re
import os
import argparse
import shutil


def remove_unsuccessful_simulations(sim_dir, max_state):
    sim_folders = os.listdir(sim_dir)
    for i in sim_folders:
        sim_dir_i = os.path.join(sim_dir, str(i))
        if (not os.path.exists(sim_dir_i)) or (not os.path.isdir(sim_dir_i)):
            continue
        evol_path = os.path.join(sim_dir_i, "simulatedEvolutionPaths.txt")
        if not os.path.exists(evol_path):
            continue
        max_state_evol = get_max_state_transition(evol_path)
        if max_state_evol == max_state:
            shutil.rmtree(sim_dir_i)


def get_max_state_transition(evol_path):
    file = open(evol_path, 'r')
    content = file.read()
    file.close()
    pattern_state_to_state = re.compile("from state:[\s]+([\d]+)[\s]+t[\s]+=[\s]+[\S]+[\s]+to state[\s]+=[\s]+([\d]+)")
    states = pattern_state_to_state.findall(content)
    max_state = 0
    for state_from, state_to in states:
        if int(state_from) > max_state:
            max_state = int(state_from)
        if int(state_to) > max_state:
            max_state = int(state_to)
        else:
            continue
    return max_state


def pick_successful_simualtions(sim_dir, num_of_simulations):
    dirs = os.listdir(sim_dir)
    lst_of_dirs = []
    for file in dirs:
        sim_i_dir = os.path.join(sim_dir, file)
        if not os.path.isdir(sim_i_dir):
            continue
        lst_of_dirs.append(int(file))
    lst_of_dirs.sort()
    for i in range(num_of_simulations):
        curr_dir = os.path.join(sim_dir, str(lst_of_dirs[i]))
        dst = os.path.join(sim_dir, str(i))
        if i != lst_of_dirs[i]:
            os.rename(curr_dir, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer ChromEvol models')
    parser.add_argument('--num_of_simulations', '-n', type=int, help='number of simulations')
    parser.add_argument('--max_chr_state', '-m', type=int, help='output file path of stats')
    parser.add_argument('--sim_dir', '-d', help='simulation directory')

    # parse arguments
    args = parser.parse_args()
    num_of_simulations = args.num_of_simulations
    max_chr_state = args.max_chr_state
    sim_dir = args.sim_dir
    remove_unsuccessful_simulations(sim_dir, max_chr_state)
    pick_successful_simualtions(sim_dir, num_of_simulations)



