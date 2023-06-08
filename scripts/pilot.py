import argparse
import simulate_models
import infer_on_sims

path = r'C:\Users\ItayMNB03\source\repos\chromevol\scripts'

simulate_models.main( argparse.Namespace(num_of_simulations=120,
                   in_dir=path+r'\DAT', model='g+L_l+L_du+E_de+C_b+C',
                   sampling_fractions=[0.5, 0.2], multipliers=2.0, manipulated_rates='all',
                   empirical_hetero_params=None, out_dir=path+r'\RES', q_on=False) )

infer_on_sims.main( argparse.Namespace(num_of_simulations=120,
                   sims_dir=path+r'\SIM', tree_path=path+r'\SIM\tree.newick',
                   simulation_range='1:7', randomize=4, seed=None, min_clade=0,
                   memcpu_ratio=1.5, speed_multiplier=6, out_dir=path+r'\OUT', q_on=False) )


