QUEUE = 'itaym'
# old version; simulates
CHROMEVOL_SIM_EXE = '/groups/itay_mayrose/ronenshtein/chrevodata/test_likelihood_chr'
# new version; doesn't simulate
CHROMEVOL_EMP_EXE = '/groups/itay_mayrose/ronenshtein/chromEvol/chromevol/ChromEvol/chromEvol'
# for example: 'module load python', 'module load python/python-anaconda3.7', or '' [if no conda activate]
MODULE_COMMAND = ''
# for example: 'conda activate env\nexport PATH=$CONDA_PREFIX/bin:$PATH', or '' [if no module load]
ACTIVATE_ENV = '''conda activate phylo
export PATH=$CONDA_PREFIX/bin:$PATH'''