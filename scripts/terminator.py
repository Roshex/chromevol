import os
import argparse
from utils import read_write_append

def main(args):

    path = os.path.dirname(os.path.dirname(args.data))
    read_write_append(path, ['failed_outputs.txt', 'completed_outputs.txt'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tracks completed runs')
    parser.add_argument('--data', '-d', type=str, help='path to summary csv file')

    main(parser.parse_args())
