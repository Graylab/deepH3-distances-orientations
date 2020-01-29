import argparse
from os import listdir
from os.path import join, getsize
from deeph3.preprocess.generate_h5_pnet_files import pnet_to_h3


desc = 'Creates h5 files from all the ProteinNet text files in a directory'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('protein_net_file_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--num_evo_entries', type=int,
                    help='Number of evolutionary entries (features) in '
                         'a ProteinNet file, default is 20.',
                    default=20)
parser.add_argument('--overwrite', type=bool,
                    help='Whether or not to overwrite a file or not,'
                         ' if it exists',
                    default=False)
args = parser.parse_args()
pnet_dir = args.protein_net_file_dir
output_dir = args.output_dir

files = [str(_) for _ in listdir(pnet_dir)]
files.sort(key=lambda file: getsize(join(pnet_dir, file)))
print('Files to convert: {}'.format(files))
for f in files:
    print('Creating h5 file for {}...'.format(f))
    output_file = join(output_dir, f.split('.')[0] + '.h5')
    pnet_to_h3(join(pnet_dir, f), output_file,
               num_evo_entries=args.num_evo_entries,
               overwrite=args.overwrite,
               print_progress=True)
    print('Wrote {}'.format(output_file))
