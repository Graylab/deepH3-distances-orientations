#!/usr/bin/env python
#
# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @file   create_antibody_db.py
## @brief  Script to create database for RosettaAntibody
## @author Jeliazko Jeliazkov

## @details download non-redundant Chothia Abs from SAbDab
## Abs are downloaded by html query (is there a better practice)?
## Abs are Chothia-numbered, though we use Kabat to define CDRs.
## After download, trim Abs to Fv and extract FR and CDR sequences.
## For the purpose of trimming, truncate the heavy @112 and light @109
## Some directories (antibody_database, info, etc...) are hard coded.

import os
import sys
import requests
import pandas as pd
import argparse
from deeph3.util import RawTextArgumentDefaultsHelpFormatter
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_sabdab_summary(file_name):
    """
    SAbDab produces a rather unique summary file.
    This function reads that file into a dict with the key being the
    4-letter PDB code.
    """
    # dict[pdb] = {col1 : value, col2: value, ...}
    sabdab_dict = {}

    with open(file_name, "r") as f:
        # first line is the header, or all the keys in our sub-dict
        header = f.readline().strip().split("\t")
        # next lines are data
        for line in f.readlines():
            split_line = line.strip().split("\t")
            td = {} # temporary dict of key value pairs for one pdb
            for k,v in zip(header[1:], split_line[1:]):
                # pdb id is first, so we skip that for now
                td[k] = v
            # add temporary dict to sabdab dict at the pdb id
            sabdab_dict[split_line[0]] = td

    return sabdab_dict


def truncate_chain(pdb_text, chain, resnum, newchain):
    """
    Read PDB line by line and return all lines for a chain,
    with a resnum less than or equal to the input.
    This has to be permissive for insertion codes.
    This will return only a single truncated chain.
    This function can update chain to newchain.
    """
    trunc_text = ""
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[21] == chain and int(line[22:26]) <= resnum:
            trunc_text += line[:21] + newchain + line[22:]
            trunc_text += "\n"
    return trunc_text


def truncate_antibody_pdbs(sabdab_dir=None):
    """
    We only use the Fv as a template, so this function loads each pdb
    and deletes excess chains/residues. We define the Fv, under the
    Chothia numbering scheme as H1-H112 and L1-L109.
    """
    # count warnings
    warn_pdbs = []

    # count delete files (without sabdab info)
    remove_pdbs = []

    # count deleted files (VH+VL same chain)
    same_chain = []

    if sabdab_dir is None:
        create_antibody_db_py_dir = os.path.dirname(os.path.realpath(__file__))
        sabdab_dir = str(Path(create_antibody_db_py_dir).parent.joinpath('data'))

    # read SAbDab info for chain identities
    sabdab_dict = parse_sabdab_summary("info/sabdab_summary.tsv")

    # iterate over PDBs in antibody_database and truncate accordingly
    #unique_pdbs = set([x[:4] for x in os.listdir("antibody_database") if x.endswith(".pdb")])
    unique_pdbs = set([x[:4] for x in os.listdir("antibody_database") if x.endswith(".pdb")])

    print('Truncating pdb files...')
    for pdb in tqdm(unique_pdbs):

        # check if "pdb_trunc.pdb" exits, if not then generate it
        if os.path.isfile("antibody_database/" + pdb + "_trunc.pdb"):
            print("We think a truncated version of " + pdb + " was found. Skipping.")
            os.remove("antibody_database/" + pdb + ".pdb") # delete old file just in case
            continue

        pdb_text = ""
        try:
            with open("antibody_database/" + pdb + ".pdb", "r") as f:
                pdb_text = f.read() # want string not list
        except IOError:
            sys.exit("Failed to open {} in antibody_database/ !".format(pdb))

        # should have pdb_text now
        if len(pdb_text) == 0: sys.exit("Nothing parsed for PDB {} !".format(pdb))

        # test if pdb is in sabdab summary file, if not skip and delete PDB from db
        try:
            sabdab_dict[pdb]
        except KeyError:
            remove_pdbs.append(pdb)
            print(pdb + " not in sabdab summary file, removing ...")
            os.remove("antibody_database/" + pdb + ".pdb")
            continue

        hchain = sabdab_dict[pdb]["Hchain"]
        hchain_text = ""
        lchain = sabdab_dict[pdb]["Lchain"]
        lchain_text = ""

        # we do not currently have a good way of handling VH & VL on the same chain
        if hchain == lchain:
            same_chain.append(pdb)
            print(pdb + " has the VH+VL on a single chain, removing...")
            os.remove("antibody_database/" + pdb + ".pdb")
            continue

        if not hchain == "NA":
            hchain_text = truncate_chain(pdb_text, hchain, 112, "H")
            if len(hchain_text) == 0:
                # could not find heavy chain -- do not overwrite, but warn!
                warn_pdbs.append(pdb)
                print("Warning, could not find " + hchain + " chain for " + pdb + " !")
                print("It was not reported to be NA, so the file may have been altered!")
                continue

        if not lchain == "NA":
            lchain_text = truncate_chain(pdb_text, lchain, 109, "L")
            if len(lchain_text) == 0:
                # could not find heavy chain -- do not overwrite, but warn!
                warn_pdbs.append(pdb)
                print("Warning, could not find " + lchain + " chain for " + pdb + " !")
                print("It was not reported to be NA, so the file may have been altered!")
                continue

        # write new file to avoid bugs from multiple truncations
        with open("antibody_database/" + pdb+"_trunc.pdb", "w") as f:
            f.write(hchain_text + lchain_text)

        # remove old file
        os.remove("antibody_database/" + pdb + ".pdb")

    for pdb in remove_pdbs:
        print("Deleted " + pdb + " from database because it is missing from summary file")

    for pdb in same_chain:
        print("Deleted " + pdb + " from database because it has VH+VL on the same chain.")
    print("Deleted {} total of same chain VH+VLs.".format(len(same_chain)))

    if len(warn_pdbs) > 0:
        print("Finished truncating, with {} warnings.".format(len(warn_pdbs)))
        #sys.exit("Exiting prematurely due to warnings.")

    return


def download_file(url, output_path):
    with open(output_path, 'w') as f:
        f.write(requests.get(url).content.decode('utf-8'))


def download_chothia_pdb_files(summary_file='info/sabdab_summary.tsv',
                               antibody_database_dir='antibody_database',
                               max_workers=16):
    download_url = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=chothia'
    summary_dataframe = pd.read_csv(summary_file, sep='\t')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        unique_pdbs = summary_dataframe['pdb'].unique()
        pdb_file_paths = [os.path.join(antibody_database_dir, pdb + '.pdb') for pdb in unique_pdbs]
        urls = [download_url.format(pdb) for pdb in unique_pdbs]
        results = [executor.submit(lambda a: download_file(*a), args) for args in zip(urls, pdb_file_paths)]
        print('Downloading chothia files to {}/ from {} ...'.format(
            antibody_database_dir, download_url))
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass


def download_sabdab_summary_file(seqid=99, paired=True, nr_complex='All', nr_rfactor='', nr_res=3,
                                 summary_file='info/sabdab_summary.tsv'):
    base_url = 'http://opig.stats.ox.ac.uk'
    search_url = base_url + '/webapps/newsabdab/sabdab/search/'
    params = dict(seqid=seqid, paired=paired, nr_complex=nr_complex,
                  nr_rfactor=nr_rfactor, nr_res=nr_res)
    query = requests.get(search_url, params=params)
    html = BeautifulSoup(query.content, 'html.parser')
    summary_file_url = base_url + html.find(id='downloads').find('a').get('href')
    print('Downloading sabdab summary to {} from: {} ...'.format(
        summary_file, summary_file_url))
    download_file(summary_file_url, summary_file)


def download_chothias(**kwargs):
    # Change the working directory to the deeph3/data directory
    cur_dir = os.getcwd()
    create_antibody_db_py_dir = os.path.dirname(os.path.realpath(__file__))
    sabdab_dir = Path(create_antibody_db_py_dir).parent.joinpath('data')
    os.chdir(sabdab_dir)

    # Make required directories. If they already exist, delete all contents
    required_dirs = ['antibody_database', 'info']
    for dir_ in required_dirs:
        if os.path.isdir(dir_):
            print('Clearing {} directory...'.format(dir_))
            for file in os.listdir(dir_):
                os.remove(os.path.join(dir_, file))
        else:
            print('Making {} directory...'.format(dir_))
            os.mkdir(dir_)
    
    download_sabdab_summary_file(**kwargs)
    download_chothia_pdb_files()
    truncate_antibody_pdbs()

    os.chdir(cur_dir)  # Go back to original working dir


def _cli():
    desc = (
        '''
        Downloads chothia files from SabDab
        ''')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument('--seqid', type=int, default=99)
    parser.add_argument('--paired', type=bool, default=True)
    parser.add_argument('--nr_complex', type=str, default='All')
    parser.add_argument('--nr_rfactor', type=str, default='')
    parser.add_argument('--nr_res', type=int, default=3)
    args = parser.parse_args()

    download_chothias(seqid=args.seqid, paired=args.paired, nr_complex=args.nr_complex,
                      nr_rfactor=args.nr_rfactor, nr_res=args.nr_res)


if __name__ == '__main__':
    _cli()

