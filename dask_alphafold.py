# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
import queue
from typing import Dict, Union, Optional, List, Tuple
from bisect import bisect
from glob import glob
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import haiku

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.efficient_folding.load_models import load_models_and_params
from alphafold.efficient_folding.pad_inputs import pad_inputs
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
import numpy as np
import dask
from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_string(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_list(
    'is_prokaryote_list', None,
    'Optional for multimer system, not used by the '
    'single chain system. This list should contain a boolean for each fasta '
    'specifying true where the target complex is from a prokaryote, and false '
    'where it is not, or where the origin is unknown. These values determine '
    'the pairing method for the MSA.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string(
    'max_template_date', '9999-12-31', 'Maximum template release date '
    'to consider. Important if folding historical test sets.')
flags.DEFINE_enum(
    'db_preset', 'full_dbs', ['full_dbs', 'reduced_dbs'],
    'Choose preset MSA database configuration - '
    'smaller genetic database config (reduced_dbs) or '
    'full genetic database config  (full_dbs)')
flags.DEFINE_enum(
    'model_preset', 'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Choose preset model configuration - the monomer model, '
    'the monomer model with extra ensembling, monomer model with '
    'pTM head, or multimer model')
flags.DEFINE_boolean(
    'benchmark', False, 'Run multiple JAX model evaluations '
    'to obtain a timing that excludes the compilation time, '
    'which should be more indicative of the time required for '
    'inferencing many proteins.')
flags.DEFINE_integer(
    'random_seed', None, 'The random seed for the data '
    'pipeline. By default, this is randomly generated. Note '
    'that even if this is set, Alphafold may still not be '
    'deterministic, because processes like GPU inference are '
    'nondeterministic.')
flags.DEFINE_boolean(
    'use_precomputed_msas', False, 'Whether to read MSAs that '
    'have been written to disk. WARNING: This will not check '
    'if the sequence, database or configuration have changed.')
flags.DEFINE_integer('recycles', 3, '')
flags.DEFINE_integer('parallel', 1, '')
flags.DEFINE_integer('cpu_nodes', 10, '')
flags.DEFINE_integer('gpu_nodes', 1, '')
flags.DEFINE_integer('gpu_jobs', 5, '')
flags.DEFINE_integer('cpu', 3, '')
flags.DEFINE_boolean('no_amber', False, '')
flags.DEFINE_boolean('no_msa', False, '')

FLAGS = flags.FLAGS

ROCKFISH_CPU_CORE_PER_NODE = 48
ROCKFISH_CPU_MEM_PER_CORE = 4
ROCKFISH_GPU_CORE_PER_NODE = 12
ROCKFISH_GPU_MEM_PER_CORE = 4


def migrate_data(data_dir, local_disk):
    local_data_dir = os.path.join(local_disk, os.path.split(data_dir)[1])
    logging.log(logging.INFO,
                f"Extracting data from {data_dir} to {local_data_dir}")
    if not os.path.isdir(local_data_dir):
        os.system(f"tar -xf {data_dir} -C {local_disk}")


def preprocess_sequence(args):
    fasta_file, output_dir, data_dir, model_preset, cpu, no_amber, no_msa, recycles = args
    # migrate_data(data_dir, "/tmp/")

    preprocess_command = f"""
        python run_alphafold.py
        --fasta_paths {fasta_file}
        --output_dir {output_dir}
        --data_dir {data_dir}
        --model_preset {model_preset}
        --cpu {cpu}
        --no_amber {no_amber}
        --no_msa {no_msa}
        --recycles {recycles}
        --preprocess
    """.replace("\n", " ")

    logging.log(logging.INFO, f"Running {preprocess_command}")
    os.system(preprocess_command)

    fasta_name = os.path.splitext(os.path.basename(fasta_file))[0]
    result_pkl = os.path.join(output_dir, fasta_name, "features.pkl")
    if os.path.exists(result_pkl):
        return args, True
    else:
        return args, False


def predict_structure(args):
    fasta_file, output_dir, data_dir, model_preset, cpu, no_amber, no_msa, recycles = args

    predict_command = f"""
        python run_alphafold.py
        --fasta_paths {fasta_file}
        --output_dir {output_dir}
        --data_dir {data_dir}
        --model_preset {model_preset}
        --cpu {cpu}
        --no_amber {no_amber}
        --no_msa {no_msa}
        --recycles {recycles}
    """.replace("\n", " ")

    logging.log(logging.INFO, f"Running {predict_command}")
    os.system(predict_command)

    fasta_name = os.path.splitext(os.path.basename(fasta_file))[0]
    result_pdb = os.path.join(output_dir, fasta_name, "ranked_0.pdb")
    if os.path.exists(result_pdb):
        return True
    else:
        logging.log(logging.INFO, f"{fasta_file} failed prediction")
        return False


def main(argv):
    # Check for duplicate FASTA file names.
    _fasta_paths = FLAGS.fasta_paths
    fasta_paths = []
    if os.path.isdir(_fasta_paths):
        fasta_paths.extend(list(glob(os.path.join(_fasta_paths, "*.fasta"))))
    else:
        fasta_paths.append(_fasta_paths)

    fasta_paths = list(sorted(fasta_paths))
    fasta_paths = [
        fp for fp in fasta_paths if not os.path.exists(
            os.path.join(FLAGS.output_dir,
                         os.path.split(fp)[1][:-6], "ranked_0.pdb"))
    ]

    fasta_names = [pathlib.Path(p).stem for p in fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')

    user_home_dir = "/home/jruffol1"
    scratch_dir = os.path.join(user_home_dir, "scratch")
    os.system("mkdir {}".format(scratch_dir))

    # cpu_processes = FLAGS.cpu_nodes * ROCKFISH_CPU_CORE_PER_NODE // FLAGS.cpu
    cpu_processes = ROCKFISH_CPU_CORE_PER_NODE // FLAGS.cpu
    cpu_memory = f"{ROCKFISH_CPU_MEM_PER_CORE * FLAGS.cpu}GB"
    cpu_cluster = SLURMCluster(
        cores=ROCKFISH_CPU_CORE_PER_NODE,
        processes=cpu_processes,
        memory=cpu_memory,
        queue="defq",
        local_directory=scratch_dir,
        walltime="40:00:00",
        job_extra=["-o {}".format(os.path.join(scratch_dir, "slurm-%j.out"))],
    )
    print(cpu_cluster.job_script())
    cpu_cluster.scale(FLAGS.cpu_nodes)
    cpu_client = Client(cpu_cluster)

    # gpu_cpus = ROCKFISH_CPU_CORE_PER_NODE // FLAGS.gpu_jobs
    # gpu_memory = f"{ROCKFISH_GPU_MEM_PER_CORE * gpu_cpus}GB"
    # gpu_cluster = SLURMCluster(
    #     cores=gpu_cpus,
    #     processes=FLAGS.gpu_jobs,
    #     memory=gpu_memory,
    #     queue="a100",
    #     local_directory=scratch_dir,
    #     walltime="40:00:00",
    #     job_extra=[
    #         "--account=jgray21_gpu --gres=gpu:1 -o {}".format(
    #             os.path.join(scratch_dir, "slurm-%j.out"))
    #     ],
    # )
    # print(cpu_cluster.job_script())
    # gpu_cluster.scale(FLAGS.gpu_nodes)
    # gpu_client = Client(gpu_cluster)

    cpu_args = [
        (fasta_file, FLAGS.output_dir, FLAGS.data_dir, FLAGS.model_preset,
         FLAGS.cpu, FLAGS.no_amber, FLAGS.no_msa, FLAGS.recycles)
        for fasta_file in fasta_paths
    ]
    preprocess_results = cpu_client.map(preprocess_sequence, cpu_args)
    # print(cpu_client.gather(preprocess_results))
    # [preprocess_sequence(args) for args in cpu_args]

    # for fasta_file in tqdm(fasta_paths, total=len(fasta_paths)):
    #     cpu_result = cpu_client.submit(preprocess_sequence, cpu_args)
    #     preprocess_results.append(cpu_result)

    # predict_queue = queue.Queue()
    # [
    #     predict_queue.put((f, r))
    #     for f, r in zip(fasta_paths, preprocess_results)
    # ]
    # prediction_results = []
    # pbar = tqdm(total=len(fasta_paths))
    # while not predict_queue.empty():
    #     time.sleep(0.1)
    #     fasta_file, preprocess_result = predict_queue.get()
    #     if preprocess_result.done():
    #         gpu_args = (fasta_file, FLAGS.output_dir, FLAGS.data_dir,
    #                     FLAGS.model_preset, FLAGS.cpu, FLAGS.no_amber,
    #                     FLAGS.no_msa, FLAGS.recycles)
    #         gpu_result = gpu_client.submit(predict_structure, gpu_args)
    #         prediction_results.append(gpu_result)
    #         pbar.update(1)
    #     else:
    #         predict_queue.put((fasta_file, preprocess_result))

    for batch in as_completed(preprocess_results, with_results=True).batches():
        for _, (gpu_args, success) in batch:
            print(success)
            if not success:
                logging.log(logging.INFO, f"{gpu_args[0]} failed prediction")

    #     gpu_args = [a for a in gpu_args if a[0]]
    #     predict_structure(gpu_args)

    # wait(prediction_results)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        'data_dir',
        # 'uniref90_database_path',
        # 'mgnify_database_path',
        # 'template_mmcif_dir',
        # 'max_template_date',
        # 'obsolete_pdbs_path',
    ])

    app.run(main)
