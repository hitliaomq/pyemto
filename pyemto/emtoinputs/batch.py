# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:09:24 2014

@author: Matti Ropo
@author: Henrik Levämäki

"""

from __future__ import print_function
import sys
import os
import datetime
import re
import pyemto.common.common as common
from pyemto.latticeinputs.batch import batch_head
from monty.os.path import which

class Batch:
    """Creates a batch script for running KGRN and KFCD calculations on a
    supercomputer environment (EMTO 5.8).

    !!! Currently only SLURM is supported. !!!

    :param jobname: Name for the KGRN and KFCD jobs. This will become
                    the first part of the input and output file names.
    :type jobname: str
    :param runtime: Maximum running time for the individual batch jobs.
                    The format of this entry should be 'xx:yy:zz', where
                    xx is hours, yy is minutes and zz is seconds.
    :type runtime: str
    :param EMTOdir: Path to the EMTO installation (Default value = '$HOME/EMTO5.8')
    :type EMTOdir: str
    :param emtopath: Path to the folder where the KGRN and KFCD input files are
                     located
    :type emtopath: str
    :param runKGRN: True if KGRN should be run, False if KGRN should not be run
    :type runKGRN: boolean
    :param runKFCD: True if KFCD should be run, False if KFCD should not be run
    :type runKFCD: boolean
    :returns: None
    :rtype: None
    """

    def __init__(self, jobname=None, runtime=None, EMTOdir=None,
                 emtopath=None, runKGRN=None, runKFCD=None,
                 account="open", KGRN_file_type="scf", KFCD_file_type="fcd",
                 slurm_options=None, parallel=None, queue_type="pbs", 
                 queue_options={"node": 1, "ncore": 24, "pmem": "8gb", "module": ["intel/16.0.3", "mkl"]}):

        # Batch script related parameters
        self.jobname = jobname
        self.runtime = runtime
        self.emtopath = emtopath
        self.EMTOdir = EMTOdir
        self.runKGRN = runKGRN
        self.runKFCD = runKFCD
        self.account = account
        if KGRN_file_type is not None:
            self.KGRN_file_type = KGRN_file_type
        else:
            self.KGRN_file_type = 'kgrn'
        if KFCD_file_type is not None:
            self.KFCD_file_type = KFCD_file_type
        else:
            self.KFCD_file_type = 'kfcd'
        self.slurm_options = slurm_options
        self.parallel = parallel
        self.queue_type = queue_type.lower()
        self.queue_options = queue_options
        self.use_module = False
        return

    def output(self):
        """
        Output first part of the kgrn input file in formated string

        :returns: Batch job script file in the form of a long string
        :rtype: str
        """
        queue_type = self.queue_type
        queue_options = self.queue_options
        jobname = self.jobname
        emtopath = self.emtopath

        line = batch_head(jobname=jobname, latpath=emtopath, runtime=self.runtime, 
                 account=self.account, queue_type=queue_type, queue_options=queue_options)
        
        self.use_module = False
        if self.slurm_options is not None:
            for tmp in self.slurm_options:
                if 'module load emto' in tmp:
                    self.use_module = True
                    break
        line += "\n"

        sub_module = ["kgrn_cpa", "kfcd_cpa"]
        sub_module_run = [self.runKGRN, self.runKFCD]
        file_type = [self.KGRN_file_type, self.KFCD_file_type]
        output_file_ext = ["kgrn", "kfcd"]

        #elapsed_time = "/usr/bin/time "
        elapsed_time = ""
        if self.parallel is True:
            sub_module = ["kgrn_omp", "kfcd_cpa"]

        if which("kfcd_cpa") is not None:
            self.use_module = True
        if not self.use_module:
            module_path = [os.path.join(self.EMTOdir, module_i) for module_i in sub_module]
        else:
            module_path = sub_module

        for i in range(0, len(sub_module)):
            if sub_module_run[i]:
                runStr = [elapsed_time, module_path[i], "<", 
                         os.path.join(emtopath, jobname + "." + file_type[i]), ">",
                         os.path.join(emtopath, jobname + "_" + output_file_ext[i] + ".output")]
                line += " ".join(runStr).strip() + "\n"

        return line

    def write_input_file(self, folder=None):
        """(self,str) ->(None)

            Save BMDL input data to file named filename

        :param folder:  (Default value = None)
        :type folder:
        :returns:
        :rtype:
        """

        # Check data integrity before anything is written on disk or run
        self.check_input_file()

        if folder is None:
            #sys.exit('Batch_emto.write_input_file: \'folder\' has to be given!')
            folder = "./"
        else:
            common.check_folders(folder)

        fl = open(folder + '/{0}.'.format(self.jobname) + self.queue_type, "w")
        fl.write(self.output())
        fl.close()
        return

    def set_values(self, key, value):
        """

        :param key:
        :type key:
        :param value:
        :type value:
        :returns:
        :rtype:
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print('WARNING: Batch_emto() class has no attribute \'{0}\''.format(key))
        return

    def check_input_file(self):
        """Perform various checks on the class data to
            make sure that all necessary data exists
            before we attempt to write the input file to disk
        """

        # Mission critical parameters
        if self.jobname is None:
            sys.exit('Batch_emto.check_input_file: \'jobname\' has to be given!')

        if self.runtime is None:
            self.runtime = "48:00:00"
        if self.emtopath is None:
            self.emtopath = "./"
        if self.EMTOdir is None:
            self.EMTOdir = "$HOME/EMTO5.8/"
        if self.runKGRN is None:
            self.runKGRN = True
        if self.runKFCD is None:
            self.runKFCD = True
        return
