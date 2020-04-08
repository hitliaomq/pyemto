# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:09:24 2014

@author: Matti Ropo
@author: Henrik Levämäki

"""

from __future__ import print_function
import sys
import os
import pyemto.common.common as common
from monty.os.path import which

def batch_head(jobname, latpath="./", runtime="24:00:00", account="open", queue_type="pbs", 
    queue_options={"node": 1, "ncore": 24, "pmem": "8gb", "module": ["intel/16.0.3", "mkl"]}):
    
    line = "#!/bin/bash" + "\n" + "\n"
    if queue_type == "pbs":
        pbsjobname = jobname
        if len(pbsjobname) > 15:
            pbsjobname = pbsjobname[0:15]
        line += "#PBS -N " + pbsjobname + "\n"
        line += "#PBS -l nodes=1:ppn=1\n"
        line += "#PBS -l walltime=" + runtime + "\n"
        line += "#PBS -l pmem=" + queue_options["pmem"] + "\n"
        line += "#PBS -A {0}".format(account) + "\n"
        line += "#PBS -q open\n"
        line += "#PBS -j oe \n\n"
        line += "cd $PBS_O_WORKDIR"
        line += "\n"
        for dep_module in queue_options["module"]:
            line += "module load " + dep_module + "\n"
    elif queue_type == "slurm":
        line += "#SBATCH -J " + jobname + "\n"
        line += "#SBATCH -t " + runtime + "\n"
        line += "#SBATCH -o " + \
            common.cleanup_path(
                latpath + "/" + jobname) + ".output" + "\n"
        line += "#SBATCH -e " + \
            common.cleanup_path(
                latpath + "/" + jobname) + ".error" + "\n"
        if account is not None:
            line += "#SBATCH -A {0}".format(account) + "\n"
        if queue_options is not None:
            for so in queue_options:
                # Do not use more than one CPU for the structure calculations
                if "#SBATCH -n " in so:
                    pass
                else:
                    line += so + "\n"
    return line

class Batch:
    """Creates a batch script for running BMDL, KSTR and SHAPE calculations

    This class is used to to create batch scripts for a supercomputer environment (EMTO 5.8).
    !!! Currently only SLURM is supported. !!!

    :param jobname_lat:  (Default value = None)
    :type jobname_lat:
    :param lat:  (Default value = None)
    :type lat:
    :param runtime:  (Default value = None)
    :type runtime:
    :param latpath:  (Default value = None)
    :type latpath:
    :param EMTOdir:  (Default value = None)
    :type EMTOdir:
    :param runBMDL:  (Default value = None)
    :type runBMDL:
    :param runKSTR:  (Default value = None)
    :type runKSTR:
    :param runKSTR2:  (Default value = None)
    :type runKSTR2:
    :param runSHAPE:  (Default value = None)
    :type runSHAPE:
    :param kappaw:  (Default value = None)
    :type kappaw:
    :param kappalen:  (Default value = None)
    :type kappalen:
    :returns: None
    :rtype: None
    """

    def __init__(self, jobname_lat=None, lat=None, runtime=None, latpath=None,
                 EMTOdir=None, runBMDL=None, runKSTR=None, runKSTR2=None,
                 runSHAPE=None, kappaw=None, kappalen=None,
                 slurm_options=None, account="open", queue_type="pbs", 
                 queue_options={"node": 1, "ncore": 24, "pmem": "8gb", "module": ["intel/16.0.3", "mkl"]}):

        # Batch script related parameters
        self.jobname_lat = jobname_lat
        self.lat = lat
        self.latpath = latpath
        self.runtime = runtime
        self.EMTOdir = EMTOdir
        self.runBMDL = runBMDL
        self.runKSTR = runKSTR
        self.runKSTR2 = runKSTR2
        self.runSHAPE = runSHAPE
        self.kappaw = kappaw
        self.kappalen = kappalen
        self.account = account
        self.slurm_options = slurm_options
        self.queue_type = queue_type.lower()
        self.queue_options = queue_options
        self.use_module = False
        #print('BMDL self.slurm_options = ',self.slurm_options)

    def output(self):
        """(self) -> (str)

            Output first part of the kgrn input file in formated string

        :returns:
        :rtype:
        """

        # Clean up path names

        queue_type = self.queue_type
        queue_options = self.queue_options
        jobname = self.jobname_lat
        latpath = self.latpath

        line = batch_head(jobname=jobname, latpath=latpath, runtime=self.runtime, 
                 account=self.account, queue_type=queue_type, queue_options=queue_options)

        sub_module = ["bmdl", "kstr", "kstr", "shape"]
        sub_module_run = [self.runBMDL, self.runKSTR, self.runKSTR2, self.runSHAPE]
        jobname_m = [jobname, jobname, jobname + "M", jobname]

        self.use_module = False
        if self.slurm_options is not None:
            for tmp in self.slurm_options:
                if 'module load emto' in tmp:
                    self.use_module = True
                    break
        if which("bmdl") is not None:
            self.use_module = True
        line += "\n"

        #elapsed_time = "/usr/bin/time "
        elapsed_time = ""

        if not self.use_module:
            module_path = [os.path.join(self.EMTOdir, module_i) for module_i in sub_module]
        else:
            module_path = sub_module

        for i in range(0, len(sub_module)):
            if sub_module_run[i]:
                runStr = [elapsed_time, module_path[i], "<", 
                         os.path.join(latpath, jobname_m[i] + "." + sub_module[i]), ">",
                         os.path.join(latpath, jobname_m[i] + "_" + sub_module[i] + ".output")]
                line += " ".join(runStr).strip() + "\n"

        return line

    def write_input_file(self, folder=None):
        """(self,str) ->(None)

            Save batch input data to file named filename

        :param folder:  (Default value = None)
        :type folder:
        :returns:
        :rtype:
        """

        # Check data integrity before anything is written on disk or run
        self.check_input_file()

        if folder is None:
            #sys.exit('Batch_lattice.write_input_file: \'folder\' has to be given!')
            folder = "./"
        else:
            common.check_folders(folder)

        fl = open(folder + '/{0}.sh'.format(self.jobname_lat), "w")
        fl.write(self.output())
        fl.close()

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
            print('WARNING: Batch_lattice() class has no attribute \'{0}\''.format(key))
        return

    def check_input_file(self):
        """Perform various checks on the class data to
            make sure that all necessary data exists
            before we attempt to write the input file to disk

        :returns:
        :rtype:
        """

        # Mission critical parameters
        if self.jobname_lat is None:
            if self.lat is not None:
                self.jobname_lat = self.lat
            else:
                sys.exit(
                    'Batch_lattice.check_input_file: \'jobname_lat\' or' +\
                    ' \'lat\' (jobname_lat = lat) has to be given!')

        if self.latpath is None:
            self.latpath = "./"
        if self.runtime is None:
            self.runtime = "01:00:00"
        if self.EMTOdir is None:
            self.EMTOdir = "$HOME/EMTO5.8/"
        if self.runBMDL is None:
            self.runBMDL = True
        if self.runKSTR is None:
            self.runKSTR = True
        if self.runSHAPE is None:
            self.runSHAPE = True
        if self.kappaw is None:
            self.kappaw = [0.0]

        self.kappalen = len(self.kappaw)

        if self.kappalen == 2:
            self.runKSTR2 = True
        return
