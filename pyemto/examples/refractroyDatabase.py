#!python
#
#Calculate the lattice constant and elastic constant of refractory HEAs

import os
import re
import shutil
import operator
from itertools import combinations
from pymatgen.core.periodic_table import Element
import scipy.constants
from pyemto.latticeinputs.batch import batch_head
from pyemto.utilities import distort
from monty.serialization import loadfn

import numpy as np
from pyemto.EOS import EOS
from pyemto.emto_parser import EMTOPARSER
import json

from pyemto.examples.emto_input_generator import EMTO
import math

def input_gen_eqv(jobname="NbTiVZr", emtopath="./", latname="bcc", sws0=3.0, sws_percent=10.0, sws_step=11, 
    concs=[[0.25, 0.25, 0.25, 0.25]], species = [['Nb','Ti','V','Zr']]):
    """
    Generate input file and batch file for equial volume

    Parameter
        jobname: str
            The jobname
        latname: str
            The lattice name
        sws0: float
            The initial sws, A
        sws_percent: float
            The percentage for sws, %
        sws_step: int
            The number of point
        emtopath: str
            The path of
        concs: list-2D
            The concentrations
        species: list-2D
            The species
    Return
        None
    """

    find_primitive = False
    make_supercell = None
    coords_are_cartesian = True
    runtime = "12:00:00"

    nkx = 21
    nky = 21
    nkz = 21
    ncpu = 1

    splts = [[0]*len(species[0])]

    latpath = emtopath

    swsmin = sws0 * (1.0 - sws_percent/100.0)
    swsmax = sws0 * (1.0 + sws_percent/100.0)

    if latname == "bcc":
        prims0 = np.array([
            [-0.5,0.5,0.5],
            [0.5,-0.5,0.5],
            [0.5,0.5,-0.5]
        ])
        basis0 = np.array([
            [0.0,0.0,0.0]
        ])
    elif latname == "fcc":
        prims0 = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        basis0 = np.array([
            [0.0,0.0,0.0]
        ])
    elif latname == "hcp":
        prims0 = np.array([
            [0.5, -math.sqrt(3.)/2., 0.0],
            [0.5,  math.sqrt(3.)/2., 0.0],
            [0.0, 0.0,-1.0]
        ])
        basis0 = np.array([
            [1./3., 2./3., 0.25],
            [2./3., 1./3., 0.75]
        ])
    else:
      raise ValueError("Current lattice({}) is not supported".format(latname))


    input_creator = EMTO(folder=emtopath, EMTOdir='/storage/home/mjl6505/bin')
    jobnamei = jobname + "_" + latname
    input_creator.prepare_input_files(latpath=latpath, jobname=jobnamei, species=species, splts=splts, concs=concs,
        prims=prims0, basis=basis0, find_primitive=find_primitive, coords_are_cartesian=coords_are_cartesian, 
        latname=latname, ncpa=15, sofc='Y', nkx=nkx, nky=nky, nkz=nkz, ncpu=ncpu, parallel=False, alpcpa=0.6, 
        runtime=runtime, KGRN_file_type='scf', KFCD_file_type='fcd', amix=0.01, tole=1e-6, tolef=1e-6, iex=4, 
        niter=200, kgrn_nfi=91, make_supercell=make_supercell)
        
    sws_range = np.linspace(swsmin, swsmax, sws_step)

    input_creator.write_bmdl_kstr_shape_input()
    input_creator.write_kgrn_kfcd_swsrange(sws=sws_range)

def input_gen_elastic(jobname="NbTiVZr", emtopath="./", latname="bcc", sws0=3.0, delta_max=0.05, delta_step=6, 
    concs=[[0.25, 0.25, 0.25, 0.25]], species = [['Nb','Ti','V','Zr']]):
    find_primitive = False
    make_supercell = None
    coords_are_cartesian = True

    ncpu = 1
    runtime = "12:00:00"

    splts = [[0]*4]

    if isinstance(sws0, (float, int, str)):
        sws0 = [float(sws0)]

    latpath = emtopath

    deltas = np.linspace(0, delta_max, delta_step)

    if latname == "bcc":
        prims0 = np.array([
            [-0.5,0.5,0.5],
            [0.5,-0.5,0.5],
            [0.5,0.5,-0.5]
        ])
        basis0 = np.array([
            [0.0,0.0,0.0]
        ])
    elif latname == "fcc":
        prims0 = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        basis0 = np.array([
            [0.0,0.0,0.0]
        ])
    elif latname == "hcp":
        prims0 = np.array([
            [0.5, -math.sqrt(3.)/2., 0.0],
            [0.5,  math.sqrt(3.)/2., 0.0],
            [0.0, 0.0,-1.0]
        ])
        basis0 = np.array([
            [1./3., 2./3., 0.25],
            [2./3., 1./3., 0.75]
        ])
    else:
      raise ValueError("Current lattice({}) is not supported".format(latname))

    # We need to use a non-zero value for the first delta to break the symmetry of the structure.
    deltas[0] = 0.001
    # Only two distortions for cubic (third one is bulk modulus EOS fit)
    distortions = ['Cprime', 'C44']

    for i, distortion in enumerate(distortions):
        print('#'*100)
        print('distortion = ',distortion)
        print('#'*100)
        for delta in deltas:
            print('#'*100)
            print('delta = ',delta)
            print('#'*100)

            # These distortion matrices are from the EMTO book.            
            if distortion == 'Cprime':
                dist_matrix = np.array([
                        [1+delta,0,0],
                        [0,1-delta,0],
                        [0,0,1/(1-delta**2)]
                        ])
            elif distortion == 'C44':
                dist_matrix = np.array([
                        [1, delta, 0],
                        [delta ,1, 0],
                        [0, 0, 1/(1-delta**2)]
                        ])

            # Calculate new lattice vectors and atomic positions
            prims = distort(dist_matrix, prims0)
            basis = distort(dist_matrix, basis0)

            # Each different distortion might need different set of nkx, nky, nkz
            if distortion == 'Cprime':
                nkx = 21; nky = 21; nkz = 21
            elif distortion == 'C44':
                nkx = 20; nky = 20; nkz = 25

            input_creator = EMTO(folder=emtopath, EMTOdir='/storage/home/mjl6505/bin')
            latnamei = latname + '{0}_{1:4.2f}'.format(i+1, delta)
            jobnamei = jobname + "_" + latnamei
            input_creator.prepare_input_files(latpath=latpath, jobname=jobnamei, species=species, splts=splts, 
                concs=concs, prims=prims, basis=basis, find_primitive=find_primitive, latname=latnamei, 
                coords_are_cartesian=coords_are_cartesian, ncpa=15, sofc='Y', nkx=nkx, nky=nky, nkz=nkz, 
                ncpu=ncpu, parallel=False, alpcpa=0.6, runtime=runtime, KGRN_file_type='scf', KFCD_file_type='fcd', 
                amix=0.01, tole=1e-6, tolef=1e-6, iex=4, niter=200, kgrn_nfi=91, make_supercell=make_supercell)
            
            sws_range = np.array(sws0)
            
            input_creator.write_bmdl_kstr_shape_input()
            input_creator.write_kgrn_kfcd_swsrange(sws=sws_range)

def write_eqv_post(jobname="NbTiVZr", folder="./", lat="bcc", concs=[[0.25, 0.25, 0.25, 0.25]], 
    species=[['Nb','Ti','V','Zr']], DLM=False):

    lines = """#!python  

import numpy as np
from pyemto.EOS import EOS
from pyemto.emto_parser import EMTOPARSER
import json
import os

def equilv_result(jobname="test", folder="./", lat="bcc", concs=[[0.25, 0.25, 0.25, 0.25]], 
    species = [['Nb','Ti','V','Zr']], DLM=False):

    if lat == 'bcc':
        Natom = 2
    elif lat == 'fcc':
        Natom = 2
    elif lat == 'hcp':
        Natom = 2
    else:
        raise ValueError("Not supported")

    ry2J = 2.1798741e-18
    A2m = 1e-10
    bohr2A = 0.529177249


    # Equilibrium lattice constant output files
    kgrn_path = os.path.join(folder, "kgrn", "*")
    kfcd_path = os.path.join(folder, "kfcd", "*")
    vol_data = EMTOPARSER(kgrn_path, kfcd_path, suffix='prn', DLM=DLM)

    vol_data.create_df()
    vol_df = vol_data.main_df

    eos = EOS('test', method='morse')

    indMin = 0
    indMax = len(vol_df.SWS)

    SWS0, E0, B0, grun0, error0 = eos.fit(vol_df.SWS[indMin:indMax], vol_df.EPBE[indMin:indMax], show_plot=False)
    vol0 = 4./3*np.pi*(SWS0*bohr2A)**3
    a = (vol0 * Natom) ** (1./3.)

    eos_result = {"sws": SWS0, "vol": vol0 , "a": a, "E0": E0, "B0": B0, "grun": grun0, 
        "error": error0, "jobname": jobname, "latname": lat, "concs": concs, "species": species}

    with open(os.path.join(folder, jobname+"-eqv.json"), 'w+') as f:
        json.dump(eos_result, f, indent=4)
    """
    lines += "\njobname = '{}'\n".format(jobname)
    lines += "folder = '{}'\n".format(folder)
    lines += "lat = '{}'\n".format(lat)
    concs_str = [str(coni) for coni in concs[0]]
    lines += "concs = [[{}]]\n".format(", ".join(concs_str))
    lines += "species = [['{}']]\n".format("', '".join(species[0]))
    lines += "\nequilv_result(jobname=jobname, folder=folder, lat=lat, concs=concs, species=species)\n"

    with open(jobname + "_eqv_post.py", "w+") as fid:
        fid.write(lines)

def write_elastic_post(jobname="NbTiVZr", folder="./", latname="bcc", sws0=3.0, delta_max=0.05, delta_step=6, 
    B=267.0, concs=[[0.25, 0.25, 0.25, 0.25]], species=[['Nb','Ti','V','Zr']], DLM=False):

    lines = """#!python

import numpy as np
from pyemto.EOS import EOS
from pyemto.emto_parser import EMTOPARSER
import json
import os

def elastic_result(jobname="NbTiVZr", folder="./", latname="bcc", sws0=3.0, delta_max=0.05, delta_step=6, 
    B=267.0, concs=[[0.25, 0.25, 0.25, 0.25]], species = [['Nb','Ti','V','Zr']], DLM=False):

    ry2J = 2.1798741e-18
    A2m = 1e-10
    bohr2A = 0.529177249

    deltas = np.linspace(0, delta_max, delta_step)
    deltas[0] = 0.001

    # Elastic constants output files
    elastic_kgrn_path = os.path.join(folder, "kgrn", "*")
    elastic_kfcd_path = os.path.join(folder, "kfcd", "*")
    ec_data = EMTOPARSER(elastic_kgrn_path, elastic_kfcd_path, suffix='prn', DLM=DLM)

    ec_data.create_df()
    ec_df = ec_data.main_df

    eos = EOS('test', method='morse')

    # d1 is the c' distortion
    d1 = np.asarray(ec_df[ec_df.Struc.str.contains(latname + '1')].EPBE)

    # d2 is the c44 distortion
    d2 = np.asarray(ec_df[ec_df.Struc.str.contains(latname + '2')].EPBE)

    cprime_coeffs, cprime_error = eos.distortion_fit(deltas, d1, num=1)
    c44_coeffs, c44_error = eos.distortion_fit(deltas, d2, num=1)

    # Change units to GPa
    vol0 = 4./3*np.pi*(sws0*bohr2A)**3
    cprime_coeffs = cprime_coeffs[0] * ry2J / (vol0*A2m**3) / 2 / 1e9
    c44_coeffs = c44_coeffs[0] * ry2J / (vol0*A2m**3) / 2 / 1e9

    def calc_CIJ(B, Cprime, C44):
        C11 = (3.*B + 4.*Cprime)/3.
        C12 = (3.*B - 2.*Cprime)/3.
        CIJ = [[C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44]]
        return CIJ
    CIJ = calc_CIJ(B, cprime_coeffs, c44_coeffs)

    elastic_result = {"vol": vol0, "B": B, "jobname": jobname, "latname": latname, 
        "concs": concs, "species": species,"delta": deltas.tolist(), "CIJ": CIJ, "data_Cprime": d1.tolist(), 
        "fit_Cprime": cprime_coeffs.tolist(), "fit_err_Cprime": cprime_error.tolist(), 
        "data_C44": d2.tolist(), "fit_C44": c44_coeffs.tolist(), "fit_err_C44": c44_error.tolist()}

    with open(os.path.join(folder, jobname+"-elastic.json"), 'w+') as f:
        json.dump(elastic_result, f, indent=4)
    """

    lines += "\njobname = '{}'\n".format(jobname)
    lines += "folder = '{}'\n".format(folder)
    lines += "latname = '{}'\n".format(latname)
    concs_str = [str(coni) for coni in concs[0]]
    lines += "concs = [[{}]]\n".format(", ".join(concs_str))
    lines += "species = [['{}']]\n".format("', '".join(species[0]))
    lines += "sws0 = {}\n".format(str(float(sws0)))
    lines += "delta_max = {}\n".format(str(float(delta_max)))
    lines += "delta_step = {}\n".format(str(int(delta_step)))
    lines += "B = {}\n".format(str(float(B)))
    lines += "\nelastic_result(jobname=jobname, folder=folder, latname=latname, " + \
        "sws0=sws0, delta_max=delta_max, delta_step=delta_step, B=B, " + \
        "concs=concs, species=species)\n"

    with open(jobname + "_elastic_post.py", "w+") as fid:
        fid.write(lines)

def parase_pbs_script(filename = "emtojob.pbs"):
    """
    Parse the exe part of pbs file

    Parameter
        filename: str (filename-like)
            The filename of the pbs script
    Return
        param_dict: dict
            The dict of parameters.
    """
    s = {"-q": "queue", "-A": "account", "-N": "job_name", "-V": "env",
         "-G": "group_name"}
    submit_s = {"nodes": "node", "ppn": "core", "pmem": "pmem"}
    param_dict = {"module": [], "cmds": []}
    with open(filename, "r") as fid:
        for eachline in fid:
            eachline = eachline.strip()
            if eachline.startswith("#PBS"):
                line_list = re.split("\s+", eachline)
                if line_list[1] == "-l":
                    if line_list[2].startswith("walltime"):
                        # walltime
                        param_dict["walltime"] = line_list[2].split("=")[1]
                    else:
                        for item in line_list[2].split(":"):
                            key = item.split("=")[0]
                            # nodes, ppn, pmem
                            value = item.split("=")[1]
                            if key in submit_s:
                                param_dict[submit_s[key]] = value
                else:
                    if line_list[1] in s:
                        param_dict[s[line_list[1]]] = line_list[2]
            elif eachline.startswith("module"):
                modules = eachline.split()[2:]
                for module in modules:
                    param_dict["module"].append(module)
            elif eachline.startswith(("cd $", "#")) or (not eachline):
                #The cd $PBS_O_WORKDIR, or comments(#) or empty
                pass
            else:
                param_dict["cmds"].append(eachline + "\n")
    return param_dict

def parse_queue_script(template="emtojob.pbs", queue_type="pbs"):
    """
    Parse the queue script. Currently only pbs is supported

    Parameter
        template: str (filename-like)
            The filename of the queue script. Default: vaspjob.pbs
        queue_type: str
            The type of queue system. Default: pbs
    Return
    """
    param_dict = {}
    if queue_type == "pbs":
        param_dict = parase_pbs_script(filename=template)
    else:
        raise ValueError("Only PBS is supported now. Other system will coming soon...")
    return param_dict

def merge_batchfile(batchfiles, latpath="./", jobname="test", queue_type="pbs"):
    """
    Merge batch files. (Merge head file)
    """
    cmd_lines = ""
    if isinstance(batchfiles, str):
        batchfiles = [batchfiles]
    for file in batchfiles:
        if os.path.isfile(file):
            param_dict = parse_queue_script(template=file, queue_type=queue_type)
        runtime = param_dict["walltime"]
        account = param_dict["account"]
        head_lines = batch_head(jobname, latpath=latpath, runtime=runtime, account=account, 
            queue_type=queue_type, queue_options=param_dict)
        cmd_lines += "".join(param_dict["cmds"])
    return cmd_lines, head_lines

def evaluate_V0(alloy={"Nb": 1, "Ti": 1, "V": 1, "Zr": 1}, norm=True):
    '''
    Calculate the volume of alloys using mix rule (volme conservation)

    Parameter
        alloy: dict
            The alloy composition
        norm: bool
            Normalize or not
    Return
        v_alloy: float, unit: angstrom^3
            The volume of the alloy, if norm=True, then it's the volume of single atom
    '''
    Avogadro = scipy.constants.Avogadro
    if norm:
            Natom = sum(alloy.values())
    v_alloy = 0
    for ele in alloy:
        density = float(Element(ele).density_of_solid)/1000.
        mass = float(Element(ele).atomic_mass)
        if norm:
            alloy[ele] = alloy[ele] / Natom
        v_alloy = v_alloy + mass/density * alloy[ele]
    return 1e24/Avogadro * v_alloy

def creat_folders(folder):
    """
    Create folders if not exist, leave a warning if exists
    """
    if os.path.exists(folder):
        print("WARNING: " + folder + " exists!")
    else:
        os.makedirs(folder)

def issublist(listi, listAll):
    """
    If listi(1D) is a sub list of listAll (2D)

    Parameter
        listi: list (1D)
        listAll: list (2D)
    Return
        flag: bool
            If listi is a sub list of listAll, flat=True, else False
    """
    flag = False
    for listj in listAll:
        if operator.eq(listi, listj):
            flag = True
            return flag
    return flag

def creat_alloy(sys_alloy=('Nb', 'Ti'), n_point=10):
    """
    Create non-redundent alloys by interpolation 
        It will create Ax(BC)1-x like alloys
        E.g. when n_point = 10,
            For binary, it will generate AB9, A2B8, A3B7 ...
            For ternary, it will generate A(BC)9, A2(BC)8, ... B(AC)9, B2(AC)8 ...

    Parameter
        sys_alloy: list or tuple
            The alloy system
        n_point: int
            The 
    Return
        alloy_dict: dict
            The returned alloys system. The key is the alloy's name, the value is the normalized(to sum=1) composition
    """
    alloys = []
    alloy_dict = {}
    sys_alloy = list(sys_alloy)
    n_ele = len(sys_alloy)
    for i_ele in range(n_ele):
        eles = sys_alloy.copy()
        VarEle = eles.pop(i_ele)
        for i_com in range(1, n_point):
            #Create A1B9, A2B8, A3B7 like alloys
            alloyi = []
            num1 = i_com
            num2 = n_point - i_com
            if num1 == 1:
                num1 = ""
            if num2 == 1:
                num2 = ""
            if n_ele == 2:
                alloyName = VarEle + str(num1) + "".join(eles) + str(num2)
            else:
                alloyName = VarEle + str(num1) + "(" + "".join(eles) + ")" + str(num2)
            com_other = round((1. - float(i_com)/float(n_point))/float(n_ele - 1), 3)
            com_main = round(1 - (n_ele - 1) * com_other, 3)
            for i in range(n_ele):
                if i == i_ele:
                    alloyi.append(com_main)
                else:
                    alloyi.append(com_other)
            if not issublist(alloyi, alloys):
                alloys.append(alloyi)
                alloy_dict[alloyName] = alloyi
    return alloy_dict

def find_pbs_script(folder="./", jobname="NbTiVZr", latname="bcc", ext="sh"):
    if isinstance(latname, str):
        latname = [latname]
    files = os.listdir(folder)
    pbs_scripts = []
    for lat in latname:
        lat_script = lat + "." + ext
        pbs_scripts.append(lat_script)
        for file in files:
            if file.endswith(ext) and (lat in file) and (file != lat_script):
                #ensure it is script, and include the lat, and not the lat script
                pbs_scripts.append(file)
    return pbs_scripts

def wflow_eqv(eqv_folder="eqv", jobname="NbTiVZr", latname="bcc", queue_type="pbs",
    emtopath="./", sws0=3.0, sws_percent=10.0, sws_step=11,
    concs=[[0.25, 0.25, 0.25, 0.25]], species=[['Nb','Ti','V','Zr']]):
    """
    Workflow for equialium volume

    Parameter
        eqv_folder: str
            The folder name for the input and result, "eqv"
        jobname: str
            The jobname, "NbTiVZr"
        latname: str
            The lattice name, "bcc"
        queue_type = "pbs"
        emtopath = "./"
        sws0 = 3.0
        sws_percent = 10.0
        sws_step = 11
        concs = [[0.25, 0.25, 0.25, 0.25]]
        species = [['Nb','Ti','V','Zr']]
    Return
        None
    """

    creat_folders(eqv_folder)
    os.chdir(eqv_folder)
    input_gen_eqv(jobname=jobname, emtopath=emtopath, latname=latname, sws0=sws0, sws_percent=sws_percent, 
        sws_step=sws_step, concs=concs, species = species)
    write_eqv_post(jobname=jobname, folder=emtopath, lat=latname, concs=concs, species=species, DLM=False)

    pbs_scripts = find_pbs_script(folder=emtopath, jobname=jobname, latname=latname, ext=queue_type)
    cmd_lines, head_lines = merge_batchfile(pbs_scripts, latpath=emtopath, jobname=jobname, queue_type=queue_type)
    script_lines = head_lines
    script_lines += "\n#Change to " + eqv_folder + " folder.\n"
    script_lines += "cd " + eqv_folder + "\n"
    script_lines += cmd_lines
    script_lines += "\npython " + jobname + "_eqv_post.py\n"
    script_lines += "cd .."
    os.chdir("..")
    with open(jobname + "_" + eqv_folder + "." + queue_type, "w+") as fid:
        fid.write(script_lines)

def wflow_elastic(elastic_folder="elastic", jobname="NbTiVZr", latname="bcc", queue_type="pbs",
    emtopath="./", sws0=3.06, B=115.86, delta_max=0.05, delta_step=6,
    concs=[[0.25, 0.25, 0.25, 0.25]], species=[['Nb','Ti','V','Zr']]):
    """
    Workflow for single elastic constant
        elastic_folder = "elastic"
        jobname = "NbTiVZr"
        latname = "bcc"
        queue_type = "pbs"
        emtopath = "./"
        sws0 = 3.0626926707382136
        B = 115.86573735739826
        delta_max = 0.05
        delta_step = 6
        concs = [[0.25, 0.25, 0.25, 0.25]]
        species = [['Nb','Ti','V','Zr']]
    """

    creat_folders(elastic_folder)
    os.chdir(elastic_folder)
    input_gen_elastic(jobname=jobname, emtopath=emtopath, latname=latname, sws0=sws0, delta_max=delta_max, 
        delta_step=delta_step, concs=concs, species=species)
    write_elastic_post(jobname=jobname, folder=emtopath, latname=latname, sws0=sws0, delta_max=delta_max, 
        delta_step=delta_step, B=B, concs=concs, species=species, DLM=False)

    deltas = np.linspace(0, delta_max, delta_step)
    deltas[0] = 0.001
    lat_name = []
    for i in range(2):
        for delta in deltas:
            lat_name.append(latname + "_" + '{0}_{1:4.2f}'.format(i+1, delta))
    pbs_scripts = find_pbs_script(folder=emtopath, jobname=jobname, latname=lat_name, ext=queue_type)
    cmd_lines, head_lines = merge_batchfile(pbs_scripts, latpath=emtopath, jobname=jobname, queue_type=queue_type)
    script_lines = head_lines
    script_lines += "\n#Change to " + elastic_folder + " folder.\n"
    script_lines += "cd " + elastic_folder + "\n"
    script_lines += cmd_lines
    script_lines += "\npython " + jobname + "_elastic_post.py\n"
    script_lines += "cd .."
    os.chdir("..")
    with open(jobname + "_" + elastic_folder + "." + queue_type, "w+") as fid:
        fid.write(script_lines)

def wflow_eqv_elastic(jobname="NbTiVZr", latname="bcc", queue_type="pbs", emtopath="./", sws0=3.0,
    sws_percent=10.0, sws_step=11, delta_max=0.05, delta_step=6,
    concs=[[0.25, 0.25, 0.25, 0.25]], species=[['Nb','Ti','V','Zr']]):
    elastic_folder = "elastic"
    eqv_folder = "eqv"
    wflow_eqv(eqv_folder=eqv_folder, jobname=jobname, latname=latname, queue_type=queue_type,
        emtopath=emtopath, sws0=sws0, sws_percent=sws_percent, sws_step=sws_step,
        concs=concs, species=species)
    param_dict = parse_queue_script(template=jobname + "_" + eqv_folder + "." + queue_type, queue_type="pbs")
    os.chdir(eqv_folder)
    for cmd in param_dict["cmds"]:
        if cmd.startswith("cd"):
            pass
        else:
            os.system(cmd.strip())
    os.chdir("..")
    #Note: only work in current, the emtopath="./"
    eqv_result = loadfn(os.path.join(eqv_folder, jobname+"-eqv.json"))
    wflow_elastic(elastic_folder=elastic_folder, jobname=jobname, latname=latname, queue_type=queue_type,
        emtopath=emtopath, sws0=eqv_result["sws"], B=eqv_result["B0"], delta_max=delta_max, delta_step=delta_step,
        concs=concs, species=species)
    param_dict = parse_queue_script(template=jobname + "_" + elastic_folder + "." + queue_type, queue_type="pbs")
    os.chdir(elastic_folder)
    for cmd in param_dict["cmds"]:
        if cmd.startswith("cd"):
            pass
        else:
            os.system(cmd.strip())
    os.chdir("..")

def write_input(**kwargs):
    with open("input.json", "w+") as fid:
        json.dump(kwargs, fid, indent=4)

def batch_run_emto(files_to_copy=['run_emto_elastic.py', ' NbTiVZr.pbs']):
    phase = 'bcc'
    #The number of elements is range from (NEleMin, NEleMax), 0 for unary, 1 for binary
    NEleMin = 0
    NEleMax = 3

    files_to_copy = [os.path.abspath(filei) for filei in files_to_copy]

    eles = ['Al', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']
    folderName = ['unary', 'binary', 'ternary', 'quaternary', 'quinary']
    n_point = [1, 10, 6, 4, 5]

    bohr2A = 0.529177249
    if phase == 'bcc':
        Natom = 2
    elif phase == 'fcc':
        Natom = 4
    elif phase == 'hcp':
        Natom = 2
    else:
        raise Exception('Current {phase} is not supported.'.format(phase=phase))

    for i in range(NEleMin, NEleMax):
        creat_folders(folderName[i])
        os.chdir(folderName[i])
        ele_comb = list(combinations(eles, i+1))
        for sys_alloy in ele_comb:
            subFolderName = "".join(sys_alloy)
            creat_folders(subFolderName)
            if i > 0:
                os.chdir(subFolderName)
                alloy_dict = creat_alloy(sys_alloy=sys_alloy, n_point=n_point[i])
                for alloy in alloy_dict:
                    creat_folders(alloy)
                    os.chdir(alloy)
                    alloy_i = {sys_alloy[i]: alloy_dict[alloy][i] for i in range(len(sys_alloy))}
                    v0 = evaluate_V0(alloy=alloy_i, norm=True)
                    sws0 = math.pow(3.*v0 / (4.*np.pi), 1./3.)/bohr2A
                    write_input(jobname=alloy, latname="bcc", queue_type="pbs", emtopath="./", sws0=sws0,
                        sws_percent=10.0, sws_step=11, delta_max=0.05, delta_step=6,
                        concs=[alloy_dict[alloy]], species=[sys_alloy])
                    for filei in files_to_copy:
                        shutil.copyfile(filei, os.path.join("./", os.path.basename(filei)))
                    os.chdir("..")
                os.chdir("..")
            else:
                #unary
                pass
        os.chdir("..")
    #print(straini)

#wflow_eqv()
#batch_run_emto()