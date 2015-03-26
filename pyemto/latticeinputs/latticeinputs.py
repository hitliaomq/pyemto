# -*- coding: utf-8 -*-
"""

Created on Wed Dec  3 14:25:06 2014

@author: Matti Ropo
@author: Henrik Levämäki

"""

import sys
import numpy as np


class Latticeinputs:
    """Class which is used to communicate with the Bmdl, Kstr and Shape classes.

    :returns: None
    :rtype: None
    """

    def __init__(self):

        # Import necessary packages
        from pyemto.latticeinputs.bmdl import Bmdl
        from pyemto.latticeinputs.kstr import Kstr
        from pyemto.latticeinputs.shape import Shape
        from pyemto.latticeinputs.batch import Batch

        self.bmdl = Bmdl()
        self.kstr = Kstr()
        self.shape = Shape()
        self.batch = Batch()

        return

    def set_values(self, **kwargs):
        """Passes various input parameters down to the Kgrn and Kfcd classes.

        :param **kwargs:
        :type **kwargs:
        :returns:
        :rtype:
        """

        for key in kwargs:
            attr_found = False
            if hasattr(self.bmdl, key):
                self.bmdl.set_values(key, kwargs[key])
                attr_found = True
            if hasattr(self.kstr, key):
                self.kstr.set_values(key, kwargs[key])
                attr_found = True
            if hasattr(self.shape, key):
                self.shape.set_values(key, kwargs[key])
                attr_found = True
            if hasattr(self.batch, key):
                self.batch.set_values(key, kwargs[key])
                attr_found = True
            if attr_found == False:
                print(
                    'WARNING: Neither Bmdl(), Kstr(), Shape() nor Batch_lattice()' +\
                    ' classes have the attribute \'{0}\''.format(key))
        return

    def distortion(self, lat=None, dist=None, ca=None, index=None, deltas=None, dmaxs=None):
        """A function which sets various class data to create distorted lattice structures.

        Distorted lattices are used to calculate elastic constants.        
        An integer *index* is used to specify for which delta and dmax value we
        want to generate the distortion input file data.

        Default naming convention for the structure files:

        ======= = ====
        bcc bco = bcco
        bcc fco = bccm
        fcc bco = fccm
        fcc fco = fcco
        ======= = ====

        ===== ==================================================================
        lat   Original undistorted lattice
        dist  Which distortion we want to calculate.
              Possible values: 'ortho' or 'mono' for lat='bcc', 'fcc' or 'hcp'.
        ca    c over a for hcp structures.
        index Index specifying an element in the delta array.
              Possible values: 0,1,2,3,4 or 5. 0 = No distortion.
        delta Array of displacement values. Optional, default value
              is good enough almost always (if not always).
        dmax  Array of 'dmax' values for KSTR. They have to be chosen in
              such fashion so as to keep the number of lattice vectors
              constant (or as close to a constant as possible) for all
              of the distorted lattices. Optional, only needed when
              a custom delta array is used.
        ===== ==================================================================

        :param lat: The original, undistorted lattice (Default value = None)
        :type lat: str
        :param dist: The type of distortion (Default value = None)
        :type dist: str
        :param ca: hcp c/a ratio (Default value = None)
        :type ca: float
        :param index: Index for selecting a delta and dmax from the arrays (Default value = None)
        :type index: int
        :param deltas: List of delta values (Default value = None)
        :type deltas: np.array(float)
        :param dmaxs: List of dmax values (Default value = None)
        :type dmaxs: np.array(float)
        :returns: None
        :rtype: None
        """
        
        default_deltas = np.linspace(0.0, 0.05, 6)
        delta_tol = 1.0E-6

        # Mission critical parameters
        # if folder == None:
        #    folder = "./"
        if lat is None:
            sys.exit('latticeinputs.distortion(): \'lat\' has to be given!')

        elif lat == 'hcp' and ca is None:
            sys.exit(
                'latticeinputs.distortion(): \'ca\' (c over a) has to be given for hcp structures!')

        if dist is None:
            sys.exit(
                'latticeinputs.distortion(): \'dist\' (distortion) has to be given!')

        if index is None:
            sys.exit(
                'latticeinputs.distortion(): \'index\'' +\
                ' (delta = delta_array[index]) has to be given!')

        defaultDelta = False
        defaultDmax = False
        if deltas is None:
            deltas = default_deltas
            defaultDelta = True
        else:
            # Check whether the input delta array is equivalent to the default
            same_delta_count = 0
            if len(deltas) == len(default_deltas):
                for i in range(len(deltas)):
                    if np.abs(deltas[i] - default_deltas[i]) < delta_tol:
                        same_delta_count += 1
                if same_delta_count == len(deltas):
                    deltas = default_deltas
                    defaultDelta = True

        delta = deltas[index]

        if defaultDelta == False and dmaxs == None:
            sys.exit(
                'latticeinputs.distortion(): \'dmax\' has to be' +\
                ' given when custom \'delta\' values are used!')

        elif defaultDelta == False and dmaxs != None:
            dmax = dmaxs[index]

        elif defaultDelta == True and dmaxs != None:
            dmax = dmaxs[index]

        # Default dmax will be used
        elif defaultDelta == True and dmaxs == None:
            if lat == 'bcc':
                if dist == 'ortho':
                    dmax_dict = {
                        0: 2.25, 1: 2.25, 2: 2.25, 3: 2.25, 4: 2.25, 5: 2.25}
                    dmax = dmax_dict[index]
                elif dist == 'mono':
                    dmax_dict = {
                        0: 1.59, 1: 1.59, 0o2: 1.59, 3: 1.59, 4: 1.59, 5: 1.59}
                    dmax = dmax_dict[index]

            elif lat == 'fcc':
                if dist == 'ortho':
                    #dmax_dict = {0:1.6,1:1.6,2:1.6,3:1.6,4:1.6,5:1.6}
                    # High-accuracy
                    dmax_dict = {
                        0: 1.9, 1: 1.9, 2: 1.9, 3: 1.9, 4: 1.9, 5: 1.9}
                    dmax = dmax_dict[index]
                elif dist == 'mono':
                    #dmax_dict = {0:2.40,1:2.40,2:2.30,3:2.22,4:2.21,5:2.20}
                    # High-accuracy
                    dmax_dict = {
                        0: 2.70, 1: 2.70, 2: 2.70, 3: 2.70, 4: 2.70, 5: 2.65}
                    dmax = dmax_dict[index]

            elif lat == 'hcp':
                if dist == 'ortho':
                    dmax_dict = {
                        0: 2.52, 1: 2.49, 2: 2.455, 3: 2.43, 4: 2.4, 5: 2.4}
                    dmax = dmax_dict[index]
                elif dist == 'mono':
                    dmax_dict = {
                        0: 2.51, 1: 2.51, 2: 2.51, 3: 2.49, 4: 2.51, 5: 2.49}
                    dmax = dmax_dict[index]

        # Details can be found in Vitos' book pages 104-110.

        if lat == 'bcc' and dist == 'ortho':
            self.set_values(jobname='bcco{0}'.format(index))
            self.set_values(lat='bco', dmax=dmax)

            latparams = [
                1.0, (1.0 - delta) / (1.0 + delta), 1.0 / (1.0 + delta) / (1.0 - delta**2)]
            latvectors = [90.0, 90.0, 90.0]
            basis = [0.0, 0.0, 0.0]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)

        elif lat == 'bcc' and dist == 'mono':
            self.set_values(jobname='bccm{0}'.format(index))
            self.set_values(lat='fco', dmax=dmax)

            latparams = [1.0, (1.0 - delta) / (1.0 + delta),
                         1.0 / (1.0 + delta) / (1.0 - delta**2) / np.sqrt(2.0)]
            latvectors = [90.0, 90.0, 90.0]
            basis = [0.0, 0.0, 0.0]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)

        elif lat == 'fcc' and dist == 'ortho':
            self.set_values(jobname='fcco{0}'.format(index))
            self.set_values(lat='fco', dmax=dmax)

            latparams = [
                1.0, (1.0 - delta) / (1.0 + delta), 1.0 / (1.0 + delta) / (1.0 - delta**2)]
            latvectors = [90.0, 90.0, 90.0]
            basis = [0.0, 0.0, 0.0]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)

        elif lat == 'fcc' and dist == 'mono':
            self.set_values(jobname='fccm{0}'.format(index))
            self.set_values(lat='bco', dmax=dmax)

            latparams = [1.0, (1.0 - delta) / (1.0 + delta),
                         np.sqrt(2.0) / (1.0 + delta) / (1.0 - delta**2)]
            latvectors = [90.0, 90.0, 90.0]
            basis = [0.0, 0.0, 0.0]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)

        elif lat == 'hcp' and dist == 'ortho':
            self.set_values(jobname='hcpo{0}'.format(index))
            self.set_values(lat='baco', dmax=dmax)

            latparams = []
            latparams.append(1.0)
            latparams.append(np.sqrt(3.0) * (1.0 - delta) / (1.0 + delta))
            latparams.append(ca / (1.0 + delta) / (1.0 - delta**2))

            latvectors = [90.0, 90.0, 90.0]

            basis = [
                [0.0, 0.0, 0.0], [0.0, latparams[1] / 3.0, latparams[2] / 2.0]]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)

        elif lat == 'hcp' and dist == 'mono':
            self.set_values(jobname='hcpm{0}'.format(index))
            self.set_values(lat='sm', dmax=dmax)

            bam = ca  # Distorted b over a
            cam = np.sqrt(3.0) / np.sqrt(1.0 + delta**2) / \
                (1.0 - delta**2)  # Distorted c over a
            latparams = [1.0, bam, cam]

            bs1 = [1.0, 0.0, 0.0]
            bs2 = [2.0 * delta / (1.0 + delta**2) * ca,
                   (1.0 - delta**2) / (1.0 + delta**2) * ca, 0.0]
            bs3 = [0.0, 0.0, cam]
            latvectors = [bs1, bs2, bs3]

            pos1 = [0.0, 0.0, 0.0]
            pos2 = [ca * delta / (1.0 + delta**2), ca *
                    (1.0 - delta**2) / (1.0 + delta**2) / 2.0, -cam / 3.0]
            pos3 = [0.5, 0.0, -cam / 2.0]
            pos4 = [pos2[0] + pos3[0], pos2[1] + pos3[1], pos2[2] + pos3[2]]
            basis = [pos1, pos2, pos3, pos4]

            self.set_values(
                latparams=latparams, latvectors=latvectors, basis=basis)
        return
