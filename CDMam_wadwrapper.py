#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# PyWAD is open-source software and consists of a set of modules written in python for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://github.com/wadqc
# 
# The pywad package includes modules for the automated analysis of QC images for various imaging modalities. 
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC) 
#
#
# Changelog:
#   20170825: fixed misinterpretation of auto_suffix
#   20170802: added cdcom params
#   20161220: removed class variables; removed testing stuff
#   20160802: sync with wad2.0
#   20160622: removed adding limits (now part of analyzer)
#   20160620: remove quantity and units
#
# ./CDMam_wadwrapper.py -c Config/mg_cdmam32_L50.json -d TestSet/StudyCDMAM -r results_cdmam.json
from __future__ import print_function

__version__ = '20170825'
__author__ = 'aschilham'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

import numpy as np
if not 'MPLCONFIGDIR' in os.environ:
    os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import CDMam_lib
try:
    import pydicom as dicom
except ImportError:
    import dicom

def logTag():
    return "[CDMam_wadwrapper] "

# MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!


##### Series wrappers
def cdmamsetup_series(inputfile,params,headers_only):
    """
    Shared routine to set runtime parameters and build structure
    Workflow:
        1. Set runtime parameters
        2. Check data format
        3. Build and populate qcstructure
    """
    # 1. Set runtime parameters
    if not headers_only: # then we can skip the parameter check
        try:
            phantomversion = params["phantomversion"]
        except AttributeError:
            raise ValueError(logTag()+" missing phantomversion parameter!")
    
        try:
            modeCDCOM = (str(params["modeCDCOM"]).lower() == 'true')
        except AttributeError:
            raise ValueError(logTag()+" missing cdcommode parameter!")
        try:
            parsCDCOM = params.get('parsCDCOM', '')
            parsCDCOM = parsCDCOM.split(';')
        except AttributeError:
            raise ValueError(logTag()+" malformed parsCDCOM parameter!")
    else:
        phantomversion = '3.2' # dummy for headers
        modeCDCOM = False
        parsCDCOM = []
        
    # 2. Check data format
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(inputfile,headers_only=headers_only,logTag=logTag())

    # 3. Build and populate qcstructure
    qclib = CDMam_lib.CDMam()
    cs = CDMam_lib.CDMamStruct(dcmInfile,pixeldataIn,phantomversion)
    cs.imnum = 0 # needs to be set to an increasing number if multiple files are used
    cs.verbose = False # do not produce detailed logging
    cs.imageFileName = inputfile[0] # only for CDCOM.exe
    cs.parsCDCOM = parsCDCOM # extra runtime flags ('c','high')
    qclib.determineScannerID(cs)
    return qclib,cs,modeCDCOM

def identifyName(cs):
    """
    Come up with a simple identification tag
    """
    idname = "_XX"
    if "RHO" in cs.filtername:
        idname = "_RH"
    elif "MOL" in cs.filtername:
        idname = "_MO"
    elif "ALU" in cs.filtername:
        idname = "_AL" # only one image "_SUMPRES"
    elif "SILV" in cs.filtername:
        idname = "_AG"

    if 'SUM' in cs.energypresentation: # sum, high, low
        idname += 'su'
    elif 'HIGH' in cs.energypresentation: # sum, high, low
        idname += 'hi'
    elif 'LOW' in cs.energypresentation: # sum, high, low
        idname += 'lo'
    if 'PROC' in cs.energypresentation: # presentation or processing
        idname += 'proc'
    elif 'PRES' in cs.energypresentation: # presentation or processing
        idname += 'pres'
    
    return idname

def cdmamqc_list(data,results,params):
    """
    CDMAM analysis for a list of images (as should be!) see cdmamqc_series for more information
    """
    
    finalscore = (np.zeros(np.shape(cs.phantom.groundtruth),dtype=float)).tolist()
    # 1.-3.
    imnum = 0
    for fname in data.series_filelist[0]:
        inputfile = [fname]
        qclib,cs,modeCDCOM = cdmamsetup_series(inputfile, params, headers_only=False)
        cs.imnum = imnum

        if not modeCDCOM:
            score = qclib.CDMamSingle(cs)
        else:
            score = qclib.CDCOMSingle(cs)
        for y in range(len(score)):
            for x in range(len(score[0])):
                finalscore[y][x] += score[y][x]
        
        imnum +=1
        
    # calculate average score
    for y in range(len(finalscore)):
        for x in range(len(finalscore[0])):
            finalscore[y][x] /= len(imagelist)

    qclib.thresholdThickness(cs,finalscore)

def qc_series(data, results, action):
    """
    CDMAM analysis:
        Use a model observer to detect location of gold disc, for different thicknesses and diameters of the disc.
        Predict the threshold thicknesses for each diameter for a human observer
        Calculate a comprehensive image quality factor from the predicted human observer performance.
        NOTE: CDMAM analysis should be performed on 10-16 slightly shifted images to get a meaningful number.
              Here the analysis is done on one image, but the interface for multiple images already exist as cdmamqc_list.
    Params needs to define:
      1. phantomversion = phantomversion (which version of the phantom is used? '3.2' or '3.4')
      2. modeCDCOM = 'False' or 'True' (use external executable cdcom.exe ('True') or not ('False'))

    Workflow:
        1. Set runtime parameters
        2. Check data format
        3. Build and populate qcstructure
        4. Run tests
        5. Build xml output
        6. Build artefact picture thumbnail
    """

    # 1.-3.
    try:
        params = action['params']
    except KeyError:
        params = {}

    inputfile = data.series_filelist[0]  # give me a [filename]
    qclib,cs,modeCDCOM = cdmamsetup_series(inputfile, params, headers_only=False)

    # 4. Run tests
    if not modeCDCOM:
        score = qclib.CDMamSingle(cs)
    else:
        score = qclib.CDCOMSingle(cs)
    qclib.thresholdThickness(cs,score)

    # 5. Build xml output
    # Struct now contains all the results and we can write these to the WAD IQ database
    includedlist = [
        'diam_mm', # list of diameters in mm
        'limit_um', # list of predicted human detectable thickness thresholds in um
        'iqf', # image quality factor
        'threshold_fnames',
        'fit_fnames'
    ]
    idname = identifyName(cs)
    for di,lim in zip(cs.diam_mm,cs.limit_um):
        varname = 'limit_um_'+str('%0.2f'%di).zfill(4)+idname
        results.addFloat(varname, lim) #quantity=str('threshold_limit'),
    varname = 'iqf'+idname
    results.addFloat(varname, cs.iqf)

    # 6. also store images as results
    for fn in cs.threshold_fnames:
        varname = os.path.splitext(fn)[0]
        results.addObject(varname, fn)
    for fn in cs.fit_fnames:
        varname = os.path.splitext(fn)[0]
        results.addObject(varname, fn)

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action):
    """
    Read selected dicomfields and write to IQC database

    Workflow:
        1. Set runtime parameters
        2. Check data format
        3. Build and populate qcstructure
        4. Run tests
        5. Build xml output
    """
    # 1.-3.
    try:
        params = action['params']
    except KeyError:
        params = {}

    inputfile = data.series_filelist[0]  # give me a [filename]
    qclib,cs,modeCDCOM = cdmamsetup_series(inputfile, params, headers_only=True)

    # 4. Run tests
    dicominfo = qclib.DICOMInfo(cs,'qc')

    ## find filtername
    idname = identifyName(cs)
        
    ## 2. Add results to 'result' object
    varname = 'pluginversion'+idname
    results.addString(varname, str(qclib.qcversion))
    for di in dicominfo:
        varname = di[0]+idname
        results.addString(varname, str(di[1])[:min(len(str(di[1])),100)])

if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'qc_series':
            qc_series(data, results, action)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
