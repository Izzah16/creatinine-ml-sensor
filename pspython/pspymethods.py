import clr
from enum import Enum
import os
import sys

# Load DLLs
scriptDir = os.path.dirname(os.path.realpath(__file__))
# This dll contains the classes in which the data is stored
clr.AddReference(scriptDir + '\\PalmSens.Core.dll')
# This dll is used to load your session file
clr.AddReference(scriptDir + '\\PalmSens.Core.Windows.dll')

from PalmSens.Techniques import AmperometricDetection
from PalmSens.Techniques import ImpedimetricMethod
from PalmSens.Techniques import DifferentialPulse
from PalmSens.Techniques import CyclicVoltammetry

def differential_pulse_voltammetry(**kwargs):
    e_begin = kwargs.get('e_begin', -0.5)
    e_end = kwargs.get('e_end', 0.5)
    e_step = kwargs.get('e_step', 0.005)
    pulse_height = kwargs.get('pulse_height', 0.05)
    pulse_width = kwargs.get('pulse_width', 0.05)
    scan_rate = kwargs.get('scan_rate', 0.01)
    
    dpv = DifferentialPulse()
    dpv.BeginPotential = e_begin
    dpv.EndPotential = e_end
    dpv.StepPotential = e_step
    dpv.PulseHeight = pulse_height
    dpv.PulseWidth = pulse_width
    dpv.ScanRate = scan_rate
    
    return dpv

def chronoamperometry(**kwargs):
    e_deposition = kwargs.get('e_deposition', 0.0)
    t_deposition = kwargs.get('t_deposition', 0.0)
    e_conditioning = kwargs.get('e_conditioning', 0.0)
    t_conditioning = kwargs.get('t_conditioning', 0.0)
    equilibration_time = kwargs.get('equilibration_time', 0.0)
    interval_time = kwargs.get('interval_time', 0.1)
    e = kwargs.get('e', 0.0)
    run_time = kwargs.get('run_time', 1.0)
    ca = AmperometricDetection()
    ca.DepositionPotential = e_deposition
    ca.DepositionTime = t_deposition
    ca.ConditioningPotential = e_conditioning
    ca.ConditioningTime = t_conditioning
    ca.EquilibrationTime = equilibration_time
    ca.IntervalTime = interval_time
    ca.Potential = e
    ca.RunTime = run_time
    return ca

def electrochemical_impedance_spectroscopy(**kwargs):
    scan_type = kwargs.get('scan_type', 2)  # (0 = potential, 1 = time, 2 = fixed)
    freq_type = kwargs.get('freq_type', 1)  # (0 = fixed, 1 = scan)
    equilibration_time = kwargs.get('equilibration_time', 0.0)
    e_dc = kwargs.get('e_dc', 0.0)
    e_ac = kwargs.get('e_ac', 0.01)
    n_frequencies = kwargs.get('n_frequencies', 11)
    max_frequency = kwargs.get('max_frequency', 1e5)
    min_frequency = kwargs.get('min_frequency', 1e4)
    eis = ImpedimetricMethod()
    eis.ScanType = scan_type
    eis.FreqType = freq_type
    eis.EquilibrationTime = equilibration_time
    eis.Potential = e_dc
    eis.Eac = e_ac
    eis.nFrequencies = n_frequencies
    eis.MaxFrequency = max_frequency
    eis.MinFrequency = min_frequency
    return eis

def cyclic_voltammetry(**kwargs):
    equilibration_time = kwargs.get('t_equilibration', 2.0)  # Added equilibration time
    e_begin = kwargs.get('e_begin', -1.0)  # Matches Starting Potential
    e_end = kwargs.get('e_end', 0.5)  # Matches Ending/Vertex 2
    e_step = kwargs.get('e_step', 0.01)  # Matches Step Potential
    scan_rate = kwargs.get('scan_rate', 0.5)  # Matches Scan Rate
    vertex1 = kwargs.get('e_vtx1', e_begin)  # Matches Vertex 1 (defaults to e_begin)
    vertex2 = kwargs.get('e_vtx2', e_end)  # Matches Vertex 2 (defaults to e_end)
    n_scans = kwargs.get('n_scans', 2)  # Matches Number of Scans

    cv = CyclicVoltammetry()
    cv.EquilibrationTime = equilibration_time  # Ensuring equilibration time is set
    cv.BeginPotential = e_begin
    cv.EndPotential = e_end
    cv.StepPotential = e_step
    cv.ScanRate = scan_rate
    cv.VertexPotential1 = vertex1
    cv.VertexPotential2 = vertex2
    cv.NumberOfScans = n_scans
    
    return cv

# just a test
if __name__ == '__main__':
    ca = chronoamperometry(interval_time=.01, e=.2, run_time=5.0)
    cv = cyclic_voltammetry()
    test = 'test'
