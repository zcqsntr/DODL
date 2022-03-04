import sys
from opentrons.simulate import simulate, format_runlog

protocol_file = open('run_script.py')
# simulate() the protocol, keeping the runlog
runlog, _bundle = simulate(protocol_file)
# print the runlog
print(format_runlog(runlog))

