#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import daemon
from barista.baristanet import BaristaNet
from barista.ipc_utils import write_ipc_interface


def start_barista(solver_filename):
    pid = os.getpid()
    print("Daemon PID:", pid, file=sys.stderr)

    net = BaristaNet(solver_filename)

    ipc_interface = net.get_ipc_interface()
    ipc_interface_filename = os.path.join(os.getcwd(),
                                          "barista-%d.interface" % pid)
    write_ipc_interface(ipc_interface, ipc_interface_filename)
    print(ipc_interface_filename + "," + str(pid), file=sys.stdout)

    # Close stdout so the pipe can return
    sys.stdout.close()
    os.close(1)

    # Compute until the cows come home
    while True:
        net.full_pass()

if __name__ == "__main__":
    pid = os.getpid()
    print("Script PID:", pid, file=sys.stderr)
    solver_filename = next(sys.stdin).strip()

    fp = open('barista-log.'+str(os.getpid()), 'w+')
    with daemon.DaemonContext(stdout=sys.stdout,
                              stderr=fp,
                              working_directory=os.getcwd(),
                              files_preserve=[fp]):
        start_barista(solver_filename)
