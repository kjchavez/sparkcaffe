from __future__ import print_function
import sys
import os
import posix_ipc
from posix_ipc import O_CREAT, O_EXCL

import numpy as np
import caffe
import barista

from barista.ipc_utils import create_shmem_ndarray, prepend_pid


class SharedData:
    _null_array = None

    def __init__(self, net, layer_idx):
        self.data = None
        self.label = None
        self.shmem = []

        self.name = net._layer_names[layer_idx]
        handles = self.name.split('-', 1)

        # print "Allocating shared memory for %s." % handles[0]
        handle = prepend_pid(handles[0])
        shmem, arr = create_shmem_ndarray('/'+handle,
                                          net.blobs[handles[0]].data.shape,
                                          np.float32,
                                          flags=posix_ipc.O_CREAT)
        self.data = arr
        self.shmem.append(shmem)

        if len(handles) == 2:
            handle = prepend_pid(handles[1])
            shmem, arr = create_shmem_ndarray('/'+handle,
                                              net.blobs[handles[1]].data.shape,
                                              np.float32,
                                              flags=posix_ipc.O_CREAT)
            self.label = arr
            self.shmem.append(shmem)
            net.set_input_arrays(self.data, self.label, layer_idx)

        else:
            if SharedData._null_array is None:
                SharedData._null_array = np.empty(
                                             (self.data.shape[0], 1, 1, 1),
                                             dtype=np.float32)

            print("[SharedData] Warning: didn't specify a handle for the "
                  "label in layer", layer_idx, ". Should not be used in net.",
                  file=sys.stderr)

            net.set_input_arrays(self.data, SharedData._null_array, layer_idx)

    def sync(self):
        pass  # Nothing to do

    def get_handles(self):
        handles = [(self.shmem[0].name, self.data.shape, str(self.data.dtype))]
        if self.label is not None:
            handles.append((self.shmem[1].name,
                            self.label.shape, str(self.label.dtype)))
        return handles

    def __del__(self):
        for shmem in self.shmem:
            shmem.close_fd()
            shmem.unlink()


class SharedParameter:
    POSTFIX = ['W', 'b', 'c', 'd']

    def __init__(self, net, param_name):
        self.caffe_params = []
        self.shared_params = []
        self.shmem = []

        for i, param in enumerate(net.params[param_name]):
            # Typically, we'll have two, a weight and bias.
            handle = prepend_pid(param_name + '_' + SharedParameter.POSTFIX[i])
            shmem, arr = create_shmem_ndarray('/' + handle,
                                              param.data.shape,
                                              np.float32,
                                              flags=posix_ipc.O_CREAT)

            self.caffe_params.append(param.data)
            self.shared_params.append(arr)
            self.shmem.append(shmem)

        self.sync_from_caffe()

    def sync_to_caffe(self):
        for i in xrange(len(self.shared_params)):
            np.copyto(self.caffe_params[i], self.shared_params[i],
                      casting='no')

    def sync_from_caffe(self):
        for i in xrange(len(self.shared_params)):
            np.copyto(self.shared_params[i], self.caffe_params[i],
                      casting='no')

    def get_handles(self):
        handles = [(self.shmem[i].name, self.shared_params[i].shape, str(self.shared_params[i].dtype))
                   for i in xrange(len(self.shared_params))]

        return handles

    def __del__(self):
        for shmem in self.shmem:
            shmem.close_fd()
            shmem.unlink()


class SharedGradient:
    POSTFIX = ['dW', 'db', 'dc', 'dd']

    def __init__(self, net, param_name):
        self.caffe_grads = []
        self.shared_grads = []
        self.shmem = []

        for i, param in enumerate(net.params[param_name]):
            # Typically, we'll have two, a weight and bias.
            handle = prepend_pid(param_name + '_' + SharedGradient.POSTFIX[i])
            shmem, arr = create_shmem_ndarray('/' + handle,
                                              param.diff.shape,
                                              np.float32,
                                              flags=posix_ipc.O_CREAT)

            self.caffe_grads.append(param.diff)
            self.shared_grads.append(arr)
            self.shmem.append(shmem)

    def sync_from_caffe(self):
        for i in xrange(len(self.shared_grads)):
            np.copyto(self.shared_grads[i], self.caffe_grads[i],
                      casting='no')

    def sync_to_caffe(self):
        """ Warning: generally don't want to do this unless you know what
        you're doing."""
        for i in xrange(len(self.shared_grads)):
            np.copyto(self.caffe_grads[i], self.shared_grads[i],
                      casting='no')

    def get_handles(self):
        handles = [(self.shmem[i].name, self.shared_grads[i].shape, str(self.shared_grads[i].dtype))
                   for i in xrange(len(self.shared_grads))]

        return handles

    def __del__(self):
        for shmem in self.shmem:
            shmem.close_fd()
            shmem.unlink()


class BaristaNet:
    def __init__(self, solver_file, architecture_file=None, model_file=None):
        solver = caffe.SGDSolver(solver_file)
        self.net = solver.net

        if model_file and architecture_file:
            # Load the parameters from file
            pretrained_net = caffe.Net(architecture_file, model_file)
            self.net.copy_from(pretrained_net)

        # Allocate shared memory for all memory data layers in the network
        self.shared_data = []
        self.shared_params = []
        self.shared_grads = []
        self.shmem = {}

        for idx, layer in enumerate(self.net.layers):
            if layer.type == barista.MEMORY_DATA_LAYER:
                self.shared_data.append(SharedData(self.net, idx))

        for param in self.net.params:
            self.shared_params.append(SharedParameter(self.net, param))
            self.shared_grads.append(SharedGradient(self.net, param))

        self.batch_size = self.shared_data[0].data.shape[0]

        # Allow convenient access to certain Caffe net properties
        self.blobs = self.net.blobs

        # Create semaphore for interprocess synchronization
        self.compute_semaphore = posix_ipc.Semaphore(
                                     "/"+prepend_pid("compute"),
                                     flags=O_CREAT | O_EXCL)
        self.model_semaphore = posix_ipc.Semaphore(
                                   "/"+prepend_pid("model"),
                                   flags=O_CREAT | O_EXCL)

    def sync_parameters(self):
        """ Sync the parameters from shared memory to Caffe memory. """
        for param in self.shared_params:
            param.sync_to_caffe()

    def sync_gradients(self):
        """ Sync gradients from Caffe memory to shared memory. """
        for grad in self.shared_grads:
            grad.sync_from_caffe()

    def full_pass(self):
        print('Waiting for compute semaphore', self.compute_semaphore.name, file=sys.stderr)
        self.compute_semaphore.acquire()
        self.sync_parameters()
        self.net.forward()
        print('Forward pass complete', file=sys.stderr)
        self.net.backward()
        self.sync_gradients()
        print('Backward pass complete', file=sys.stderr)
        self.model_semaphore.release()
        print('Model semaphore released', file=sys.stderr)

    def forward(self, end=None):
        self.net.forward(end=end)

    def get_ipc_interface(self):
        interface = []
        for data in self.shared_data:
            interface += data.get_handles()

        for param in self.shared_params:
            interface += param.get_handles()

        for grad in self.shared_grads:
            interface += grad.get_handles()

        return (self.compute_semaphore.name,
                self.model_semaphore.name,
                interface)

    def ipc_interface_str(self):
        interface_str = ""
        comp_sem, model_sem, interface = self.get_ipc_interface()
        interface_str += comp_sem + '\n'
        interface_str += model_sem + '\n'
        for name, shape, dtype in interface:
            interface_str += ':'.join([name, str(shape), dtype]) + '\n'

        return interface_str

    def __del__(self):
        self.compute_semaphore.unlink()
        self.compute_semaphore.close()
        self.model_semaphore.unlink()
        self.model_semaphore.close()
