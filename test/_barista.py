from barista.baristanet import BaristaNet
import numpy as np

# To access the memory address for the buffer of a numpy array, use
#     addr, flag = arr.__array_interface__['data']


def write_ipc_interface(barista_net, filename):
    comp_sem, model_sem, interface = barista_net.get_ipc_interface()
    with open(filename, 'w') as fp:
        print >> fp, comp_sem
        print >> fp, model_sem
        for name, shape, dtype in interface:
            print >> fp, ':'.join([name, str(shape), dtype])


def main(solver):
    barista_net = BaristaNet(solver)
    write_ipc_interface(barista_net, 'barista-ipc-interface.txt')

    print "Barista running. Waiting on compute semaphore:",
    print barista_net.compute_semaphore

    print "Conv1_dW norm:", np.linalg.norm(barista_net.net.params['conv1'][0].diff)
    print "Data norm:", np.linalg.norm(barista_net.net.blobs['data'].data)
    for i in range(10):
        barista_net.full_pass()
        print "Completed full pass #%d" % i
        print "Conv1_dW norm:", np.linalg.norm(barista_net.net.params['conv1'][0].diff)
        print "Data norm:", np.linalg.norm(barista_net.net.blobs['data'].data)


if __name__ == "__main__":
    main("models/solver.prototxt")
