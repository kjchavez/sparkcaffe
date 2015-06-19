import sys
from barista.customer import Customer
import numpy as np

# Subclass Customer
class DDQCustomer(Customer):
    def update_data(self):
        self.state[...] = np.random.randn(*self.state.shape)
        print "Update data"

    def process_model(self):
        print "Pull out gradients"
        print np.linalg.norm(self.state)


def main(compute_semaphore, model_semaphore, handles):
    ddq = DDQCustomer(compute_semaphore, model_semaphore, handles)
    for _ in range(5):
        ddq.run_transaction()


def parse_ipc_handle_file(filename):
    with open(filename) as fp:
        compute_semaphore = next(fp).strip()
        model_semaphore = next(fp).strip()
        handles = []
        for line in fp:
            name, shape, dtype = line.split(':')
            shape = tuple(int(x) for x in shape[1:-1].split(','))
            handles.append((name.strip(), shape, dtype.strip()))

    return compute_semaphore, model_semaphore, handles

if __name__ == "__main__":
    compute_semaphore, model, handles = parse_ipc_handle_file(sys.argv[1])
    main(compute_semaphore, model, handles)

