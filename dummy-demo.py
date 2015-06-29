# Dummy Spark App demo
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles

import numpy as np
from barista.customer import Customer

conf = SparkConf().setAppName("Dummy Demo")
sc = SparkContext(conf=conf)

# Add prototxt files to Spark Context
sc.addFile("models/solver.prototxt")
sc.addFile("models/train_val.prototxt")

# Add barista module
sc.addPyFile("barista.zip")
sc.addPyFile("barista/start.py")


# Subclass generic barista Customer
class MyCustomer(Customer):
    def __init__(self, filename):
        compute_semaphore, model_semaphore, handles = \
            Customer.parse_ipc_interface_file(filename)
        Customer.__init__(self, compute_semaphore, model_semaphore, handles)

    def update_data(self):
        self.arrays['data'][:] = np.random.randn(*self.arrays['data'].shape)
        self.arrays['label'][:] = np.random.choice(
                                      xrange(10),
                                      size=self.arrays['label'].shape)

    def process_model(self):
        pass

# Create some dummy data
dataRDD = sc.parallelize(xrange(100))

# Create some barista instances
num_baristas = 2
start_script = 'python -m barista.start'
solver = SparkFiles.get("solver.prototxt")
interfaces = sc.parallelize([solver]*num_baristas, num_baristas) \
                  .pipe(start_script) \
                  .collect()


# Join the data
def train(interface, data):
    solver_filename, pid = interface.split(',')
    customer = MyCustomer(solver_filename)
    customer.run_transaction()
    grad_norm = np.linalg.norm(customer.arrays['conv1_dW'])
    return grad_norm

grad_norms = dataRDD.map(lambda x: train(interfaces[0], x)).collect()
print grad_norms

sc.stop()
