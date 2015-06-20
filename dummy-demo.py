# Dummy Spark App demo
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles

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
    def update_data(self):
        pass

    def process_model(self):
        pass

# Create some dummy data
dataRDD = sc.parallelize(xrange(1000))

# Create some barista instances
num_baristas = 2
start_script = 'python -m barista.start'
solver = SparkFiles.get("solver.prototxt")
interfaces = sc.parallelize([solver]*num_baristas, num_baristas) \
                  .pipe(start_script) \
                  .collect()

# Join the data

print interfaces

sc.stop()