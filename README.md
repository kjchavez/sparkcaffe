# SparkCaffe
Package to allow you to use Caffe (almost) effortlessly inside the Apache Spark framework. 

## Getting Started

### Caffe
Please use the version of Caffe included as a submodule in this project. This is a temporary requirement. Once the rest of the system is running, I will make the necessary changes to make it compatible with the latest version of Caffe. If the `caffe` folder is empty, use

    git submodule init; git submodule update

to get the (slightly outdated and modified) Caffe source code. Build Caffe following the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html).

### Tests
Run the `barista` test script,

    python -m test._barista

and in a separate terminal, run the customer test script,

    python -m test._customer barista-ipc-interface.txt

You should see output on both terminals immediately, and both scripts should exit gracefully. If not, please check the **Common Problems** section or contact me at kevin.j.chavez@gmail.com.


## Todo
- Create a script that will fire up a barista daemon and return the ipc-interface
- Create very simple Spark App that simply applies batch gradient update, with random input data.
- Create MNIST digit classifier Spark App