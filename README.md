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

## Debugging Notes
- When running with Spark, Caffe hangs. I've isolated the exact Caffe function where this happens; it's in `caffe_cpu_gemm` in *caffe/utils/math_functions.cpp*. This function, in turn, calls cblas_sgemm. It's possible some of the pointers are corrupt? We'll find out.
- The problem actually is NOT with Spark. It happens when we run Barista as a daemon process, even without Spark. So what's the problem here?
    + Perhaps cblas_sgemm uses stdin somehow? It's been rerouted to devnull so this would break, or maybe wait for ever for a return value? But setting stdin to sys.stdin doesn't help at all.
    + Just kidding. It does work if I don't use Spark. The exact sequence of commands:

        `echo "models/solver.prototxt" | python -m barista.start`
        
        `python -m test._customer barista-####.interface`

    in particular, notice that we didn't open the log file in between. And it works if you run the customer test multiple times. And if you close the original terminal where you started barista. Also works if I open and close the log file. It seems to work almost no matter what.
    + Ok, now it also works with Spark. Not sure what happened when I was having issues. Will keep an eye out. Please report bugs.
