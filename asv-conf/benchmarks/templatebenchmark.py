# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class MobilenetTimeSuite:
#    """
#    An example benchmark that times the performance of various kinds
#    of iterating over dictionaries in Python.
 #   """
    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]

#class TestBench:
#    def setup(self):
#        import os
#        pass
#    def time_TEST(self):
#        print('hello')
#        pass
#class MemSuite:
#    def mem_list(self):
#        return [0] * 256

##class BenchmarkTens:
##    timeout = 600
##    def setup(self):
#        import subprocess
#        subprocess.call("pwd")
#        subprocess.call("ls")
#        return subprocess.call("ls")
##         subprocess.run(['sudo','/home/theodore/resmod/resnet50.sh'])
#        return(rc)
#        pass
##    def time_resnet50(self):
##         import subprocess
#         subprocess.run(['sudo','/home/theodore/resmod/resnet50.sh','restart'])
##         subprocess.run(['/home/theodore/resmod/run-resnet50.sh'])
##         pass

#class BenchmarkTens:
#    def setup(self):
#        self.d = {}
#        import subprocess
#        rc = subprocess.call("./home/theodore/santen/mlperf/test-definitions/automated/linux/tensorflow/imagenet-resnet50.sh -a /home/theodore/santen -t false")
#        pass
#
#    def time_resnet50(self):
#        ac = subprocess.call("./home/theodore/santen/mlperf/src/inference/vision/classification_and_detection/run_local.sh tf resnet50")
#        pass
