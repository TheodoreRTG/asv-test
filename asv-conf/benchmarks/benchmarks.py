class TensorflowMLPerfSuite:
    timeout = 600
    def setup(self):
        import subprocess
        subprocess.run(['/root/asv-test/asv-conf/asv-test/mlperf-setup.sh'],  shell=True)
        pass
    def QPS_resnet50_list(self):
        import subprocess
        import re
        import os
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-resnet50.sh'], ">>", "results-resnet50.txt", shell=True)
        for i, line in enumerate(open("results-resnet5.txt")):
            if 'qps=' in line:
                result = re.search('qps=(.*),', line)
        os.remove("results-resnet50.txt")
        return result
    def QPS_mobilenet_list(self):
        import subprocess
        import re
        import os
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-mobilenet.sh'], ">>", "results-mobilenet.txt", shell=True)
        for i, line in enumerate(open("results-mobilenet.txt")):
            if 'qps=' in line:
                result = re.search('qps=(.*),', line)
        os.remove("results-mobilenet.txt")
        return result
#    def QPS_ssd_resnet34_list(self):
#        import subprocess
#        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-ssd-resnet34.sh'], ">>", "results-ssd-resnet-34.txt", shell=True)
#        for i, line in enumerate(open("results-ssd-resnet34.txt")):
#            if 'qps=' in line:
#                result = re.search('qps=(.*),', line)
#        os.remove("results-ssd-resnet34.txt")
#        return result
