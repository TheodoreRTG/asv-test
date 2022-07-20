class TensorflowMLPerfSuite:
    timeout = 1800
    def setup(self):
        import subprocess
        subprocess.run(['/root/asv-test/asv-conf/asv-test/mlperf-setup.sh'],  shell=True)
        pass
    def track_QPS_resnet50(self):
        import subprocess
        import re
        import os
        r = open('results-resnet50.txt', 'w')
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-resnet50.sh'], stdout=r)
        r.close()
        for i, line in enumerate(open("results-resnet50.txt")):
            if 'qps=' in line:
                result = re.search('qps=(.*?),', line)
                finresult = result.group(1)
                os.remove("results-resnet50.txt")
                return float(finresult)
    track_QPS_resnet50.unit = "qps"
    def track_QPS_mobilenet(self):
        import subprocess
        import re
        import os
        m = open('results-mobilenet.txt', 'w')
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-mobilenet.sh'], stdout=m)
        m.close()
        for i, line in enumerate(open("results-mobilenet.txt")):
            if 'qps=' in line:
                result = re.search('qps=(.*?),', line)
                finresult = result.group(1)
                os.remove("results-mobilenet.txt")
                return float(finresult)
    track_QPS_mobilenet.unit = "qps"
#    def QPS_ssd_resnet34_list(self):
#        import subprocess
#        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-ssd-resnet34.sh'], ">>", "results-ssd-resnet-34.txt", shell=True)
#        for i, line in enumerate(open("results-ssd-resnet34.txt")):
#            if 'qps=' in line:
#                result = re.search('qps=(.*),', line)
#        os.remove("results-ssd-resnet34.txt")
#        return result

