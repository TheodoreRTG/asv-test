import subprocess
import re
import os
import sys

class TensorflowMLPerfSuite:
    timeout = 1800
    def setup_cache(self):
        pass
        # print("Running setup")

        # out = subprocess.run(['bash /home/ubuntu/sources/theodore/asv-test/mlperf-setup.sh'],  shell=True, capture_output=True)

        # print("Stdout: ", out.stdout)
        # print("Stderr: ", out.stderr)
        # pass

    # def time_resnet50(self):
    #     out = subprocess.run(['bash /home/ubuntu/sources/theodore/asv-test/run-resnet50.sh'], shell=True)

    def track_QPS_resnet50(self):
        # import subprocess
        # import re
        # import os
        # print("Running benchmarks")
        # r = open('results-resnet50.txt', 'w')
        out = subprocess.run(['bash /home/ubuntu/sources/theodore/asv-test/run-resnet50.sh'], shell=True, capture_output=True)
        # print("Stdout: ", out.stdout)
        # print("Stderr: ", out.stderr)
        sout = out.stdout.decode(sys.stdout.encoding).split('\n')
        serr = out.stderr.decode(sys.stdout.encoding).split('\n')
        # with open('/home/ubuntu/sources/theodore/asv-test/output.me', 'w') as fd:
        #     fd.write('\n'.join(sout))
        # with open('/home/ubuntu/sources/theodore/asv-test/output.err.me', 'w') as fd:
        #     fd.write('\n'.join(serr))
        for i, line in enumerate(sout):
            if 'qps=' in line:
                result = re.search('qps=(.*?),', line)
                finresult = result.group(1)
                return float(finresult)
        # r.close()
        # for i, line in enumerate(open("results-resnet50.txt")):
        #     if 'qps=' in line:
        #         result = re.search('qps=(.*?),', line)
        #         finresult = result.group(1)
        #         os.remove("results-resnet50.txt")
        #         return float(finresult)
        return 1.0
    track_QPS_resnet50.unit = "qps"
    # def track_QPS_mobilenet(self):
    # #     import subprocess
    # #     import re
    # #     import os
    # #     m = open('results-mobilenet.txt', 'w')
    #     out = subprocess.run(['bash /home/ubuntu/sources/theodore/asv-test/run-mobilenet.sh'], shell=True, capture_output=True)
    #     # print("Stdout: ", out.stdout)
    #     # print("Stderr: ", out.stderr)
    #     sout = out.stdout.decode(sys.stdout.encoding).split('\n')
    #     serr = out.stderr.decode(sys.stdout.encoding).split('\n')
    #     with open('/home/ubuntu/sources/theodore/asv-test/output.me', 'w') as fd:
    #         fd.write('\n'.join(sout))
    #     with open('/home/ubuntu/sources/theodore/asv-test/output.err.me', 'w') as fd:
    #         fd.write('\n'.join(serr))
    #     for i, line in enumerate(sout):
    #         if 'qps=' in line:
    #             result = re.search('qps=(.*?),', line)
    #             finresult = result.group(1)
    #             return float(finresult)
    # #     subprocess.run(['/root/asv-test/asv-conf/asv-test/run-mobilenet.sh'], stdout=m)
    # #     m.close()
    # #     for i, line in enumerate(open("results-mobilenet.txt")):
    # #         if 'qps=' in line:
    # #             result = re.search('qps=(.*?),', line)
    # #             finresult = result.group(1)
    # #             os.remove("results-mobilenet.txt")
    # #             return float(finresult)
    # track_QPS_mobilenet.unit = "qps"
#    def QPS_ssd_resnet34_list(self):
#        import subprocess
#        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-ssd-resnet34.sh'], ">>", "results-ssd-resnet-34.txt", shell=True)
#        for i, line in enumerate(open("results-ssd-resnet34.txt")):
#            if 'qps=' in line:
#                result = re.search('qps=(.*),', line)
#        os.remove("results-ssd-resnet34.txt")
#        return result
