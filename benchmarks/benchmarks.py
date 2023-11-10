import subprocess
import re
import os
import sys

class TensorflowMLPerfSuite:
    timeout = 1800
    def setup_cache(self):
        pass
    def track_QPS_resnet50(self):
        out = subprocess.run(['bash "${WORKSPACE}"/asv-test/run-resnet50.sh'], shell=True, capture_output=True)
        sout = out.stdout.decode(sys.stdout.encoding).split('\n')
        serr = out.stderr.decode(sys.stdout.encoding).split('\n')
        for i, line in enumerate(sout):
            if 'qps=' in line:
                result = re.search('qps=(.*?),', line)
                finresult = result.group(1)
                return float(finresult)
        return 1.0
    track_QPS_resnet50.unit = "qps"
    def track_QPS_mobilenet(self):
        out = subprocess.run(['bash "${WORKSPACE}"/asv-test/run-mobilenet.sh'], shell=True, capture_output=True)
        sout = out.stdout.decode(sys.stdout.encoding).split('\n')
        serr = out.stderr.decode(sys.stdout.encoding).split('\n')
        for i, line in enumerate(sout):
            if 'qps=' in line:
                result = re.search('qps=(.*?),', line)
                finresult = result.group(1)
                return float(finresult)
        return 1.0
    track_QPS_mobilenet.unit = "qps"
