class TensorflowTimeSuite:
    timeout = 600
    def setup(self):
        import subprocess
        subprocess.run(['sudo','/root/asv-test/asv-conf/asv-test/mlperf-setup.sh'])
        pass
    def time_resnet50(self):
        import subprocess
#         subprocess.run(['sudo','/root/asv-test/asv-conf/asv-test/resnet50.sh','restart'])
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-resnet50.sh'])
        pass
    def time_mobilenet(self):
        import subprocess
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-mobilenet.sh'])
        pass
    def time_ssd_resnet34(self):
        import subprocess
        subprocess.run(['/root/asv-test/asv-conf/asv-test/run-ssd-resnet34.sh'])
        pass
