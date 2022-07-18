class TensorflowTimeSuite:
    timeout = 600
    def setup(self):
        import subprocess
        subprocess.run(['sudo','/home/theodore/resmod/mlperf-setup.sh'])
        pass
    def time_resnet50(self):
        import subprocess
#         subprocess.run(['sudo','/home/theodore/resmod/resnet50.sh','restart']
        subprocess.run(['/home/theodore/resmod/run-resnet50.sh'])
        pass
    def time_mobilenet(self):
        import subprocess
        subprocess.run(['/home/theodore/resmod/run-mobilenet.sh'])
        pass
    def time_ssd_resnet34(self):
        import subprocess
        subprocess.run(['/home/theodore/resmod/run-ssd-resnet34'])
        pass
