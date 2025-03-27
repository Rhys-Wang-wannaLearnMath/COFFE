import docker
import os
from datetime import datetime

class SandBox(object):
    def __init__(self, workdir, perf_path):
        self.workdir = workdir
        self.image_tag = "coffe"
        self.client = docker.from_env()
        self.perf_path = perf_path

    def _run(self, args):
        self.run(args[0], args[1], args[2])

    def run(self, command, worker_id, timeout):
        container_workdir = '/data'
        mount = docker.types.Mount(target = container_workdir, source = self.workdir, type = 'bind', read_only = False)

        buf_prefix = 'stdbuf -i0 -o0 -e0'
        timeout_prefix = 'timeout {}'.format(timeout)
        command = " ".join([buf_prefix, timeout_prefix, command])
        exit_code = 0

        try:
            container = self.client.containers.run(image=self.image_tag, command=['/bin/bash', '-c', command], detach=True, security_opt=["seccomp=" + open(self.perf_path, "r").read()], network_mode='host', mounts=[mount])
        except Exception as e:
            print(f"Worker {worker_id}: container running failed, reason: {e}")
            os.system(f'echo "[{datetime.now()}]Worker {worker_id} running failed." >> {self.workdir}/ERROR_{worker_id}')
            exit_code = -1
            return exit_code

        try:
            exit_code = container.wait(timeout = timeout + 100, condition = 'not-running')['StatusCode']
        except Exception as e:
            print(f'Worker {worker_id}: Container time out, killed.')
            try:
                if container.status == 'running':
                    container.kill()
            except Exception as e:
                print(e)
                os.system(f'echo "[{datetime.now()}]Worker {worker_id} timeout" >> {self.workdir}/ERROR_{worker_id}')
            exit_code = -1
        finally:
            try:
                log = container.logs(stdout = True, stderr = True).decode(encoding = 'utf-8', errors = 'ignore').strip()
                with open(os.path.join(self.workdir, f'CHECK_LOG_{worker_id}'), 'w', encoding = 'utf-8') as lf:
                    lf.write(log)
                container.remove(v=True, force = True)
            except Exception as e:
                print(e)
                os.system(f'echo "[{datetime.now()}]Worker {worker_id} logerror" >> {self.workdir}/ERROR_{worker_id}')
        return exit_code
