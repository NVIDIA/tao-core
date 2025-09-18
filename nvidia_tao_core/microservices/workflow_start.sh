#!/usr/bin/env bash
umask 0
python3 $(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia_tao_core/microservices/workflow.py