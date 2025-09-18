
#!/bin/bash

# Generate _pb2.py files

apt-get install -y protobuf-compiler

protoc nvidia_tao_core/proto/*.proto --python_out=.