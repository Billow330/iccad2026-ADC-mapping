#!/usr/bin/env python3
# Simple test script
import os
with open('/home/ubuntu/iccad2026_bxkj/NeuroSim/pytest_result.txt', 'w') as f:
    f.write('python_works\n')
    f.write(f'pid={os.getpid()}\n')
print("wrote file")
