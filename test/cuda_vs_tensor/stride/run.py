import os

for range_type in range(5):
    size = 16
    cmd = "build/main 128 {} {} {}"
    while size <= 8192:
        print(cmd.format(size, size, range_type))
        os.system(cmd.format(size, size, range_type))
        
        size = size * 2