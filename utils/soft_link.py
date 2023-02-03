import os

def create_softlink(path_a, path_b):
    # Use os.path.abspath to convert relative paths to absolute paths
    path_a = os.path.abspath(path_a)
    path_b = os.path.abspath(path_b)

    # Detect if path_b is occupied
    if os.path.exists(path_b):
        os.remove(path_b)
        
    # Use os.symlink to create a soft link from path_a to path_b
    
    os.symlink(path_a, path_b)
    
    # Print the link created to confirm
    # print("Softlink created from {} to {}".format(path_a, path_b))


if __name__ == '__main__':
    create_softlink('/home/zhuang/Code/BFT-benchmark/utils/__init__.py', '/home/zhuang/__init__.py')