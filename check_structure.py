import os
print("CWD:", os.getcwd())
print("List root:", os.listdir("."))
for root, dirs, files in os.walk("."):
    for f in files:
        print(os.path.join(root, f))
