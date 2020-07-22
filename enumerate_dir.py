for root, dirs, files in os.walk(folder):
    for i,f in enumerate(files):
        absname = os.path.join(root, f)
        newname = os.path.join(root, str(i))
        os.rename(absname, newname)