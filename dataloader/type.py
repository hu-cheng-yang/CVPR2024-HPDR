import os

def getType(name, image_path):
    if name == "OULU":
        typeidtrue = image_path.split('/')[-2].split('_')[1]
        tid = 0
        if typeidtrue == '2' or typeidtrue == '3':
            tid = 1
        else:
            tid = 2
        return tid
    
    if name == 'CASIA':
        typeidtrue = image_path.split('/')[-1][: -9]
        tid = 0
        if typeidtrue == '7' or typeidtrue == '8' or typeidtrue == 'HR_4':
            tid = 2
        else:
            tid = 1
        return tid

    if name == 'MSU':
        tid = 0
        if "photo" in image_path:
            tid = 1
        else:
            tid = 2
        return tid
    
    if name == "idiap":
        tid = 0
        if "photo" in image_path:
            tid = 1
        else:
            tid = 2
        return tid
    
    raise Exception("Name must be in OULU, MSU, idiap or CASIA!")

