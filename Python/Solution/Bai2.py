import math

def Tinh_F1_Score(tp, fp, fn):
    if not isinstance(tp, int):
        print('tp must be int')
        return
    
    if not isinstance(fp,int):
        print('fp must be int')
        return
    
    if not isinstance(fn, int):
        print('fn must be int')
        return
    
    
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    F1_Score = 2 * (Precision * Recall)/(Precision + Recall)
    
    print("Precision = ", Precision)
    print("Recall = ", Recall)
    print("F1-Score = ", F1_Score)
    
    
