COST_FN = 10000
COST_FP = 500

def business_cost(y_true: int, y_pred: int) -> int:
    if y_true == 1 and y_pred == 0:
        return COST_FN
    if y_true == 0 and y_pred == 1:
        return COST_FP
    return 0
