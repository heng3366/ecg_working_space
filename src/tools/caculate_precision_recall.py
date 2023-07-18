import torch
from sklearn.metrics import precision_score, recall_score



def caculate_precision_recall(y1, y2, label):
    """
    输入：y1、y2为iso电平和st电平，label为真实数据的7个元素的标签
    输出：precision和recall
    """
    forward_output = torch.cat((y1, y2), dim=1).to("cuda")
    st = (forward_output[:, 1] - forward_output[:, 0]).
