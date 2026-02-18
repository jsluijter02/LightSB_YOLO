## This file is for AFTER running an experiment. It uses the generated metrics.json,
# and the prediction text files to get results.
import json

def add_prec_recall_f1(metrics_json_path):
    with open(metrics_json_path) as f:
        per_image_metrics = json.load(f)
    
    tp = sum(image['tp'] for image in per_image_metrics.values())
    fp = sum(image['fp'] for image in per_image_metrics.values())
    fn = sum(image['fn'] for image in per_image_metrics.values())

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))

    return {'precision': precision, 'recall': recall, 'f1': f1}

# calculates the diff per image 
def compare_metrics(metrics_json_1, metrics_json_2, compare_method = 'f1'):
    with open(metrics_json_1) as file1:
        json_1 = json.load(file1)
    with open(metrics_json_2) as file2:
        json_2 = json.load(file2)

    preds_1 = json_1.values()
    preds_2 = json_2.values()

    print(preds_1)