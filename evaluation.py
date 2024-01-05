import numpy as np
from sklearn.metrics import auc

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def evaluate(labels, distances, ts, fmr_p):
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc = \
    evaluate_sf(
        distances=distances,
        labels=labels,
        far_target=1e-3
    )

    global_h = find_nearest(false_positive_rate, fmr_p)

    tpr_1e3 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-03))]
    tpr_1e4 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-04))]
    fpr_95 = false_positive_rate[np.argmin(np.abs(true_positive_rate - 0.95))]
    fnr = false_negative_rate
    fpr = false_positive_rate
    sub = np.abs(fnr - fpr)
    h = np.min(sub[np.nonzero(sub)])
    h = np.where(sub == h)[0][0]


    print("------------------------------------------------Single fold---------------------------------------\n")

    # Print statistics and add to log
    print("Accuracy: {:.4f}\tPrecision {:.4f}\tRecall {:.4f}\t"
          "ROC Area Under Curve: {:.4f}\t".format(
                np.mean(accuracy),
                np.mean(precision),
                np.mean(recall),
                roc_auc
            )
    )


    print('fpr at tpr 0.95: {},  tpr at fpr 0.001: {}, tpr at fpr 0.0001: {}'.format(fpr_95,tpr_1e3,tpr_1e4))
    print('At FNR = FPR: FNR = {}, FPR = {}'.format(fnr[h],fpr[h]))
# with open('logs/cc_tpr_fpr_{}_{}.txt'.format(logfname, ts), 'a') as f:
#             f.writelines(''.format()

    with open('logs/log_stats_{}.txt'.format(ts), 'a') as f:

        f.writelines("--------------------------single fold-------------------------------\n"
          "\nAccuracy: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
          "ROC Area Under Curve: {:.4f}\t"
          "fpr at tpr 0.95: {},  tpr at fpr 0.001: {} | At FNR = FPR: FNR = {}, FPR = {}".format(
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                fpr_95,
                tpr_1e3,
                fnr[h],
                fpr[h]
            ) + '\n'
        )
        
    return tpr_1e3,tpr_1e4,fpr_95, h, global_h, (fnr[h]+fpr[h])/2

def evaluate_sf(distances, labels, far_target=1e-3):
    """Evaluates on the  dataset using single-fold validation based on the Cosine
    distance as a metric.
    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Acceptance Rate) (TP/(TP+FN)) when
    the rate that faces are incorrectly accepted (False Acceptance Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Acceptance Rate to calculate the True Acceptance Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False Acceptance Rate values per each fold in cross validation set.
    """

    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 2.0, 0.001)
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy = \
        calculate_roc_values_sf(
            thresholds=thresholds_roc, distances=distances, labels=labels
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc


def calculate_roc_values_sf(thresholds, distances, labels):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)

    true_positive_rates = np.zeros((num_thresholds))
    false_positive_rates = np.zeros((num_thresholds))
    false_negative_rates = np.zeros((num_thresholds))
    
    accuracies = np.zeros((num_thresholds))

    test_set = np.arange(num_pairs)
    
    

        # Test on test set using the best distance threshold
    for threshold_index, threshold in enumerate(thresholds):
        true_positive_rates[threshold_index], false_positive_rates[threshold_index], false_negative_rates[threshold_index], _, _,\
            accuracies[threshold_index] = calculate_metrics_sf(
                threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set])
    best_threshold_index = np.argmax(accuracies)
    _, _, _, precision, recall, accuracy = calculate_metrics_sf(
        threshold=thresholds[best_threshold_index], dist=distances, actual_issame=labels[test_set]
    )


    return true_positive_rates, false_positive_rates, false_negative_rates, precision, recall, accuracy


def calculate_metrics_sf(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy


def calculate_FNR_FPR_thresh(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)
    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy

