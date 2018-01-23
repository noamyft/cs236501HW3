import numpy as np

def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """

    # create binary vector for feature selection
    x = np.mat(x)
    features_select = [False for i in range(x.shape[1])]
    num_features_selected = 0


    while num_features_selected < k:
        max_score = 0
        max_feature = 0
        for i in range(len(features_select)):
            # examine each unselected feature
            if not features_select[i]:
                features_select[i] = True
                current_score = score(clf,x[:,features_select],y)
                if current_score > max_score:
                    max_score = current_score
                    max_feature = i
                features_select[i] = False

        # add the best feature
        features_select[max_feature] = True
        num_features_selected += 1

    return [i for i in range(len(features_select)) if features_select[i]]