import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from Utils.device import DEVICE as device

def ToUnknown():
    """Returns a function mapping any target to -1, the unknown class.

    Returns:
        function: lambda taking a target and returning -1.
    """
    return lambda target: -1

def hadamardProduct(features,classWeights,targets):
    """Element-wise (Hadamard) product between the features and the weights of
    the corresponding class.

    Args:
        features (torch.Tensor): features extracted from the model.
        classWeights (torch.Tensor): the model's per-class weights.
        targets (torch.Tensor): sample labels.

    Returns:
        torch.Tensor: the element-wise product.
    """
    #element-wise multiplication
    with torch.no_grad():
        return torch.mul(features,classWeights[targets])

def concatFeatures(preAttenuated,hadamard):
    """Concatenates the pre-attenuation features with the Hadamard product.

    Args:
        preAttenuated (torch.Tensor): original features, before attenuation.
        hadamard (torch.Tensor): the Hadamard product.

    Returns:
        torch.Tensor: concatenation along dimension 1 (features).
    """
    with torch.no_grad():
        return torch.concat((preAttenuated,hadamard),dim=1)

def calculateMeanConcatenatedVectors(concatVectors,targets,num_classes):
    """Mean concatenated vector of each class.

    Args:
        concatVectors (torch.Tensor): the concatenated vectors.
        targets (torch.Tensor): sample labels.
        num_classes (int): how many known classes exist.

    Returns:
        torch.Tensor: one mean vector per class, stacked.
    """
    feature_dim = concatVectors.shape[1]
    # Start at zero for every possible class.
    class_means = torch.zeros((num_classes, feature_dim), dtype=concatVectors.dtype)

    for c in range(num_classes):
        mask = targets == c
        if torch.any(mask):
            with torch.no_grad():
                class_means[c] = concatVectors[mask].mean(dim=0)

    return class_means

def GNL(ltmin,ltmax,logits):
    """Global Normalized Logits — min-max normalizes the logits into [0, 1].

    Uses the smallest and largest logits seen during fit. When that range is
    zero the normalization is undefined, and a sentinel is returned instead.

    Args:
        ltmin (float): smallest logit observed during fit.
        ltmax (float): largest logit observed during fit.
        logits (torch.Tensor): logits to normalize.

    Returns:
        torch.Tensor: logits normalized into [0, 1].
    """
    div = (ltmax - ltmin)
    if div == 0:
        norm = 999
    else:
        norm = (logits - ltmin) / div 
    with torch.no_grad():
        return torch.clamp(norm, 0, 1)

def calculateMagnitude(vectors):
    """L2 norm of each vector along dimension 1.

    Args:
        vectors (torch.Tensor): the input vectors.

    Returns:
        torch.Tensor: the L2 norm of each vector.
    """
    return torch.linalg.vector_norm(vectors,dim=1)

def costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,means,concatenatedVectors,max_logits_idx):
    """Cosine-based similarity between the concatenated vectors and the class
    means, in the form COSTARR uses:

        0.5 * (1 + cosine(vector, class_mean))

    which maps the cosine from [-1, 1] into [0, 1].

    Args:
        magnitudesMeans (torch.Tensor): magnitudes of each class mean vector.
        magnitudesConcatenatedVectors (torch.Tensor): magnitudes of the concatenated vectors.
        means (torch.Tensor): the class mean vectors.
        concatenatedVectors (torch.Tensor): the samples' concatenated vectors.
        max_logits_idx (torch.Tensor): index of the highest-logit class per sample.

    Returns:
        torch.Tensor: similarity values in [0, 1].
    """
    similarity = 1 + (torch.sum(concatenatedVectors*means[max_logits_idx],dim=1)/torch.mul(magnitudesMeans[max_logits_idx],magnitudesConcatenatedVectors))
    
    return 0.5 * similarity

def costarrPredict(model,testloader,train_calc):
    """Scores every sample in the loader with COSTARR.

    Runs the whole test loader, computing:
    1. logits normalized through GNL;
    2. the Hadamard product of features and class weights;
    3. similarity between the concatenated vectors and the class means;
    4. the final score, normalized_logits * similarity.

    Args:
        model (nn.Module): trained model exposing per-class weights.
        testloader (DataLoader): the evaluation data.
        train_calc (dict): 'min_logit', 'max_logit' and 'means', as produced
                           by costarrFit.

    Returns:
        tuple: (score, max_logits, max_logits_idx, all_targets)
            - score (torch.Tensor): the COSTARR score of each sample.
            - max_logits (torch.Tensor): the largest logit per sample.
            - max_logits_idx (torch.Tensor): the predicted class.
            - all_targets (torch.Tensor): the true labels.
    """
    all_logits = []
    all_features = []
    all_targets = []
    
    min_train_logit = train_calc["min_logit"]
    max_train_logit = train_calc["max_logit"]
    meanPerClassVector = train_calc["means"]
    for i,(X, y) in enumerate(testloader):
        
        X = X.to(device)
        with torch.no_grad():
            logits,features = model(X)

        all_logits.append(logits.cpu().detach())
        all_features.append(features.cpu().detach())
        all_targets.append(y.detach())
    
    all_logits = torch.cat(all_logits)
    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    max_logits,max_logits_idx = torch.max(all_logits,dim=1)

    normalized_logits = GNL(min_train_logit,max_train_logit,max_logits)
    hadamard = hadamardProduct(all_features,model.getPerClassWeights().cpu(),max_logits_idx)
    concatenatedVectors = concatFeatures(all_features,hadamard)

    magnitudesMeans = calculateMagnitude(meanPerClassVector)
    magnitudesConcatenatedVectors = calculateMagnitude(concatenatedVectors)

    score = normalized_logits * costarrSimilarity(magnitudesMeans,magnitudesConcatenatedVectors,meanPerClassVector,concatenatedVectors,max_logits_idx)

    return score,max_logits,max_logits_idx,all_targets

def threshold_predictions(score,max_logits_idx,epsilon):
    """Thresholds the scores into open-set predictions.

    A sample scoring below epsilon is rejected as unknown (-1); otherwise the
    originally predicted class is kept.

    Args:
        score (torch.Tensor): the COSTARR score of each sample.
        max_logits_idx (torch.Tensor): the predicted class per sample.
        epsilon (float): the rejection threshold.

    Returns:
        torch.Tensor: final predictions, where -1 marks unknown.
    """
    predictions = torch.where(score < epsilon, -1, max_logits_idx)

    return predictions

def costarrFit(model,trainloader,save_dir):
    """Measures the training-set statistics COSTARR needs at scoring time.

    Runs the whole train loader, keeping only the correctly classified samples,
    and computes:
    1. the smallest and largest logit, for the GNL normalization;
    2. the mean concatenated vector of each class;
    3. writes both to a .pt file.

    Only correctly classified samples are used, so the statistics describe what
    the model gets right rather than its errors.

    Args:
        model (nn.Module): trained model exposing per-class weights.
        trainloader (DataLoader): the training data.
        save_dir (str): path the .pt dictionary is written to.

    Returns:
        None: the dictionary with 'means', 'min_logit' and 'max_logit' is
              written to <save_dir>.
    """
    all_logits = []
    all_features = []
    all_targets = []

    for i,(X, y) in enumerate(trainloader):
        
        X = X.to(device)
        with torch.no_grad():
            logits,features = model(X)

        all_logits.append(logits.cpu().detach())
        all_features.append(features.cpu().detach())
        all_targets.append(y.detach())
    
    all_logits = torch.cat(all_logits)
    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    correctly_classified = torch.argmax(all_logits,1)==all_targets
    
    correct_logits = all_logits[correctly_classified]
    correct_features = all_features[correctly_classified]
    correct_targets = all_targets[correctly_classified]
    
    # Range used by the GNL normalization at scoring time.
    min_logit = torch.min(correct_logits)
    max_logit = torch.max(correct_logits)
    
    hadamard = hadamardProduct(correct_features,model.getPerClassWeights().cpu(),correct_targets)
    concatenatedVectors = concatFeatures(correct_features,hadamard)
    meanConcatenatedVectors = calculateMeanConcatenatedVectors(concatenatedVectors,correct_targets,len(np.unique(all_targets)))

    costarr = {"means":meanConcatenatedVectors,
               "min_logit":min_logit,
               "max_logit":max_logit}
    
    torch.save(costarr,save_dir)

def score_per_class(scores, labels):
    """Mean and standard deviation of the scores, broken down by class.

    A diagnostic helper: it is useful for picking a threshold by hand, since it
    shows how far apart the known and unknown score distributions sit.

    Args:
        scores (torch.Tensor or np.ndarray): the score of each sample.
        labels (torch.Tensor or np.ndarray): the true label of each sample.

    Returns:
        dict: {class: {'mean': float, 'std': float, 'n': int}}, where class -1
              holds the unknown (UUC) samples.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    unique_classes = np.unique(labels)
    result = {}

    for cls in sorted(unique_classes, key=lambda c: (c == -1, c)):
        mask = labels == cls
        class_scores = scores[mask]

        name = "Unknown (-1)" if cls == -1 else f"Class {cls}"
        result[name] = {
            "mean": float(np.mean(class_scores)),
            "std": float(np.std(class_scores)),
            "n": int(np.sum(mask)),
        }

    result["___OVERALL___"] = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n": len(scores),
    }

    return result
