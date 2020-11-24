import numpy as np

class Loss:
   
    def get_loss(self, y_pred, y_true):
        # Same function for all loss functions
        lst_neg_log_confidence = self.calculate(y_pred, y_true) 
        # Mean of the losses     
        return np.mean(lst_neg_log_confidence)  

    def get_accuracy(elf, y_pred, y_true):
        # Exctract index of highest likelyhood neuron
        predictions = np.argmax(y_pred, axis=1)
        # Compare with true results
        return np.mean(predictions==y_true)
        

class CrossEntropy(Loss):
    def calculate(self, y_pred, y_true):
        # Prevent division by 0 (cut both sides for safety)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # For Categorical ex: [1,2,0,2,0,0]
        if len(y_true.shape) == 1:
            lst_conf_in_correct_out = y_pred_clipped[:, y_true]
        # For Encoded ex: [0,0,1][0,1,0]
        elif len(y_true.shape) == 2:
            lst_conf_in_correct_out = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses list negative NATURAL log likelyhood
        return np.log(lst_conf_in_correct_out)
    def backpropogate():
        return -y_true/y_pred