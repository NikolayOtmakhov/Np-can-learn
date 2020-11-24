import numpy as np

class Setup():

    def output_layer_check(y):
        y = np.array(y)
        if len(y.shape)>1:
            raise Exception('List for outputs not implemented yet')    
        if y.shape[0] == 0:
            raise Exception('Outputs list empty')  
        try:
            unique_y = np.unique(y)
            n_unique_y = unique_y.shape[0]
        except:
            raise Exception('Outputs must be a 1D list/array')
        return y, n_unique_y

    def input_layer_check(x):
        x = np.array(x)
        if len(x.shape)>2:
            raise Exception('x inputs/features have more then 2 dimensions')
        try:
            features = x.shape[1]
        except:
            try:
                if x.shape[0] != 0:
                    features = 1
                else:
                    raise Exception('No inputs in list/array')
            except:
                raise Exception('x inputs must be a list/array')
        return x, features

    def hidden_layer_check(neurons_hidden):
        hidden_layer_list = 0
        hidden_layer_count = 0
        if len(np.array(neurons_hidden).shape)>1:
            raise Exception('neurons_hidden must be an int or 1D list/array')
        if neurons_hidden == 0:
            # if no hidden layers
            pass
        else:
            try: 
                # if input is list
                hidden_layer_count = np.array(neurons_hidden).shape[0]
                hidden_layer_list = np.array(neurons_hidden)
            except:
                if isinstance(neurons_hidden, int):
                    # if input is an int
                    hidden_layer_list =  np.array([neurons_hidden]) 
                    hidden_layer_count = 1
                else:
                    raise Exception('neurons_hidden must be an int or 1D list/array')
        return hidden_layer_list, hidden_layer_count