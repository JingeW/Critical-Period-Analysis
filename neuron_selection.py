import os
# import numpy as np
import pickle as pkl
import scipy.stats as stats

def oneWay_ANOVA(fullMatrix, sample_num, class_num, alpha):
    ''' ANOVA test for each neuron among all classes
            H0: The testing neuron has the same response to different images
            H1: The testing neuron has at least one different response to different images
        Parameters:
            fullMatrix: array, is the full matrix which consists of all featrue maps in the same layer
            sample_num: int, number of sample per class
            class_num: int, number of class
            alpha: float, significant level
        Return:
            sig_neuron_ind: a list store the index of neuron which ANOVA tested significantly
            non_sig_neuron_ind: a list store the index of neuron which ANOVA tested insignificantly
    '''
    row, col = fullMatrix.shape
    print('ANOVA iter range:', col)
    sig_neuron_ind = []
    non_sig_neuron_ind = []
    for i in range(col):
        neuron = fullMatrix[:, i]
        d = [neuron[i * sample_num: i * sample_num + sample_num] for i in range(class_num)]
        p = stats.f_oneway(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
                           d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
                           d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
                           d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
                           d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]
        if p < alpha:
            sig_neuron_ind.append(i)
        else:
            non_sig_neuron_ind.append(i)
    return sig_neuron_ind, non_sig_neuron_ind

if __name__ == "__main__":
    # name_list = ['fullFace_baseModel', 'eyesFoveated_baseModel', 'mouthFoveated_baseModel']
    name_list = ['fullOnEyes_epoch15', 'fullOnEyes_epoch80', 'fullOnMouth_epoch15', 'fullOnMouth_epoch80']
    # name_list = ['foreheadFoveated_baseModel', 'foreheadFoveated_50_baseModel']
    # name_list = ['fullOnForehead12_epoch15', 'fullOnForehead12_epoch80', 'fullOnForehead15_epoch15', 'fullOnForehead15_epoch80']

    for model_name in name_list:
        print(f'Now doing {model_name}'.ljust(64, '-'))
        fm_dir = './fm32/' + model_name
        res_dir = './result_neuronSelection/' + model_name
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        layer_list = [
            'layer1', 'layer2', 'layer3', 'layer4'
        ]
        print('layer_list:', layer_list)
        sample_num = 10
        class_num = 50
        alpha = 0.01

        for layer in layer_list:
            print(f'Loading feature matrix of {layer}'.ljust(64, '-'))
            FM_path = os.path.join(fm_dir, 'fullFM_' + layer + '.pkl')
            with open(FM_path, 'rb') as f:
                full_matrix = pkl.load(f)
            print('fullMatrix shape:', full_matrix.shape)
            print(f'Doing ANOVA on fullMatrix of {layer}'.ljust(64, '-'))
            sig_neuron_ind, non_sig_neuron_ind = oneWay_ANOVA(full_matrix, sample_num, class_num, alpha)

            SNI_path = os.path.join(res_dir, layer + '_sig_neuron_ind.pkl')
            with open(SNI_path, 'wb') as f:
                pkl.dump(sig_neuron_ind, f)
            print('SNI pkl saved!')
            sig_matrix = full_matrix[:, sig_neuron_ind]
            sig_matrix_path = os.path.join(res_dir, layer + '_sig_matrix.pkl')
            with open(sig_matrix_path, 'wb') as f:
                pkl.dump(sig_matrix, f)
            print('sig_matrix saved!', sig_matrix.shape)

            nonSNI_path = os.path.join(res_dir, layer + '_non_sig_neuron_ind.pkl')
            with open(nonSNI_path, 'wb') as f:
                pkl.dump(non_sig_neuron_ind, f)
            print('nonSNI pkl saved!')
            nonSig_matrix = full_matrix[:, non_sig_neuron_ind]
            nonSig_matrix_path = os.path.join(res_dir, layer + '_nonSig_matrix.pkl')
            with open(nonSig_matrix_path, 'wb') as f:
                pkl.dump(nonSig_matrix, f)
            print('nonSig_matrix saved!', nonSig_matrix.shape, '\n')


