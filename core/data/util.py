'''
Utility Functions
'''

def prep_inx_word(list_vocab, pad_index = 0, unk_index = 1):
    '''
    Function to prepare word index dictionaries.
    Input: list of vocabularies
    Return: word to index mapping(dct), index to word mapping(dct)
    '''
    word_idx = {"<PAD>": pad_index, 
                "<UNK>": unk_index}
    for word in list_vocab:
        idx = len(word_indx)
        word_idx[word] = idx
        idx_word[idx] = word
    return word_idx, idx_word

def word_embedding():
    pass



        


