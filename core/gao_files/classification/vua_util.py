"""
File for VUA CLS Model
Keeps track of sentence ids
"""
import csv
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


# christie adapted from sequence/util.write_predictions
# def write_predictions(raw_dataset, evaluation_dataloader, model, using_GPU, rawdata_filename):
#     """
#     Evaluate the model on the given evaluation_dataloader
#
#     :param raw_dataset
#     :param evaluation_dataloader:
#     :param model:
#     :param using_GPU: a boolean
#     :return: a list of
#     """
#     # Set model to eval mode, which turns off dropout.
#     model.eval()
#
#     predictions = {}
#     for (example_text, example_lengths, labels, ids) in evaluation_dataloader:
#         eval_text = Variable(example_text, volatile=True)
#         eval_lengths = Variable(example_lengths, volatile=True)
#         eval_labels = Variable(labels, volatile=True)
#         if using_GPU:
#             eval_text = eval_text.cuda()
#             eval_lengths = eval_lengths.cuda()
#             eval_labels = eval_labels.cuda()
#
#         # predicted shape: (batch_size, seq_len, 2)
#         predicted = model(eval_text, eval_lengths)
#         # get 0 or 1 predictions
#         # predicted_labels: (batch_size, seq_len)
#         _, predicted_labels = torch.max(predicted.data, 2)
#         predictions[]
#
#     # Set the model back to train mode, which activates dropout again.
#     model.train()
#     assert (len(predictions) == len(raw_dataset))
#
#     # read original data
#     data = []
#     with open(rawdata_filename, encoding='latin-1') as f:
#         lines = csv.reader(f)
#         for line in lines:
#             data.append(line)
#
#     # append predictions to the original data
#     data[0].append('prediction')
#     for i in range(len(predictions)):
#         data[i + 1].append(predictions[i])
#     return data


class VUATextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, embedded_text, labels, ids, max_sequence_length=100):
        """

        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo + suffix
        self.embedded_text = embedded_text
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length
        # concatenated text and sentence id for keeping track of genre level metrics
        self.sentence_ids = ids

    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.

        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        example_id = self.sentence_ids[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label, example_id

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.

        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []
        batch_ids = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]

        # Iterate over each example in the batch
        for text, length, label, id in batch:
            # Unpack the example (returned from __getitem__)

            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(label)
            batch_ids.append(id)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels),
                torch.LongTensor(batch_ids))
