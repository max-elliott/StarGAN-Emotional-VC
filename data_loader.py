class variable_length_dataset(data_utils.Dataset):
    def __init__(self, data, targets):
        super(variable_length_dataset, self).__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.targets[index])

    def collate_length_order(self, batch):
        '''
        batch: Batch elements are tuples ((Tensor)sequence, target)

        Sorts batch by sequence length

        returns:
            (LongTensor) sequence_padded: seqs in length order, padded to max_len
            (LongTensor) lengths: lengths of seqs in sequence_padded
            (LongTensor) labels: corresponding targets, in correct order
        '''

        # assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # Don't forget to grab the labels of the *sorted* batch
        targets = torch.LongTensor([x[1] for x in sorted_batch])

        return [sequences_padded, lengths], targets

    def collate_length_order_dims(self, batch):
        '''
        batch: Batch elements are tuples ((Tensor)sequence, target)

        Sorts batch by sequence length

        returns:
            (LongTensor) sequence_padded: seqs in length order, padded to max_len
            (LongTensor) lengths: lengths of seqs in sequence_padded
            (LongTensor) labels: corresponding targets, in correct order
        '''

        # assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # Don't forget to grab the labels of the *sorted* batch
        targets = torch.stack([x[1] for x in sorted_batch])

        return [sequences_padded, lengths], targets

def make_variable_dataloader(x, y, batch_size = 64, train_test_split = 0.9):
    split_index = int(len(x)*0.9)

    train_dataset = variable_length_dataset(x[:split_index], y[:split_index])
    test_dataset = variable_length_dataset(x[split_index:], y[split_index:])
    train_loader = data_utils.DataLoader(train_dataset, batch_size = batch_size,
                                         collate_fn = train_dataset.collate_length_order,
                                         num_workers = 4, shuffle = True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size = batch_size,
                                         collate_fn = test_dataset.collate_length_order,
                                         num_workers = 4, shuffle = True)

    return train_loader, test_loader

def make_variable_dataloader_dims(x, y, batch_size = 64, train_test_split = 0.9):
    split_index = int(len(x)*0.9)

    train_dataset = variable_length_dataset(x[:split_index], y[:split_index])
    test_dataset = variable_length_dataset(x[split_index:], y[split_index:])
    train_loader = data_utils.DataLoader(train_dataset, batch_size = batch_size,
                                         collate_fn = train_dataset.collate_length_order_dims,
                                         num_workers = 4, shuffle = True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size = batch_size,
                                         collate_fn = test_dataset.collate_length_order_dims,
                                         num_workers = 4, shuffle = True)

    return train_loader, test_loader

def make_dataloader(x, y, batch_size = 64, split = 0.9):

    split_index = int(len(x)*split)

    train_dataset = data_utils.TensorDataset(x[:split_index], y[:split_index])
    test_dataset = data_utils.TensorDataset(x[split_index:], y[split_index:])
    train_loader = data_utils.DataLoader(train_dataset, batch_size = batch_size,
                                         num_workers = 4, shuffle = True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size = batch_size,
                                         num_workers = 4, shuffle = True)

    return train_loader, test_loader
