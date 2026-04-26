dataset_size = 32
batch_size = 16

for start in range(0, dataset_size, batch_size):
    end = start + batch_size
    print(start, end)
    # batch_idx = indices[start:end]
    # X_batch = X_all[batch_idx]
    # y_batch = y_all[batch_idx]

    # preds = self.model(X_batch)
    # loss = self.loss_fn(preds, y_batch)