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

#%%

train_agent=> 'CA', 'SA', None; default None
deploy_agent=> 'CA', 'SA', 'both', None; default 'both'

if

def resolve_agent_modes(train_agent, deploy_agent):
    """
    Returns:
        splitting_agent_mode: 'train', 'predict', or None
        collecting_agent_mode: 'train', 'predict', or None
    """

    modes = {
        'SA': None,
        'CA': None,
    }

    # --- Training ---
    if train_agent in modes:
        modes[train_agent] = 'train'

    # --- Deployment ---
    if deploy_agent == 'both':
        for k in modes:
            if modes[k] is None:
                modes[k] = 'predict'
    elif deploy_agent in modes:
        if modes[deploy_agent] is None:
            modes[deploy_agent] = 'predict'

    return modes['SA'], modes['CA']