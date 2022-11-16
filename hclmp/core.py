'''
Pytorch implementation of the paper "Materials representation and transfer
learning for multi-property prediction"

Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

import textwrap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from hclmp.model import Hclmp, compute_loss
from hclmp.graph_encoder import CompositionData, collate_batch


class HclmpDataset(Dataset):
    """
    H-CLMP originally accepted the `MyDataset` class, which assumes that you
    have a file containing the indices of the dataset you want. This class
    assumes that you will not use indices, and instead take the entire dataset.
    """

    def __init__(self, Data):
        self.all_element_name = Data.pop('all_element_name')
        self.data = Data

    def get_all_target(self):
        indices = sorted(self.data.keys())
        targets = [np.expand_dims(self.data[idx]['fom'], axis=0)
                   for idx in indices]
        targets_np = np.concatenate(targets, axis=0)
        return torch.as_tensor(targets_np)

    def __getitem__(self, idx):
        item = self.data[idx]
        y = item['fom']
        ele_comp = item['composition']
        gen_feat = item['gen_dos_fea']
        return (torch.as_tensor(ele_comp), torch.as_tensor(y), torch.as_tensor(gen_feat))


class MyDataset(Dataset):

    def __init__(self, Data, idx_path):
        self.data = Data
        self.idx = np.load(idx_path)

    def __len__(self):
        return len(self.idx)

    def get_all_target(self):
        all_target = []
        for i in self.idx:
            item = self.data[i]
            all_target.append(np.expand_dims(item['fom'], axis=0))
        all_target = np.concatenate(all_target, axis=0)
        return torch.as_tensor(all_target)

    def __getitem__(self, idx):
        item = self.data[self.idx[idx]]
        y = item['fom']
        ele_comp = item['composition']
        gen_feat = item['gen_dos_fea']

        return (torch.as_tensor(ele_comp), torch.as_tensor(y), torch.as_tensor(gen_feat))


class Scaler():
    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def scale(self, data):
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def train_new(model: Hclmp,
              dataset: HclmpDataset,
              data_loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler._LRScheduler = None,
              scaler: Scaler = None,
              num_epochs: int = 100,
              cp_name: str = 'hclmp_chkpt.pth.tar',
              best_cp_name: str = 'hclmp_chkpt_best.pth.tar',
              device: torch.device = None):
    """
    Updated version of the `train` function. Change notes:
        - No validation results are reported during training. This is done to
        reduce information leakage from the validation set.
        - Update logging. This includes dynamically changing loss figures when
        operating in Jupyter notebooks.
        - Added arguments

    Args:
        model (Hclmp): The H-CLMP model instance to train

        dataset (HclmpDataset): The H-CLMP dataset to train on

        data_loader (DataLoader): The data loader to use to read minibatches
        during training

        optimizer (torch.optim.Optimizer): Optimizer to use when tuning the
        weights and biases of the network

        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate
        scheduler to use to taper the learning rate throughout training

        scaler (Scaler): The scaler object used to normalize the output values

        num_epochs (int): The number of epochs to use when training

        cp_name (str): The name of the checkpoint file to save after each epoch

        best_cp_name (str): The name of the checkpoint file to save for the
        best iteration

        device (torch.device): The torch device to use to train (e.g., GPU or
        CPU)
    """

    # Auto-detection of the device to use (if necessary)
    if device is None:
        cuda_available = bool(torch.cuda.device_count())
        device = torch.device('cuda') if cuda_available else torch.device('cpu')

    # If we're in a notebook, we'll use a dynamic plot of the loss.
    # But if we're in a normal python thread, then we'll just use tqdm and
    # print statements.
    notebook = is_notebook()
    if notebook:
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        from IPython.display import display, clear_output
        from tqdm.notebook import tqdm as tqdm_
    else:
        tqdm_ = tqdm

    # Initialize training
    model.train()
    n_data = len(dataset)
    losses = {'total': [], 'nll': [], 'nll x': [], 'kl': []}
    best_loss = np.inf
    minibatches = []

    epoch_iter = tqdm_(range(num_epochs), total=num_epochs, desc='Epochs', leave=False)
    for epoch in epoch_iter:

        # Divide training of each epoch into batches
        minibatch_iter = tqdm_(data_loader, desc='Minibatch', leave=False)
        for input_, y, gen_feat, _, _ in minibatch_iter:

            input_ = (tensor.to(device) for tensor in input_)
            y = y.to(device)
            y = scaler.scale(y).float() if scaler else y.float()
            gen_feat = gen_feat.to(device).float()

            out = model(y, gen_feat, *input_)
            total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y, out)

            # Train on this one batch
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            losses['total'].append(float(total_loss))
            losses['nll'].append(float(nll_loss_e))
            losses['nll x'].append(float(nll_loss_x))
            losses['kl'].append(float(kl_loss))
            minibatches.append(len(losses['total']))

            # Display the loss via tqdm
            if not notebook:
                loss_summary = {loss_name: np.average(loss[-n_data:])
                                for loss_name, loss in losses.items()}
                minibatch_iter.set_postfix(loss_summary)

            # Display the loss via dynamically updating plot
            else:
                ax.clear()
                ax.set_xlim([0, num_epochs * len(data_loader)])
                min_loss = float(min([0] + [min(loss) for loss in losses.values()]))
                max_loss = float(max(max(loss) for loss in losses.values()))
                ax.set_ylim([min_loss, max_loss])
                ax.set_xlabel('Minibatch #')
                ax.set_ylabel('Loss')
                for loss_name, loss in losses.items():
                    sns.lineplot(x=minibatches, y=loss, ax=ax, label=loss_name)
                display(fig)
                clear_output(wait=True)

        # Update the displayed learning rate
        lr = optimizer.param_groups[0]['lr']
        epoch_iter.set_postfix(lr=lr)

        # Save the checkpoint
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch}
        if scaler:
            checkpoint['scaler_state'] = scaler.state_dict()
        torch.save(checkpoint, cp_name)

        # Save the best model
        mean_loss = np.average(losses['nll x'][-n_data:])
        if best_loss > mean_loss:
            best_loss = mean_loss
            torch.save(checkpoint, best_cp_name)


# train and save models
def train(args):
    print(f'\nTraining on sys {args.sys_name} \n')
    Data = torch.load(args.data_path)
    dataset = MyDataset(Data, args.train_path)
    all_target = dataset.get_all_target().to(args.device)
    MyScaler = Scaler()
    MyScaler.fit(all_target)
    # MyScaler.mean = MyScaler.mean.to(args.device)
    # MyScaler.std = MyScaler.std.to(args.device)

    # composition_dataset = CompositionData(args.data_path,
    #                                       "data/embeddings/megnet16-embedding.json",
    #                                       "regression")
    # composition_dataset = CompositionData(args.data_path,
    #                                       "data/embeddings/cgcnn-embedding.json",
    #                                       "regression")
    composition_dataset = CompositionData(args.data_path,
                                          "data/embeddings/matscholar-embedding.json",
                                          "regression")
    train_idx = np.load(args.train_path)
    val_idx = np.load(args.val_path)
    train_dataset = torch.utils.data.Subset(composition_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(composition_dataset, val_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_batch)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_batch)

    elem_emb_len = composition_dataset.elem_emb_len

    model = Hclmp(args.feat_dim, args.label_dim, args.transfer_type,
                  args.gen_feat_dim, elem_emb_len, args.device).to(args.device)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-2)
    one_epoch_iter = np.ceil(len(train_dataset) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, one_epoch_iter * (args.epochs / args.decay_times),
        args.decay_ratios
    )

    best_loss = 1e+10
    for epoch in range(args.epochs):
        # train
        model.train()
        total_loss_smooth = 0
        nll_loss_e_smooth = 0
        nll_loss_x_smooth = 0
        kl_loss_smooth = 0
        count = 0
        pred_e = []
        pred_x = []
        label = []
        iterator = tqdm(train_loader,
                        mininterval=0.5,
                        desc='(Training)',
                        position=0,
                        leave=True,
                        ascii=True)
        for input_, y, gen_feat, _, _ in iterator:
            input_ = (tensor.to(args.device) for tensor in input_)
            y = y.to(args.device)
            y_norm = MyScaler.scale(y)
            gen_feat = gen_feat.to(args.device)

            out = model(y_norm, gen_feat, *input_)
            total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            pred_e.append(out['label_out'])
            pred_x.append(out['feat_out'])
            label.append(y_norm)
            total_loss_smooth += total_loss
            nll_loss_e_smooth += nll_loss_e
            nll_loss_x_smooth += nll_loss_x
            kl_loss_smooth += kl_loss
            count += 1

        pred_e = torch.cat(pred_e, dim=0)
        pred_x = torch.cat(pred_x, dim=0)
        label = torch.cat(label, dim=0)
        total_loss_smooth = total_loss_smooth / count
        nll_loss_e_smooth = nll_loss_e_smooth / count
        nll_loss_x_smooth = nll_loss_x_smooth / count
        kl_loss_smooth = kl_loss_smooth / count
        lr = optimizer.param_groups[0]['lr']

        print("\n********** TRAINING STATISTIC ***********")
        msg = textwrap.dedent(
            f"""
            epoch = {epoch}\t
            lr = {lr}\t
            total loss = {total_loss_smooth}\t
            nll loss = {nll_loss_e_smooth}\t
            nll loss x = {nll_loss_x_smooth}\t
            kl loss = {kl_loss_smooth}\t
            """
        )
        print(msg)
        print("\n*****************************************")

        # validate
        model.eval()
        total_loss_smooth = 0
        nll_loss_e_smooth = 0
        nll_loss_x_smooth = 0
        kl_loss_smooth = 0
        count = 0
        pred_e = []
        pred_x = []
        label = []

        iterator = tqdm(val_loader,
                        mininterval=0.5,
                        desc='(Validating)',
                        position=0, leave=True,
                        ascii=True)
        for input_, y, gen_feat, _, _ in iterator:
            input_ = (tensor.to(args.device) for tensor in input_)
            y = y.to(args.device)
            y_norm = MyScaler.scale(y)
            gen_feat = gen_feat.to(args.device)

            with torch.no_grad():
                out = model(y_norm, gen_feat, *input_)
                total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

            pred_e.append(out['label_out'])
            pred_x.append(out['feat_out'])
            label.append(y_norm)
            total_loss_smooth += total_loss
            nll_loss_e_smooth += nll_loss_e
            nll_loss_x_smooth += nll_loss_x
            kl_loss_smooth += kl_loss
            count += 1

        pred_e = torch.cat(pred_e, dim=0)
        pred_x = torch.cat(pred_x, dim=0)
        label = torch.cat(label, dim=0)
        total_loss_smooth = total_loss_smooth / count
        nll_loss_e_smooth = nll_loss_e_smooth / count
        nll_loss_x_smooth = nll_loss_x_smooth / count
        kl_loss_smooth = kl_loss_smooth / count

        print("\n********** VALIDATING STATISTIC ***********")
        msg = textwrap.dedent(
            f"""
            epoch = {epoch}\t
            total loss = {total_loss_smooth}\t
            nll loss = {nll_loss_e_smooth}\t
            nll loss x = {nll_loss_x_smooth}\t
            kl loss = {kl_loss_smooth}\t
            """
        )
        print(msg)
        print("\n*****************************************")

        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['scaler_state'] = MyScaler.state_dict()
        checkpoint['args'] = args
        checkpoint['epoch'] = epoch
        cp_file = args.save_path + 'checkpoint_' + args.transfer_type + '.pth.tar'
        torch.save(checkpoint, cp_file)
        if best_loss > nll_loss_x_smooth:
            best_loss = nll_loss_x_smooth
            best_file = args.save_path + 'best_' + args.transfer_type + '.pth.tar'
            torch.save(checkpoint, best_file)


# load and test models, as well as saving results
def test(args):
    print(f'\nTesting on sys {args.sys_name}... \n')
    result_path = f"results/{args.sys_name}/"
    if args.transfer_type == 'gen_feat':
        label_b, pred_b, mean_b, std_b, mae_b = run_test(args, 'best')
    else:
        label_b, pred_b, mean_b, std_b, mae_b = run_test(args, 'best')

    print("\n********** TESTING STATISTIC ***********")
    print("MAE =%.6f" % (mae_b))
    print("\n*****************************************")

    np.save(result_path + 'pred_' + args.transfer_type + '.npy', pred_b)
    np.save(result_path + 'label_' + args.transfer_type + '.npy', label_b)
    np.save(result_path + 'mean_' + args.transfer_type + '.npy', mean_b)
    np.save(result_path + 'std_' + args.transfer_type + '.npy', std_b)


def run_test(args, mode):
    path = args.save_path + mode + '_' + args.transfer_type + '.pth.tar'
    checkpoint = torch.load(path, map_location=args.device)
    args_save = checkpoint['args']
    args.feat_dim = args_save.feat_dim
    args.gen_feat_dim = args_save.gen_feat_dim
    args.label_dim = args_save.label_dim

    MyScaler = Scaler()
    MyScaler.load_state_dict(checkpoint['scaler_state'])

    composition_dataset = CompositionData(args.data_path,
                                          "data/embeddings/matscholar-embedding.json",
                                          "regression")
    test_idx = np.load(args.test_path)
    test_dataset = torch.utils.data.Subset(composition_dataset, test_idx)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=collate_batch)

    elem_emb_len = composition_dataset.elem_emb_len

    model = Hclmp(args.feat_dim, args.label_dim, args.transfer_type,
                  args.gen_feat_dim, elem_emb_len, args.device).to(args.device)
    model.load_state_dict(checkpoint['model'])

    # test
    model.eval()
    total_loss_smooth = 0
    nll_loss_e_smooth = 0
    nll_loss_x_smooth = 0
    kl_loss_smooth = 0
    count = 0
    pred_e = []
    pred_x = []
    label = []

    iterator = tqdm(test_loader,
                    mininterval=0.5,
                    desc='(Testing)',
                    position=0,
                    leave=True,
                    ascii=True)
    for input_, y, gen_feat, _, _ in iterator:
        input_ = (tensor.to(args.device) for tensor in input_)
        y = y.to(args.device)
        y_norm = MyScaler.scale(y)
        gen_feat = gen_feat.to(args.device)

        with torch.no_grad():
            out = model(y_norm, gen_feat, *input_)
            total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

        pred_e.append(out['label_out'])
        pred_x.append(out['feat_out'])
        label.append(y_norm)
        total_loss_smooth += total_loss
        nll_loss_e_smooth += nll_loss_e
        nll_loss_x_smooth += nll_loss_x
        kl_loss_smooth += kl_loss
        count += 1

    pred_e = torch.cat(pred_e, dim=0)
    pred_x = torch.cat(pred_x, dim=0)
    label = torch.cat(label, dim=0)
    total_loss_smooth = total_loss_smooth / count
    nll_loss_e_smooth = nll_loss_e_smooth / count
    nll_loss_x_smooth = nll_loss_x_smooth / count
    kl_loss_smooth = kl_loss_smooth / count

    nll_loss_e = torch.mean(torch.abs(pred_e-label))
    nll_loss_x = torch.mean(torch.abs(pred_x-label))

    label = label.data.cpu().numpy()
    pred = pred_x.data.cpu().numpy()
    mean = MyScaler.mean.data.cpu().numpy()
    std = MyScaler.std.data.cpu().numpy()

    return label, pred, mean, std, nll_loss_x


def is_notebook() -> bool:
    """
    Detects whether this Python instance is in a notebook

    :return bool: Whether this python instance is in a notebook
    """
    # Credict goes to Gustavo Bezerra
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    try:
        shell = get_ipython().__class__.__name__

        # Jupyter notebook or qtconsole
        if shell == 'ZMQInteractiveShell':
            return True

        # Terminal running IPython
        elif shell == 'TerminalInteractiveShell':
            return False

        # Other type (?)
        else:
            return False

    # Probably standard Python interpreter
    except NameError:
        return False
