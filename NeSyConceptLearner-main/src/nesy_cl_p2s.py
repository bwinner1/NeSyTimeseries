import matplotlib
matplotlib.use("Agg")
import os
import torch
import torch.nn as nn
import numpy as np
import glob
from sklearn import metrics
from tqdm import tqdm

import data_clevr_hans as data
from p2s import P2S_Dataset

import model
import utils as utils
from rtpt import RTPT
from args import get_args

torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from datasets import DatasetDict
from itertools import product

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["aeon.AEON_DEPRECATION_WARNING"] = "False"
torch.set_num_threads(6)

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
def get_confusion_from_ckpt(net, test_loader, criterion, args, datasplit, writer=None):

    true, pred, true_wrong, pred_wrong = run_test_final(net, test_loader, criterion, writer, args, datasplit)
    precision, recall, accuracy, f1_score = utils.performance_matrix(true, pred)

    # Generate Confusion Matrix
    if writer is not None:
        # TODO: Delete the following overwrite. Instead overwrite the function later 
        args.classes = np.arange(2)
#        args.classes = np.arange(args.n_classes)

        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_normalize_{}.pdf'.format(
                                  datasplit))
                              )
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_{}.pdf'.format(datasplit)))
    else:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_normalize_{}.pdf'.format(datasplit)))
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_{}.pdf'.format(datasplit)))
    return accuracy

# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------
def run_test_final(net, loader, criterion, writer, args, datasplit):
    net.eval()

    running_corrects = 0
    running_loss=0
    pred_wrong = []
    true_wrong = []
    preds_all = []
    labels_all = []
    with torch.no_grad():

        for i, (concepts, labels, _) in enumerate(tqdm(loader)):
            # input is either a set or an image
            # imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
            # img_class_ids = img_class_ids.long()

            concepts = concepts.to(args.device)
            labels = labels.to(args.device)
            #labels = labels.float()
        
            # Network usage

            output_cls, output_attr, preds = apply_net(concepts, net, args)

            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == labels)
            loss = criterion(output_cls, labels)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = labels.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            target = np.reshape(target, (len(preds), 1))

            for i in range(len(preds)):
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])

        bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong


def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0):
    if train:
        #net.img2state_net.eval()
        net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, (concepts, labels, _) in enumerate(loader, start=epoch * iters_per_epoch):

        # Move both tensors to correct device
        concepts = concepts.to(args.device)
        labels = labels.to(args.device)

        # Network usage
        output_cls, output_attr, preds = apply_net(concepts, net, args)

        loss = criterion(output_cls, labels)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(labels.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch):
            utils.write_expls(net, loader, f"Expl/{split}", epoch, writer, args)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )
    return running_loss / len(loader)


def train(args):
    print("Running train method...")

    # Create RTPT object
    rtpt = RTPT(name_initials='BI', experiment_name=f"P2S NeSyCL for TS",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    if args.dataset == "p2s":

        ### TODO: Change back to 'Normal'
        train_dataset = load_dataset('AIML-TUDA/P2S', 'Normal', download_mode='reuse_dataset_if_exists')

        # Extracting the time series data from the train and test dataset
        ts_train = np.array(train_dataset['train']['dowel_deep_drawing_ow'])
        ts_train_speeds = np.array(train_dataset['train']['speed'])
        
        # Number of samples that goes into the train dataset; the rest goes into the validation dataset
        training_samples = int(ts_train.shape[0] * 0.8)
        ts_val = ts_train[training_samples :]
        ts_train = ts_train[: training_samples]

        ts_test = np.array(train_dataset['test']['dowel_deep_drawing_ow'])

        # TODO: Add Shuffling to train and val datasets
        labels_train = torch.tensor(train_dataset['train']['label'][:training_samples])
        labels_val = torch.tensor(train_dataset['train']['label'][training_samples:])
        labels_test = torch.tensor(train_dataset['test']['label'])

    else:
        print("Wrong dataset specifier")
        exit()

    if args.concept == "sax":
        sax = model.SAXTransformer(n_segments=args.n_segments, alphabet_size=args.alphabet_size)
        """ 
        # Adding a third dimension, in the middle of the shape, as needed for SAX
        concepts_train, _, _ = sax.transform(ts_train.reshape(ts_train.shape[0], 1, ts_train.shape[1]))
        concepts_val, _, _ = sax.transform(ts_val.reshape(ts_val.shape[0], 1, ts_val.shape[1]))
        concepts_test, _, _ = sax.transform(ts_test.reshape(ts_test.shape[0], 1, ts_test.shape[1]))

        # Remove middle dimension from (samples, 1, concepts)   
        concepts_train = torch.squeeze(torch.tensor(concepts_train))
        concepts_val = torch.squeeze(torch.tensor(concepts_val))
        concepts_test = torch.squeeze(torch.tensor(concepts_test))
 """
        concepts_train = sax.transform(ts_train)
        concepts_val = sax.transform(ts_val)
        concepts_test = sax.transform(ts_test)


    ### TODO: Add other cases
    elif args.concept == "tsfresh":
        if args.load_ts:
            # Load previous .pt files
            concepts_train = torch.load(f'pretrain/tsfresh_{args.ts_setting}_train.pt')
            concepts_val = torch.load(f'pretrain/tsfresh_{args.ts_setting}_val.pt')
            concepts_test = torch.load(f'pretrain/tsfresh_{args.ts_setting}_test.pt')
            pass
        else:
            # Use tsfresh and save extracted features into a .pt file
            concepts_train, filtered_columns = model.tsfreshTransformer.transform(ts_train, labels_train, setting=args.ts_setting)
            concepts_val, _ = model.tsfreshTransformer.transform(ts_val, labels_val, filtered_columns, setting=args.ts_setting)
            concepts_test, _ = model.tsfreshTransformer.transform(ts_test, labels_test, filtered_columns, setting=args.ts_setting)

            torch.save(concepts_train, f'pretrain/tsfresh_{args.ts_setting}_train.pt')
            torch.save(concepts_val, f'pretrain/tsfresh_{args.ts_setting}_val.pt')
            torch.save(concepts_test, f'pretrain/tsfresh_{args.ts_setting}_test.pt')

    #elif args.concept == "vq-vae":
    else:
        print("Wrong concept specifier")
        exit()

    # Create Torch Tensors for TensorDataset
    ts_train = torch.tensor(ts_train)
    ts_test = torch.tensor(ts_test)
    ts_val = torch.tensor(ts_val)
    
    print("\nTrain Concepts, Labels and Time series:")
    print(concepts_train.size())
    print(labels_train.size())
    print(ts_train.size())

    print("\nVal Concepts, Labels and Time series:")
    print(concepts_val.size())
    print(labels_val.size())
    print(ts_val.size())

    print("\nTest Concepts, Labels and Time series:")
    print(concepts_test.size())
    print(labels_test.size())
    print(ts_test.size())

    #p2s_dataset = P2S_Dataset(concepts, labels)

    # Creating a Dataset using Tensors
    train_dataset = TensorDataset(concepts_train, labels_train, ts_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_dataset = TensorDataset(concepts_val, labels_val, ts_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    test_dataset = TensorDataset(concepts_test, labels_test, ts_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # In general, the SetTransformer requires the following shape:
    # (batch_size * num_elements * feature_dim)
    # n_attr being equal the feature_dim
    if args.concept == "sax":
        net = model.NeSyConceptLearner(n_attr=args.alphabet_size, device=args.device)
    elif args.concept == "tsfresh":
        # n_attr is set to the 
        net = model.NeSyConceptLearner(n_attr=concepts_train.size(2), device=args.device)
    # elif ...
    else:
        pass


    net = net.to(args.device)
    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # tensorboard writer
    writer = utils.create_writer(args)

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        scheduler.step()

        # TODO: Set value back to plot=True

        #plot = False
        #if(epoch == args.epochs - 1):
        #    plot = True
        plot = False

        val_loss = run(net, val_loader, optimizer, criterion, split='val', args=args, writer=writer,
                        train=False, plot=plot, epoch=epoch)                       
        _ = run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
                train=False, plot=False, epoch=epoch)

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": args,
        }
        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
            torch.save(results, os.path.join(writer.log_dir, "model_epoch{}_bestvalloss_{:.4f}.pth".format(epoch,
                                                                                                           val_loss)))
            cur_best_val_loss = val_loss

        # Update the RTPT (subtitle is optional)
        rtpt.step()

    # load best model for final evaluation
    
    if args.concept == "sax":
        net = model.NeSyConceptLearner(n_attr=args.alphabet_size, device=args.device)
    elif args.concept == "tsfresh":
        # n_attr is set to the 
        net = model.NeSyConceptLearner(n_attr=concepts_train.size(2), device=args.device)
    # elif ...
    else:
        pass


    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation, printing test and val statistics::\n")

    acc_test = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    acc_val = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)


    # plot expls
    # TODO: Set values back to plot=True

    run(net, train_loader, optimizer, criterion, split='train_best', args=args,
        writer=writer, train=False, plot=False, epoch=0)
    run(net, val_loader, optimizer, criterion, split='val_best', args=args,
        writer=writer, train=False, plot=False, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test_best', args=args,
        writer=writer, train=False, plot=False, epoch=0)

    writer.close()

    return acc_test, acc_val


def test(args):

    print(f"\n\n{args.name} seed {args.seed}\n")
    if args.dataset == "clevr-hans-state":
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_classes = dataset_val.n_classes
    args.class_weights = torch.ones(args.n_classes)/args.n_classes
    args.classes = np.arange(args.n_classes)
    args.category_ids = dataset_val.category_ids

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    net = model.NeSyConceptLearner(n_classes=args.n_classes, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    acc = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best', writer=None)
    print(f"\nVal. accuracy: {(100*acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best', writer=None)
    print(f"\nTest accuracy: {(100*acc):.2f}")


def plot(args):

    print(f"\n\n{args.name} seed {args.seed}\n")

    # no positional info per object
    if args.dataset == "clevr-hans-state":
        dataset_val = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", lexi=True, conf_vers=args.conf_version
        )
        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", lexi=True, conf_vers=args.conf_version
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_classes = dataset_val.n_classes
    args.class_weights = torch.ones(args.n_classes)/args.n_classes
    args.classes = np.arange(args.n_classes)
    args.category_ids = dataset_val.category_ids

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_classes, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                             n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                             category_ids=args.category_ids, device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    save_dir = args.fp_ckpt.split('model_epoch')[0]+'figures/'
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass

    # change plotting function in utils in order to visualize explanations
    assert args.conf_version == 'CLEVR-Hans3'
    utils.save_expls(net, test_loader, "test", save_path=save_dir)

def apply_net(input, net, args):
    """
    Apply given network. Depending on the chosen symbolizer, 
    the input will be prepared accordingly. F.e. the input for SAX will be one-hot encoded
    """

    # Preparing the input
    if(args.concept == "sax"):
        input = nn.functional.one_hot(input, num_classes=args.alphabet_size)


    print(f"old apply_net input and net device: {input.device}, {net.device}")
    input = input.to(args.device)
    net = net.to(args.device)
    print(f"new apply_net input and net device: {input.device}, {net.device}")
    # Applying the SetTransformer
    output_cls, output_attr = net(input)
    
    # preds = (output_cls > 0).float()
    _, preds = torch.max(output_cls, 1)
    
    return output_cls, output_attr, preds

def main():
    args = get_args()
    if args.mode == 'train':
        torch.cuda.empty_cache()
        # args.set_heads = 4
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'plot':
        plot(args)
    elif args.mode == 'gridsearch':

        # Gridsearch 3
        segments = (128, 256, 512, 1024)
        alphabet_sizes = (4, 10, 16, 32)
        set_heads = (4,)

        """ 
        # Gridsearch 2
        segments = (32, 64, 128, 256)
        alphabet_sizes = (4, 6, 8, 10)
        set_heads = (1, 2, 4, 8, 16)
         """

        test_accuracies = []
        val_accuracies = []

        for (s, a, s_h) in product(segments, alphabet_sizes, set_heads):
            args.n_segments = s
            args.alphabet_size = a
            args.set_heads = s_h

            acc_test, acc_val = train(args)
            test_accuracies.append(acc_test)
            val_accuracies.append(acc_val)

        print("n_segments,alphabet_size,set_heads,test_acc,val_acc")
        for (i, (s, a, s_h)) in enumerate(product(segments, alphabet_sizes, set_heads)):
            print(f"{s},{a},{s_h},{100 * test_accuracies[i]:.3f},{100 * val_accuracies[i]:.3f}")
            #print(f"n_segments: {s}, alphabet_size: {a}, set_heads: {s_h}, test_acc: {100 * test_accuracies[i]:.3f}; val_acc: {100 * val_accuracies[i]:.3f}")
            #print('n_segments: {}, alphabet_size: {}, test_accuracy: {:.3f}, val_accuracy: {:.3f}'
             #     .format(s, a, test_accuracies[i]*100, val_accuracies[i]*100))


if __name__ == "__main__":
    main()
