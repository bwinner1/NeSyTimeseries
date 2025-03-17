import matplotlib
matplotlib.use("Agg")
import sys
import os
import gc
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

from torch.profiler import profile, record_function, ProfilerActivity

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


    # Dictionary to save importance statistics over all samples
    # global_importance = {}

    if args.explain_all:

        args.best_features = [{}, {}]
        args.worst_features = [{}, {}]
        """ 
        if args.concept == "sax":
            pass
        elif args.concept == "tsfresh":
            # Per sample, count the top 10 features contributing positively 
            # and negatively to a class prediction, add them to the dictionary.
            # The feature name is the key, while the value is the count
            args.best_features = [{}, {}]
            args.worst_features = [{}, {}]
 """
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


    # If global explainability is desired, save best_features and worst_features into a csv
    if (args.explain_all and split[-4:]=="best"):

        utils.write_global_expl(args.concept, split, "best", args.best_features, 0)
        utils.write_global_expl(args.concept, split, "best", args.best_features, 1)
        utils.write_global_expl(args.concept, split, "worst", args.worst_features, 0)
        utils.write_global_expl(args.concept, split, "worst", args.worst_features, 1)

        """
        filename_best = f"xai/tsfresh/{split}/best_features_pred0_{utils.get_current_time()}.csv"
        with open(filename_best, "a") as file_best:
            file_best.write("feature_name;count\n")
            for f, c in args.best_features[0].items():
                file_best.write(f"{f};{c}\n")

        filename_worst = f"xai/tsfresh/{split}/worst_features_pred0_{utils.get_current_time()}.csv"
        with open(filename_worst, "a") as file_worst:
            file_worst.write("feature_name;count\n")
            for f, c in args.worst_features[0].items():
                file_worst.write(f"{f};{c}\n")    
        """

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
    rtpt = RTPT(name_initials='BI', experiment_name=f"NeSy-Cl for P2S",
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

        concepts_train = sax.transform(ts_train)
        concepts_val = sax.transform(ts_val)
        concepts_test = sax.transform(ts_test)

    elif args.concept == "tsfresh":        
        file = "pretrain"

        if args.filter_tsf:
            file += "/features_filtered"
        else:
            file += "/features_unfiltered"

        if args.normalize_tsf:
            file += "_normalized"
        
        if args.load_tsf:
            # Load previous .pt files
            concepts_train = torch.load(f'{file}/tsfresh_{args.ts_setting}_train.pt')
            concepts_val = torch.load(f'{file}/tsfresh_{args.ts_setting}_val.pt')
            concepts_test = torch.load(f'{file}/tsfresh_{args.ts_setting}_test.pt')
            column_labels = torch.load(f'{file}/tsfresh_{args.ts_setting}_column_labels.pt')

            # TODO: Delete the following, should be irrelevant after saving the scaler everything again
            scaler_path = f'{file}/tsfresh_{args.ts_setting}_scaler.pt'
            if os.path.exists(scaler_path):
                scaler = torch.load(scaler_path)
                print("Scaler loaded successfully!")
            else:
                scaler = model.tsfreshTransformer.fit_scaler(concepts_train.squeeze(1).to("cpu").numpy())
                print("Scaler computed manually")

        else:
            # Use tsfresh and save extracted features into a .pt file
            concepts_train, filtered_columns, scaler = model.tsfreshTransformer.transform(ts_train, 
                                labels_train, setting=args.ts_setting, filter=args.filter_tsf)
            concepts_val, _, _ = model.tsfreshTransformer.transform(ts_val, labels_val,
                                filtered_columns, scaler, setting=args.ts_setting, filter=args.filter_tsf)
            concepts_test, _, _ = model.tsfreshTransformer.transform(ts_test, labels_test,
                                filtered_columns, scaler, setting=args.ts_setting, filter=args.filter_tsf)
            column_labels = filtered_columns.tolist()


            torch.save(concepts_train, f'{file}/tsfresh_{args.ts_setting}_train.pt')
            torch.save(concepts_val, f'{file}/tsfresh_{args.ts_setting}_val.pt')
            torch.save(concepts_test, f'{file}/tsfresh_{args.ts_setting}_test.pt')
            torch.save(column_labels, f'{file}/tsfresh_{args.ts_setting}_column_labels.pt')
            
            torch.save(scaler, f'{file}/tsfresh_{args.ts_setting}_scaler.pt')

        args.scaler = scaler

        # args.column_labels = column_labels

        # Remove irrelevant first part "ts__" of column names
        args.column_labels = [s[4:] for s in column_labels]
        # Important for using in utils later:
        print(f"Number of column_labels: {len(column_labels)}")

    elif args.concept == "vqshape":

        vqshape = model.vqshapeTransformer()

        with torch.no_grad():
            concepts_train = vqshape.transform(ts_train)
            concepts_val = vqshape.transform(ts_val)
            concepts_test = vqshape.transform(ts_test)

        print("vqshape DONE")

    else:
        print("Given summarizer doesn't exist")
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

    # SAX input shape: (using one_hot_encoding, later in apply_net)
    # (batch_size, segments, alphabet_size)
    if args.concept == "sax":
        args.n_input_dim = args.alphabet_size

    # tsfresh input shape:
    # (batch_size, 1, feature_num)
    elif args.concept == "tsfresh":
        args.n_input_dim = concepts_train.size(2)

    # vqshape input shape:
    # (batchsize, codeblock_size, features)
    elif args.concept == "vqshape":
        print(f"concepts_train.size(2): {concepts_train.size(2)}")
        args.n_input_dim = concepts_train.size(2)
    else:
        pass

    print("printing args:")
    print(args.n_input_dim)
    print(args.n_heads)
    print(args.set_transf_hidden)

    net = model.NeSyConceptLearner(n_input_dim=args.n_input_dim, device=args.device,
                                   n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden)
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
    plot = args.explain 
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        scheduler.step()


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
    net = model.NeSyConceptLearner(n_input_dim=args.n_input_dim, device=args.device,
                                   n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden)

    net = net.to(args.device)

    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation, printing test and val statistics:\n")

    acc_test = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    acc_val = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)


    # plot expls
    # TODO: Set values back to plot=True

    run(net, train_loader, optimizer, criterion, split='train_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)
    run(net, val_loader, optimizer, criterion, split='val_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)

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
                             n_input_dim=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
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
                             n_input_dim=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
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
    the input will be prepared accordingly. E.g. the input for SAX will be one-hot encoded
    """

    # Preparing the input
    if(args.concept == "sax"):
        input = nn.functional.one_hot(input, num_classes=args.alphabet_size)
    elif(args.concept == "tsfresh"):
        input = input.squeeze(1).to("cpu").numpy()
        input = args.scaler.transform(input)
        input = torch.tensor(input, device="cuda").unsqueeze(1)

    # Applying the SetTransformer
    output_cls, output_attr = net(input)
    
    # preds = (output_cls > 0).float()
    _, preds = torch.max(output_cls, 1)
    
    return output_cls, output_attr, preds

def gridsearch(args):

    ### Warning: settings, batch_size, set_heads and hidden_dim are overwritten here,
    # so the values from the script are irrelevant for gridsearch.

    if args.concept == "sax":
    
        """
        test_accuracies = []
        val_accuracies = []

        ### SAX
        if args.concept == "sax":
            segments = (128, 256, 512, 1024)
            alphabet_sizes = (4, 10, 16, 32)
            set_heads = (4,)

            iteration_list = product(segments, alphabet_sizes, set_heads)

            for (s, a, s_h) in iteration_list:
                args.n_segments = s
                args.alphabet_size = a
                args.set_heads = s_h

                acc_test, acc_val = train(args)
                test_accuracies.append(acc_test)
                val_accuracies.append(acc_val)

            print("n_segments,alphabet_size,set_heads,test_acc,val_acc")
            for (i, (s, a, s_h)) in enumerate(iteration_list):
                print(f"{s},{a},{s_h},{100 * test_accuracies[i]:.3f},{100 * val_accuracies[i]:.3f}")
                #print(f"n_segments: {s}, alphabet_size: {a}, set_heads: {s_h}, test_acc: {100 * test_accuracies[i]:.3f}; val_acc: {100 * val_accuracies[i]:.3f}")
                #print('n_segments: {}, alphabet_size: {}, test_accuracy: {:.3f}, val_accuracy: {:.3f}'
                #     .format(s, a, test_accuracies[i]*100, val_accuracies[i]*100))
        """

# --concept sax --n-segments 32 --alphabet-size 10 --n-heads 4 --set-transf-hidden 128 \
        parameter_names = ("n_segments", "alphabet_size", "n_heads", "set_transf_hidden", "lr")
        
        # n_segments = (4, 8, 16, 32)
        n_segments = (64, 128, 256, 512)
                       
        # n_segments = (32, 64, 128, 256, 512)                
        # n_segments = (32, )            
            
        alphabet_size = (4, 8, 16, 32, 64)
        # alphabet_size = (8,10,12,14,16)
        # alphabet_size = (64, )

        n_heads = (4, )
        set_transf_hidden = (128, )
        lr = (0.0001,)
        params = (n_segments, alphabet_size, n_heads, set_transf_hidden, lr)


    ### tsfresh
    elif args.concept == "tsfresh":
        """
        # settings = ("slow", "mid")
        # set_heads = (4, 8, 16,)
        # hidden_dim = (128, 256, 512)


        ### Warning: settings, batch_size, set_heads and hidden_dim are overwritten here,
        # so the values from the script are irrelevant for gridsearch.
        settings = ("slow", )
        set_heads = (4, )
        hidden_dim = (128, )

        args.batch_size = 128


        np.random.seed(args.seed)
        # 5 seeds are generated for acc calculation.
        seeds = np.random.randint(0, 10000, size=args.num_tries).tolist()
        print(f"seeds: {seeds}")

        accs_test = []
        accs_val = []
        iteration_list = list(product(settings, set_heads, hidden_dim))

        parameter_names = ('ts_setting, n_heads, set_transf_hidden')

        filename = f"gridsearch/gridsearch_{utils.get_current_time()}.csv"
        with open(filename, "a") as file:
            file.write("setting,set_heads,hidden_dim,val_acc,test_acc\n")
            file.flush()
            # for (i, (ts_set, s, h)) in enumerate(iteration_list):
            for (i, params) in enumerate(iteration_list):

                for k in range(len(params)):
                    setattr(args, parameter_names[k], params[k])

                print(f"args.ts_setting: {args.ts_setting}")
                print(f"args.n_heads: {args.n_heads}")
                print(f"args.set_transf_hidden: {args.set_transf_hidden}")

                # args.ts_setting = ts_set
                # args.n_heads = s
                # args.set_transf_hidden = h

                for j in range(len(seeds)):

                    print(f"\nTraining {i+1}/{len(iteration_list)}, Seed {j+1}/{len(seeds)}:",
                          f" setting={params[0]}, set_heads={params[1]}, hidden_dim={params[2]}")
                    # set_seed(seeds[j])
                    utils.seed_everything(seeds[j])

                    # torch.cuda.empty_cache()
                    acc_test, acc_val = train(args)
                    accs_test.append(acc_test)
                    accs_val.append(acc_val)

                # Calculate the average and the maximum deviation OF the different seeds
                avg_acc_test = np.mean(accs_test)
                dev_acc_test = np.max(np.abs(accs_test - avg_acc_test))
                avg_acc_val = np.mean(accs_val)
                dev_acc_val = np.max(np.abs(accs_val - avg_acc_val))

                acc_test = f"{100 * avg_acc_test:.2f} ± {100 * dev_acc_test:.2f}"
                acc_val = f"{100 * avg_acc_val:.2f} ± {100 * dev_acc_val:.2f}"

                # file.write(f"{ts_set},{s},{h},{acc_val},{acc_test}\n")
                file.write(f"{params[0]},{params[1]},{params[2]},{acc_val},{acc_test}\n")
                file.flush()
                print(f"seeds: {seeds}")
                print(f"accs_test: {accs_test}")
                print(f"accs_val: {accs_val}")
            
            """
        
        # Setup of parameters to iterate over.  
        parameter_names = ("ts_setting", "n_heads", "set_transf_hidden")
        ts_setting = ("slow", )
        n_heads = (4, )
        set_transf_hidden = (128, )
        params = (ts_setting, n_heads, set_transf_hidden)

    elif args.concept == "vqshape":
        parameter_names = ("n_heads", "set_transf_hidden")
        # parameter_names = ("used_model", "n_heads", "set_transf_hidden")
        # used_model = (0, )
        n_heads = (4, )
        set_transf_hidden = (128, )
        params = (n_heads, set_transf_hidden)
        # params = (used_model, n_heads, set_transf_hidden)

    gridsearch_helper(parameter_names, params, args)


def gridsearch_helper(param_names, param_lists, args):


    iteration_list = list(product(*param_lists))

    np.random.seed(args.seed)
    # 5 seeds are generated for acc calculation.
    seeds = np.random.randint(0, 10000, size=args.num_tries).tolist()
    print(f"seeds: {seeds}")

    filename = f"gridsearch/{args.concept}/gs_{args.concept}_{utils.get_current_time()}.csv"
    with open(filename, "a") as file:

        param_names_csv = ",".join(param_names)
        file.write(f"{param_names_csv},val_acc,test_acc\n")
        file.flush()
        for (i, params) in enumerate(iteration_list):
            accs_test = []
            accs_val = []

            ### TODO: Add dictionary
            # args.ts_setting = d['ts_setting']

            ### Sth like this:
            
            # attributes = ["ts_setting", "another_setting", "hidden_dim"]  # List of keys to map

            # for attr in attributes:
            #     setattr(args, attr, d[attr])  # Equivalent to args.ts_setting = d['ts_setting']

            for k in range(len(params)):
                setattr(args, param_names[k], params[k])

            # args.ts_setting = ts_set
            # args.n_heads = s
            # args.set_transf_hidden = h

            for j in range(len(seeds)):
                params_with_values = [f"{param_names[x]}={params[x]}" for x in range(len(params))]

                # print(f"\nTraining {i+1}/{len(iteration_list)}, Seed {j+1}/{len(seeds)}: setting={ts_set}, set_heads={s}, hidden_dim={h}")
                print(f"\nTraining {i+1}/{len(iteration_list)}, Seed {j+1}/{len(seeds)}:",
                        f"{', '.join(params_with_values)}")
                    # set_seed(seeds[j])
                utils.seed_everything(seeds[j])

                # torch.cuda.empty_cache()
                acc_test, acc_val = train(args)
                accs_test.append(acc_test)
                accs_val.append(acc_val)
            """ 
            # Calculate the average and the maximum deviation of the different seeds
            avg_acc_test = np.mean(accs_test)
            dev_acc_test = np.max(np.abs(accs_test - avg_acc_test))
            avg_acc_val = np.mean(accs_val)
            dev_acc_val = np.max(np.abs(accs_val - avg_acc_val)) """


            print("accs_test")
            print(accs_test)
            print(len(accs_test))
            print("accs_val")
            print(accs_val)
            print(len(accs_val))
            # Calculate the average and the standard deviation of the different seeds
            avg_acc_test = np.mean(accs_test)
            stddiv_acc_test = np.std(accs_test)
            avg_acc_val = np.mean(accs_val)
            stddiv_acc_val = np.std(accs_val)

            acc_test = f"{100 * avg_acc_test:.2f} ± {100 * stddiv_acc_test:.2f}"
            acc_val = f"{100 * avg_acc_val:.2f} ± {100 * stddiv_acc_val:.2f}"

            file.write(f"{','.join(str(p) for p in params)},{acc_val},{acc_test}\n")
            # file.write(f"{ts_set},{s},{h},{acc_val},{acc_test}\n")
            file.flush()
            print(f"seeds: {seeds}")
            print(f"accs_test: {accs_test}")
            print(f"accs_val: {accs_val}")
            
def set_seed(seed):
    if seed == -1:  
        seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    if args.mode == 'train':
        # torch.cuda.empty_cache()
        # set_seed(args.seed)
        utils.seed_everything(args.seed)
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'plot':
        plot(args)
    elif args.mode == 'gridsearch':
        gridsearch(args)

if __name__ == "__main__":
    main()
