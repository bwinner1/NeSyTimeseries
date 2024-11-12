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
import model
import utils as utils
from rtpt import RTPT
from args import get_args

torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import DatasetDict

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

        for i, sample in enumerate(tqdm(loader)):
            # input is either a set or an image
            imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)
            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            output_cls, output_attr = net(imgs)
            # class prediction
            _, preds = torch.max(output_cls, 1)

            labels_all.extend(img_class_ids.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == img_class_ids)
            loss = criterion(output_cls, img_class_ids)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = img_class_ids.cpu().numpy()
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
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):

        # input is either a set or an image
        #imgs, target_set, img_class_ids, img_ids, _, table_expl = map(lambda x: x.cuda(), sample)


        time_series, labels, speed, mask = sample.values()
        """
        labels = sample['label']
        speeds = sample['speed']

        concepts_list = sample['dowel_deep_drawing_ow'][0]
        concepts = torch.stack(concepts_list).T
        # converts list of size (n_segments x Tensor (batch_size))
        # into a Tensor of size (batch_size, n_segments)
        
        masks_list = sample['mask']
        masks = torch.stack(masks_list).T
        # converts list 
        # into a Tensor of size (batch_size, time_series)

        print()
        print(concepts.size())
        print(labels.size())
        print(speeds.size())
        print(masks.size())
        input = (concepts_list, speeds, masks_list)
 """

        
        #img_class_ids = img_class_ids.long()
        labels = labels.long()

        input = torch.stack((time_series[0], speed, mask))
        # forward evaluation through the network
        # output_cls, output_attr = net(input)
        output_cls, output_attr = net(input)

        # class prediction
        #_, preds = torch.max(output_cls, 1)
        preds = (output_cls > 0.5).long()

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
            utils.write_expls(net, loader, f"Expl/{split}", epoch, writer)

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def train(args):
    print("running train method...")
    if args.dataset == "p2s":
        dataset = load_dataset('AIML-TUDA/P2S', 'Normal', download_mode='reuse_dataset_if_exists')

        # Extracting the time series data from the train and test dataset 
        ts_train = dataset['train']['dowel_deep_drawing_ow']
        ts_test = dataset['test']['dowel_deep_drawing_ow']

        ts_train = np.array(ts_train)
        ts_test = np.array(ts_test)

        # Adding a third dimension, in the middle of the shape, as needed for SAX
        ts_train = ts_train.reshape(ts_train.shape[0], 1, ts_train.shape[1])
        ts_test = ts_test.reshape(ts_test.shape[0], 1, ts_test.shape[1])
    else:
        print("Wrong dataset specifier")
        exit()



    if args.concept == "sax":
        sax = model.SAXTransformer(n_segments=args.n_segments, alphabet_size=args.alphabet_size)
        concepts_train, _, _ = sax.transform(ts_train)
        concepts_test, _, _ = sax.transform(ts_test)
    ### TODO: Add other cases
    #elif args.concept == "tsfresh":
    #elif args.concept == "vq-vae":
    else:
        print("Wrong concept specifier")
        exit()


    # TODO: Delete this two lines, as a dataset object can't be changed
    #dataset['train']['dowel_deep_drawing_ow'] = concepts_train
    #dataset['test']['dowel_deep_drawing_ow'] = concepts_test

    ## TODO: Make sure that the added lines here actually work:

    # For all samples in the training dataset, ignore the current value and overwrite it with the value from dataset_train
    dataset_train = dataset['train'].map(
        lambda _, idx: {'dowel_deep_drawing_ow': concepts_train[idx]},
        with_indices=True,  # Allows access to the row index
        batched=False       # Process one example at a time
    )

    # Overwrite the dataset with the concept dataset
    dataset_test = dataset['test'].map(
        lambda _, idx: {'dowel_deep_drawing_ow': concepts_test[idx]},
        with_indices=True,
        batched=False
    )

    # Update the dataset with the new train and test splits
    dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    train_loader = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset['test'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    """ 
    print("Testing loaders:")
    for i, sample in enumerate(train_loader):
        print(f"i={i}")
 """

    #n_classes is 2, as there are two possible outcomes for a sample, either defect or not 
    args.n_classes = 2
    args.class_weights = torch.ones(args.n_classes)/args.n_classes
    args.classes = np.arange(args.n_classes)

    #Probably irrelevant
    #if(args.concept == "sax"):
        #All values in SAX are means, therefore there is only one category, starting at index 0.
    #    args.category_ids = [0]
    #elif args.concept == "tsfresh":
        # add a staring index for each category type, f.e. means, general data (variance, overall mean, etc.)
    #elif args.concept == "vq-vae":


    #net = model.NeSyConceptLearner(n_classes=args.n_classes, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
     #                        n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
      #                       category_ids=args.category_ids, device=args.device)
    
    
    net = model.NeSyConceptLearner(n_classes=2, n_attr=args.n_segments,
                                 n_set_heads=args.alphabet_size, set_transf_hidden=128)

    # from other file
      
    #net = NeSyConceptLearner(n_classes=2, n_slots=10, n_iters=3, n_attr=6, n_set_heads=4, set_transf_hidden=128,
    #                         category_ids = [3, 6, 8, 10, 17], device=device).to(device)

    # load pretrained concept embedding module
    #log = torch.load("logs/slot-attention-clevr-state-3_final", map_location=torch.device(args.device))
    #net.img2state_net.load_state_dict(log['weights'], strict=True)
    #print("Pretrained slot attention model loaded!")

    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"Clevr Hans Slot Att Set Transf xil",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # tensorboard writer
    writer = utils.create_writer(args)

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        scheduler.step()
        val_loss = run(net, val_loader, optimizer, criterion, split='val', args=args, writer=writer,
                       train=False, plot=True, epoch=epoch)
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
    # net = model.NeSyConceptLearner(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
    #                        set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
    #                        device=args.device)
    net = model.NeSyConceptLearner(n_classes=2, n_attr=args.n_attr,
                           set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
                           device=args.device)
    net = net.to(args.device)

    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)

    # plot expls
    run(net, train_loader, optimizer, criterion, split='train_best', args=args,
        writer=writer, train=False, plot=True, epoch=0)
    run(net, val_loader, optimizer, criterion, split='val_best', args=args,
        writer=writer, train=False, plot=True, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test_best', args=args,
        writer=writer, train=False, plot=True, epoch=0)

    writer.close()


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


def main():
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'plot':
        plot(args)


if __name__ == "__main__":
    main()
