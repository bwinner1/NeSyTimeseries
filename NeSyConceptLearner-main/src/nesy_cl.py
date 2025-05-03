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
        args.classes = np.arange(2)

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

        for i, loaded_data in enumerate(tqdm(loader)):
            concepts = loaded_data[0]
            labels = loaded_data[1]

            if args.xil:
                masks = loaded_data[3].float().to(args.device)

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

            if args.xil:
                saliencies = utils.generate_intgrad_captum_table(net.set_cls, output_attr, preds)
                loss = calc_xil_loss(loss, masks, saliencies, args.xil_weight)


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
    # Only useful for tsfresh global importance
    if args.explain_all:
        args.best_features = [{}, {}]
        args.worst_features = [{}, {}]

    for i, loaded_data in enumerate(loader, start=epoch * iters_per_epoch):

        concepts = loaded_data[0]
        labels = loaded_data[1]
        ts = loaded_data[2]
    
        if args.xil:
            masks = loaded_data[3].float().to(args.device)

        # Move both tensors to correct device
        concepts = concepts.to(args.device)
        labels = labels.to(args.device)

        # Network usage
        output_cls, output_attr, preds = apply_net(concepts, net, args)

        loss = criterion(output_cls, labels)

        # When using XIL, calculate the importance values of the summarizer
        if args.xil:
            saliencies = utils.generate_intgrad_captum_table(net.set_cls, output_attr, preds)
            loss = calc_xil_loss(loss, masks, saliencies, args.xil_weight)


        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(labels.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

        # Plot predictions in Tensorboard
        if plot and not(i % iters_per_epoch) and not args.concept == "vqshape":
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

        # 'Normal' | 'Decoy'
        mode = 'Decoy' if args.p2s_decoy else 'Normal'
        train_dataset = load_dataset('AIML-TUDA/P2S', mode, download_mode='reuse_dataset_if_exists')

        # Extracting the time series data from the train and test dataset
        ts_train = np.array(train_dataset['train']['dowel_deep_drawing_ow'])
        # ts_train_speeds = np.array(train_dataset['train']['speed'])
        
        # Number of samples that goes into the train dataset; the rest goes into the validation dataset
        training_samples = int(ts_train.shape[0] * 0.8) 
        ts_val = ts_train[training_samples :]
        
        ts_train = ts_train[: training_samples]
        ts_test = np.array(train_dataset['test']['dowel_deep_drawing_ow'])


        labels_train = torch.tensor(train_dataset['train']['label'][:training_samples])
        labels_val = torch.tensor(train_dataset['train']['label'][training_samples:])
        labels_test = torch.tensor(train_dataset['test']['label'])

        masks_train = torch.tensor(train_dataset['train']['mask'][:training_samples])
        masks_val = torch.tensor(train_dataset['train']['mask'][training_samples:])
        masks_test = torch.tensor(train_dataset['test']['mask'])
        

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

        vqshape_model = model.vqshapeTransformer(args.model)

        with torch.no_grad():
            concepts_train = vqshape_model.transform(ts_train)
            concepts_val = vqshape_model.transform(ts_val)
            concepts_test = vqshape_model.transform(ts_test)

            if args.explain and args.save_pdf:
                dict = vqshape_model.transform(ts_test, mode="evaluate")[0]
                path = utils.prepare_xai_path("vqshape")
                utils.plot_expl_vqshape(dict, args.save_pdf, path)
        empty_gpu_memory()

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
    train_input = [concepts_train, labels_train, ts_train]
    val_input = [concepts_val, labels_val, ts_val]
    test_input = [concepts_test, labels_test, ts_test]
    
    if args.xil:
        train_input.append(masks_train)
        val_input.append(masks_val)
        test_input.append(masks_test)

    train_dataset = TensorDataset(*train_input)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_dataset = TensorDataset(*val_input)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    test_dataset = TensorDataset(*test_input)
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

    run(net, train_loader, optimizer, criterion, split='train_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)
    run(net, val_loader, optimizer, criterion, split='val_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)
    run(net, test_loader, optimizer, criterion, split='test_best', args=args,
        writer=writer, train=False, plot=plot, epoch=0)

    writer.close()

    if not args.tr_acc:
        return acc_test, acc_val
    else:

        acc_train = get_confusion_from_ckpt(net, train_loader, criterion, args=args, datasplit='test_best',
                        writer=writer)
        return acc_test, acc_val, acc_train
        
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

def calc_xil_loss(ra_loss, masks, saliencies, rr_weight):

    # saliencies = torch.max(saliencies, dim=1).values

    masks = masks.unsqueeze(1)
    masks = nn.functional.interpolate(masks, saliencies.size(1), mode='linear')  # first interpolate to seg_size timesteps
    masks = masks.squeeze(1)
    masks = masks.unsqueeze(2)

    # print("masks")
    # print(masks.size())
    # print("saliencies")
    # print(saliencies.size())

    saliencies = torch.mul(saliencies, masks)

    saliencies = torch.mul(saliencies, saliencies)
    # print("saliencies")
    # print(saliencies)

    rr_loss = torch.sum(saliencies)
    # print("rr_loss")
    # print(rr_loss)


    loss =  ra_loss + rr_weight * rr_loss

    return loss

def gridsearch(args):
    """
    Gridsearch setup method. Here you can specify the values that should be iterated 
    over during the optimal parameter search. Each parameter configuration will be run 5 
    times per default, as defined in nesy_cl.sh
    """

    ### WARNING ###
    # Every value that is used here, overwrites the given value in the script.
    # If you want to add a parameter to iterate through, do the following steps
    # 0) Below, only use the exact same parameter names as you would in the script 
    # 1) Define it as a tuple, having at least one value (e.g.: (4, ) )
    # 2) Include your parameter in params
    # 3) Add the parameter into parameter_names

    if args.concept == "sax":
         
        ### Step 1)

        n_segments = (32, )
        alphabet_size = (64, )
    
        n_heads = (4, )
        set_transf_hidden = (128, )

        p2s_decoy = (True, )
        xil = (False, )
        xil_weight = (0.0001, )
        tr_acc = (True, ) 

        ### Step 2)
        params = (n_segments, alphabet_size, n_heads,
                   set_transf_hidden, p2s_decoy, xil, xil_weight, tr_acc)

        ### Step 3)
        parameter_names = ("n_segments", "alphabet_size", "n_heads",
                            "set_transf_hidden", "p2s_decoy", "xil", "xil_weight", "tr_acc")
        
    ### tsfresh
    elif args.concept == "tsfresh":
        
        # Setup of parameters to iterate over.  
        ts_setting = ("fast", "mid", "slow")

        n_heads = (4, )
        set_transf_hidden = (128, )
        filter_tsf = (True, False)
        normalize_tsf = (True, False)

        params = (ts_setting, n_heads, set_transf_hidden, filter_tsf, normalize_tsf)
        parameter_names = ("ts_setting", "n_heads", "set_transf_hidden", "filter_tsf", "normalize_tsf")


    elif args.concept == "vqshape":

        n_heads = (4, )
        set_transf_hidden = (128, )
        model = (1, 2)
        lr = (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001,)
        p2s_decoy = (False, )

        params = (n_heads, set_transf_hidden, model, p2s_decoy, lr, )
        parameter_names = ("n_heads", "set_transf_hidden", "model", "p2s_decoy", "lr")


    gridsearch_helper(parameter_names, params, args)


def gridsearch_helper(param_names, param_lists, args):
    """
    Actual gridsearch method. Runs the model with each possible configuration and writes a file
    with the used parameters and the resulting accuracy values.
    """

    # Extract the specified parameter lists, build a product and convert the result to a list
    iteration_list = list(product(*param_lists))

    np.random.seed(args.seed)
    # 5 (num_tries) seeds are generated for accuracy calculation
    seeds = np.random.randint(0, 10000, size=args.num_tries).tolist()
    print(f"seeds: {seeds}")

    filename = f"gridsearch/{args.concept}/gs_{args.concept}_{utils.get_current_time()}.csv"
    with open(filename, "a") as file:

        param_names_csv = ",".join(param_names)
        if args.tr_acc:
            file.write(f"{param_names_csv},val_acc,test_acc\n")
        else:
            file.write(f"{param_names_csv},val_acc,test_acc,train_acc\n")
        
        file.flush()
        for (i, params) in enumerate(iteration_list):
            accs_test = []
            accs_val = []
            accs_train = []

            for k in range(len(params)):
                setattr(args, param_names[k], params[k])

            for j in range(len(seeds)):
                params_with_values = [f"{param_names[x]}={params[x]}" for x in range(len(params))]

                print(f"\nTraining {i+1}/{len(iteration_list)}, Seed {j+1}/{len(seeds)}:",
                        f"{', '.join(params_with_values)}")
                utils.seed_everything(seeds[j])

                if not args.tr_acc:
                    acc_test, acc_val = train(args)
                else:
                    acc_test, acc_val, acc_train = train(args)
                    
                accs_test.append(acc_test)
                accs_val.append(acc_val)
                
                if args.tr_acc:
                    accs_train.append(acc_train)

            print("accs_test")
            print(accs_test)
            print(len(accs_test))
            print("accs_val")
            print(accs_val)
            print(len(accs_val))
            print("accs_train")
            print(accs_train)
            print(len(accs_train))
            
            # Calculate the average and the standard deviation of the different seeds
            avg_acc_test = np.mean(accs_test)
            stddiv_acc_test = np.std(accs_test)
            avg_acc_val = np.mean(accs_val)
            stddiv_acc_val = np.std(accs_val)
            avg_acc_train = np.mean(accs_train)
            stddiv_acc_train = np.std(accs_train)

            acc_test = f"{100 * avg_acc_test:.2f} ± {100 * stddiv_acc_test:.2f}"
            acc_val = f"{100 * avg_acc_val:.2f} ± {100 * stddiv_acc_val:.2f}"
            acc_train = f"{100 * avg_acc_train:.2f} ± {100 * stddiv_acc_train:.2f}"

            if not args.tr_acc:
                file.write(f"{','.join(str(p) for p in params)},{acc_val},{acc_test}\n")
            else:
                file.write(f"{','.join(str(p) for p in params)},{acc_val},{acc_test},{acc_train}\n")
                
            file.flush()
            print(f"seeds: {seeds}")
            print(f"accs_test: {accs_test}")
            print(f"accs_val: {accs_val}")
            
def set_seed(seed):
    """
    Sets a seed for every library. Also used to generate further seeds if a model will be
    evaluated over a few runs using the same parameter configuration. 
    """
    if seed == -1:  
        seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def empty_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    torch.cuda.empty_cache()
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def main():
    args = get_args()
    if args.mode == 'train':
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
