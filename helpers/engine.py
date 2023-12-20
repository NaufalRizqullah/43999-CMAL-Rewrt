import torch
import os
import json

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from helpers.utils import timing_decorator, cosine_anneal_schedule, map_generate, attention_im, highlight_im, test_tresnetl

@timing_decorator
def train(model, dataLoader, nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, data_path = ''):
    # Create empty results dictionary
    results = {
        "epoch": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    exp_dir = store_name

    try:
        os.makedirs(exp_dir, exist_ok=True)
    except FileExistsError:
        print(f"[INFO] Directory {exp_dir} already exists")
        pass

    use_cuda =torch.cuda.is_available()
    print(f"[INFO] CUDA status: {use_cuda}")

    train_loader = dataLoader
    print("[INFO] DataLoader created.")

    print(f"[INFO] Initialize net model.")
    net = model

    print(f"[INFO] Make net model Parallel.")
    netp = torch.nn.DataParallel(net, device_ids=[0])

    print(f"[INFO] Set Device net model.")
    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()

    print(f"[INFO] Setup Optimizer.")
    optimizer = optim.SGD([
        {"params": net.classifier_concat.parameters(), "lr": 0.002},

        {"params": net.conv_block1.parameters(), "lr": 0.002},
        {"params": net.classifier1.parameters(), "lr": 0.002},

        {"params": net.conv_block2.parameters(), "lr": 0.002},
        {"params": net.classifier2.parameters(), "lr": 0.002},

        {"params": net.conv_block3.parameters(), "lr": 0.002},
        {"params": net.classifier3.parameters(), "lr": 0.002},

        {"params": net.features.parameters(), "lr": 0.0002}
    ],
       lr=0.002, momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    print(f"[INFO] Training...")
    for epoch in range(start_epoch, nb_epoch):
        print(f"\n[STATUS] Epoch: {epoch}")

        net.train()

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0

        correct = 0
        total = 0
        idx = 0


        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx

            if inputs.shape[0] < batch_size:
                continue

            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            # Trying Fix 1
            # inputs = inputs.clone().detach().requires_grad_(True)
            # targets = targets.clone().detach().requires_grad_(True)

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]["lr"] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Train `the Experts from deep to shallow` with data augmentation by multiple steps
            # e3
            optimizer.zero_grad()
            inputs3 = inputs
            output_1, output_2, output_3, _, map1, map2, map3 = netp(inputs3)

            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            p1 = net.state_dict()["classifier3.1.weight"]
            p2 = net.state_dict()["classifier3.4.weight"]
            att_map_3 = map_generate(map3, output_3, p1, p2)
            inputs3_att = attention_im(inputs, att_map_3)

            p1 = net.state_dict()["classifier2.1.weight"]
            p2 = net.state_dict()["classifier2.4.weight"]
            att_map_2 = map_generate(map2, output_2, p1, p2)
            inputs2_att = attention_im(inputs, att_map_2)

            p1 = net.state_dict()["classifier1.1.weight"]
            p2 = net.state_dict()["classifier1.4.weight"]
            att_map_1 = map_generate(map1, output_1, p1, p2)
            inputs1_att = attention_im(inputs, att_map_1)

            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)

            # e2
            optimizer.zero_grad()
            flag = torch.rand(1).item()
            if flag < 1/3:
                inputs2 = inputs3_att
            elif flag < 2/3:
                inputs2 = inputs1_att
            else:
                inputs2 = inputs

            _, output_2, _, _, _, map2, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # e1
            optimizer.zero_grad()
            flag = torch.rand(1).item()
            if flag < 1/3:
                inputs1 = inputs3_att
            elif flag < 2/3:
                inputs1 = inputs2_att
            else:
                inputs1 = inputs

            output_1, _, _, _, map1, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Train `the Experts and their concatenation` with the overall attention region in one go.
            optimizer.zero_grad()
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = netp(inputs_ATT)

            concat_loss_ATT = CELoss(output_1_ATT, targets) + \
                            CELoss(output_2_ATT, targets) + \
                            CELoss(output_3_ATT, targets) + \
                            CELoss(output_concat_ATT, targets) * 2
            concat_loss_ATT.backward()
            optimizer.step()

            optimizer.zero_grad()
            _, _, _, output_concat, _, _, _ = netp(inputs)

            concat_loss = CELoss(output_concat, targets)
            concat_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss_ATT.item()
            train_loss5 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f'Step: {batch_idx} | Loss1: {train_loss1 / (batch_idx + 1):.3f} | Loss2: {train_loss2 / (batch_idx + 1):.5f} | Loss3: {train_loss3 / (batch_idx + 1):.5f} | Loss_ATT: {train_loss4 / (batch_idx + 1):.5f} | Loss_concat: {train_loss5 / (batch_idx + 1):.5f} | Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})'
                )

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)

        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                f'Iteration {epoch} | train_acc = {train_acc:.5f} | train_loss = {train_loss:.5f} | '
                f'Loss1: {train_loss1 / (idx + 1):.3f} | Loss2: {train_loss2 / (idx + 1):.5f} | '
                f'Loss3: {train_loss3 / (idx + 1):.5f} | Loss_ATT: {train_loss4 / (idx + 1):.5f} | '
                f'Loss_concat: {train_loss5 / (idx + 1):.5f} |\n'
            )


        print(f"[INFO] Testing...")
        print(f"[INFO] Evaluation Test on Epoch: {epoch}")
        val_acc_com, val_loss = test_tresnetl(net, CELoss, 3, data_path+'/test')
        print(f"Validation Combine Loss: {val_acc_com} | Validation Loss: {val_loss} \n")
        
        with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc_com, val_loss))
        
        MODEL_FILENAME = f"/model_{epoch}_epoch.pth"
        
        # Only Save model if better in validation
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com

            net.cpu()

            print(f"[INFO] Saving Model cause Better Validation ACC")
            torch.save(net, './' + store_name + MODEL_FILENAME)

            net.to(device)

        # Update results dictionary
        results["epoch"] = epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc_com)

        # Save to JSON file
        with open(exp_dir + '/results.json', 'w') as json_file:
            json.dump(results, json_file)

    # return some value to eval later (results of data loss acc)
    return results