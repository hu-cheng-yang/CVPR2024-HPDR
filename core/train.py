import os
from collections import OrderedDict
import torch
import torch.optim as optim
from misc.util import get_inf_iterator, mkdir
from torch.nn import DataParallel
from tqdm import trange
from misc.evaluate import eval
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from loss.loss import Loss

def Train(args, model,
          data_loader1_real, data_loader1_fake,
          data_loader2_real, data_loader2_fake,
          data_loader3_real, data_loader3_fake,
          valid_dataloader,
          Saver, saverootname, init_epoch=None, init_step=None):

    
    model = model.cuda()
    model = DataParallel(model)
    model.train()

    summary_save_path = os.path.join(args.results_path, saverootname, 'summary')
    mkdir(summary_save_path)
    writer = SummaryWriter(summary_save_path)

    criterion = Loss(20, 256, nb_proxies_top=256).cuda()

    criterion.train()

    training_parameters = list(model.parameters()) + list(criterion.parameters())
    optimizer_all = optim.Adam(training_parameters, lr=args.lr, betas=[0.9, 0.999])
    iternum = max(len(data_loader1_real), len(data_loader1_fake),
                  len(data_loader2_real), len(data_loader2_fake),
                  len(data_loader3_real), len(data_loader3_fake))

    print('iternum={}'.format(iternum))

    global_step = 0
    epoch = 0
    max_iter = 0
    max_epoch = 0
    max_auc = 0
    min_hter = 1.0

    if init_epoch != None:
        epoch = init_epoch - 1
        print('epoch', epoch)
        init_epoch = None
    for epoch in range(args.epochs):

        data1_real = get_inf_iterator(data_loader1_real)
        data1_fake = get_inf_iterator(data_loader1_fake)

        data2_real = get_inf_iterator(data_loader2_real)
        data2_fake = get_inf_iterator(data_loader2_fake)

        data3_real = get_inf_iterator(data_loader3_real)
        data3_fake = get_inf_iterator(data_loader3_fake)

        step = 0

        if init_step != None:
            step = init_step - 1
            print('step', step)
            init_step = None
        range_bar = trange(iternum)
        for step in range_bar:
            model.train()
            criterion.train()

            cat_img1_real, cat_img1_real_k, depth_img1_real, lab1_real, _, _ = next(data1_real)
            cat_img1_fake, cat_img1_fake_k, depth_img1_fake, lab1_fake, _, _ = next(data1_fake)

            cat_img2_real, cat_img2_real_k, depth_img2_real, lab2_real, _, _ = next(data2_real)
            cat_img2_fake, cat_img2_fake_k, depth_img2_fake, lab2_fake, _, _ = next(data2_fake)

            cat_img3_real, cat_img3_real_k, depth_img3_real, lab3_real, _, _ = next(data3_real)
            cat_img3_fake, cat_img3_fake_k, depth_img3_fake, lab3_fake, _, _ = next(data3_fake)

            bs = cat_img3_real.shape[0]

            catimg1 = torch.cat([cat_img1_real, cat_img1_fake], 0).cuda()
            catimg1_k = torch.cat([cat_img1_real_k, cat_img1_fake_k], 0).cuda()
            depth_img1 = torch.cat([depth_img1_real, depth_img1_fake], 0).cuda()
            lab1 = torch.cat([lab1_real, lab1_fake], 0).float().cuda()

            catimg2 = torch.cat([cat_img2_real, cat_img2_fake], 0).cuda()
            catimg2_k = torch.cat([cat_img2_real_k, cat_img2_fake_k], 0).cuda()
            depth_img2 = torch.cat([depth_img2_real, depth_img2_fake], 0).cuda()
            lab2 = torch.cat([lab2_real, lab2_fake], 0).float().cuda()

            catimg3 = torch.cat([cat_img3_real, cat_img3_fake], 0).cuda()
            catimg3_k = torch.cat([cat_img3_real_k, cat_img3_fake_k], 0).cuda()
            depth_img3 = torch.cat([depth_img3_real, depth_img3_fake], 0).cuda()
            lab3 = torch.cat([lab3_real, lab3_fake], 0).float().cuda()

            catimg = torch.cat([catimg1, catimg2, catimg3], 0)
            catimg_k = torch.cat([catimg1_k, catimg2_k, catimg3_k], 0)
            depth_GT = torch.cat([depth_img1, depth_img2, depth_img3], 0)
            label = torch.cat([lab1, lab2, lab3], 0)

            domain_label = torch.cat([torch.ones(bs) * 0,
                                      torch.ones(bs) * 3,
                                      torch.ones(bs) * 1,
                                      torch.ones(bs) * 4,
                                      torch.ones(bs) * 2,
                                      torch.ones(bs) * 5]).long().cuda()

            _, _, feat = model(catimg)
            _, _, feat_1 = model(catimg_k)
            loss_prot, loss_dict = criterion(feat, feat_1, label.long(), domain_label.long())

            Loss_all = loss_prot
            Loss_all.backward()

            optimizer_all.step()
            optimizer_all.zero_grad()

            range_bar.set_postfix(**loss_dict)

            for k, v in loss_dict.items():
                writer.add_scalar(k, v, global_step=global_step)

            if (step + 1) % args.log_step == 0:
                errors = loss_dict
                best = OrderedDict([
                    ('BEST epoch', max_epoch),
                    ("BEST iter", max_iter),
                    ("BEST HTER", min_hter),
                    ("BEST AUC", max_auc),
                ])
                Saver.print_current_errors((epoch + 1), (step + 1), errors)
                Saver.print_current_errors((epoch + 1), (step + 1), best)

            global_step += 1

            if ((step + 1) % args.model_save_step == 0):
                model_save_path = os.path.join(args.results_path, saverootname, 'snapshots')
                _, HTER, AUC, _, _= eval(valid_dataloader, model, criterion, isLoop=True, isProt=args.test_prot)
                metric = OrderedDict([
                       ('FULL', int(0)),
                       ('HTER', HTER),
                       ('AUC', AUC),
                       ])
                Saver.print_current_errors((epoch + 1), (step + 1), metric)
                print(str(epoch + 1) + " " + str(step + 1) + " " + str(metric))
                if HTER < args.hter or AUC > args.auc:
                    _, HTER, AUC, _, _ = eval(valid_dataloader, model, criterion, isLoop=False, isProt=args.test_prot)
                    metric = OrderedDict([
                           ('FULL', int(1)),
                           ('HTER', HTER),
                           ('AUC', AUC),
                           ])
                    Saver.print_current_errors((epoch + 1), (step + 1), metric)
                    if min_hter > HTER or max_auc < AUC:
                       min_hter = HTER
                       max_auc = AUC
                       max_iter = step + 1
                       max_epoch = epoch + 1
                       args.hter = HTER + 0.02
                       args.auc = AUC - 0.02
                       torch.save(model.state_dict(), os.path.join(model_save_path,
                                                                "model-{}-{}.pt".format(epoch + 1, step + 1)))
                       torch.save(criterion.state_dict(), os.path.join(model_save_path,
                                                                 "prot-{}-{}.pt".format(epoch + 1, step + 1)))
                       
                       torch.save(model.state_dict(), os.path.join(model_save_path,
                                                                "model-best.pt"))
                       torch.save(criterion.state_dict(), os.path.join(model_save_path,
                                                                 "prot-best.pt"))
                    print("BEST epoch: {} iter: {}, HTER: {}, AUC: {}".format(max_epoch, max_iter, min_hter, max_auc))
