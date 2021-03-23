# encoding: utf-8


import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.Discriminator import Discriminator
from models.HidingRes import HidingRes
import numpy as np
from PIL import Image
from vgg import Vgg16


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--Dnet', default='',
                    help="path to Discriminator (to continue training)")
parser.add_argument('--trainpics', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output test images')
parser.add_argument('--runfolder', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='/data-x/g10/zhangjie/PAMI/exp_chk/debone/HR/',
                    help='folder to save the experiment codes')

parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


#datasets to train
parser.add_argument('--datasets', type=str, default='/data-x/g10/zhangjie/PAMI/datasets/debone/For_HR',
                    help='denoise/derain')

#read secret image
parser.add_argument('--secret', type=str, default='flower',
                    help='secret folder')

#hyperparameter of loss

parser.add_argument('--beta', type=float, default=1,
                    help='hyper parameter of beta :secret_reveal err')
parser.add_argument('--betagan', type=float, default=1,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betagans', type=float, default=0.01,
                    help='hyper parameter of beta :gans weight')
parser.add_argument('--betapix', type=float, default=0,
                    help='hyper parameter of beta :pixel_loss weight')

parser.add_argument('--betamse', type=float, default=10000,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betacons', type=float, default=1,
                    help='hyper parameter of beta: consist_loss')
parser.add_argument('--betaclean', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanA', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betacleanB', type=float, default=1,
                    help='hyper parameter of beta: clean_loss')
parser.add_argument('--betavgg', type=float, default=0,
                    help='hyper parameter of beta: vgg_loss')
parser.add_argument('--num_downs', type=int, default= 7 , help='nums of  Unet downsample')
parser.add_argument('--clip', action='store_true', help='clip container_img')


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, optimizerD, writer, logPath, schedulerH, schedulerR
    global val_loader, smallestLoss,  mse_loss, gan_loss, pixel_loss, patch, criterion_GAN, criterion_pixelwise,vgg, vgg_loss

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############

    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = opt.hostname  + "_" + opt.remark + "_" + cur_time
    opt.outckpts += experiment_dir + "/checkPoints"
    opt.trainpics += experiment_dir + "/trainPics"
    opt.validationpics += experiment_dir + "/validationPics"
    opt.outlogs += experiment_dir + "/trainingLogs"
    opt.outcodes += experiment_dir + "/codes"
    opt.testPics += experiment_dir + "/testPics"
    opt.runfolder += experiment_dir + "/run"

    if not os.path.exists(opt.outckpts):
        os.makedirs(opt.outckpts)
    if not os.path.exists(opt.trainpics):
        os.makedirs(opt.trainpics)
    if not os.path.exists(opt.validationpics):
        os.makedirs(opt.validationpics)
    if not os.path.exists(opt.outlogs):
        os.makedirs(opt.outlogs)
    if not os.path.exists(opt.outcodes):
        os.makedirs(opt.outcodes)
    if not os.path.exists(opt.runfolder):
        os.makedirs(opt.runfolder)        
    if (not os.path.exists(opt.testPics)) and opt.test != '':
        os.makedirs(opt.testPics)



    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)
    # tensorboardX writer
    writer = SummaryWriter(log_dir=opt.runfolder, comment='**' + opt.hostname + "_" + opt.remark)

    
    DATA_DIR = opt.datasets
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    
    train_dataset = MyImageFolder(
        traindir,  
        transforms.Compose([ 
            trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
                        
        ]))
    val_dataset = MyImageFolder(
        valdir,  
        transforms.Compose([  
            trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(), 
        ]))
		

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=int(opt.workers))    	


    Hnet = UnetGenerator(input_nc=2, output_nc=1, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)

    Rnet = HidingRes(in_c=1, out_c=1)
    Rnet.cuda()
    Rnet.apply(weights_init)

    Dnet = Discriminator(in_channels=1)
    Dnet.cuda()
    Dnet.apply(weights_init)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.imageSize // 2 ** 4, opt.imageSize // 2 ** 4)


    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

    optimizerD = optim.Adam(Dnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=5, verbose=True)


    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)


    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    print_network(Rnet)

    if opt.Dnet != '':
        Dnet.load_state_dict(torch.load(opt.Dnet))
    if opt.ngpu > 1:
        Dnet = torch.nn.DataParallel(Dnet).cuda()
    print_network(Dnet)


    # define loss
    mse_loss = nn.MSELoss().cuda()
    criterion_GAN = nn.MSELoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()
    vgg = Vgg16(requires_grad=False).cuda()       

    smallestLoss = 10000
    print_log("training is beginning .......................................................", logPath)
    for epoch in range(opt.niter):
        ######################## train ##########################################
        train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet)

        ####################### validation  #####################################
        val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses,vgg_loss, val_sumloss = validation(val_loader,  epoch, Hnet=Hnet, Rnet=Rnet, Dnet=Dnet)

        ####################### adjust learning rate ############################
        schedulerH.step(val_sumloss)
        schedulerR.step(val_rloss)
        schedulerD.step(val_dloss)

        # save the best model parameters
        if val_sumloss < globals()["smallestLoss"]:
            globals()["smallestLoss"] = val_sumloss

            torch.save(Hnet.module.state_dict(),
                       '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss, val_hloss))
            torch.save(Rnet.module.state_dict(),
                       '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss, val_rloss))
            torch.save(Dnet.module.state_dict(),
                       '%s/netD_epoch_%d,sumloss=%.6f,Dloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss, val_dloss))
    writer.close()


def train(train_loader,  epoch, Hnet, Rnet, Dnet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter()  
    R_mselosses = AverageMeter()
    R_consistlosses = AverageMeter()
    Dlosses = AverageMeter()
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses =AverageMeter()    
    SumLosses = AverageMeter()  

    # switch to train mode
    Hnet.train()
    Rnet.train()
    Dnet.train()

    # Tensor type
    Tensor = torch.cuda.FloatTensor 

    loader = transforms.Compose([trans.Grayscale(num_output_channels=1),
        transforms.ToTensor(),])
    clean_img = Image.open("../secret/clean.png")
    clean_img = loader(clean_img)  
    secret_img = Image.open("../secret/flower.png")
    secret_img = loader(secret_img)  

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        this_batch_size = int(data.size()[0]) 
        cover_img = data[0:this_batch_size, :, :, :]  
        cover_img_A = cover_img[ :, :, 0:256, 0:256]
        cover_img_B = cover_img[ :, :, 0:256, 256:512]

        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
        secret_img = secret_img[0:this_batch_size, :, :, :]  
   
        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)         
        clean_img = clean_img[0:this_batch_size, :, :, :] 

        if opt.cuda:
            cover_img = cover_img.cuda()
            cover_img_A = cover_img_A.cuda()
            cover_img_B = cover_img_B.cuda()        
            secret_img = secret_img.cuda()
            clean_img = clean_img.cuda()


        concat_img = torch.cat([cover_img_B, secret_img], dim=1)
        concat_imgv = Variable(concat_img) 
        cover_imgv = Variable(cover_img_B)
        container_img = Hnet(concat_imgv)  
        A_imgv = Variable(cover_img_A)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)

        pred_fake = Dnet(container_img)
        gan_loss = criterion_GAN(pred_fake, valid)
        pixel_loss = criterion_pixelwise(container_img, cover_imgv)  #l1

        container_img_rgb = container_img.repeat(1,3,1,1)
        cover_imgv_rgb = cover_imgv.repeat(1,3,1,1)
        cover_imgv_rgb.detach()
        vgg_loss = mse_loss(vgg(container_img_rgb).relu2_2, vgg(cover_imgv_rgb).relu2_2)        
     
        errH = opt.betamse * mse_loss(container_img, cover_imgv) + opt.betagans * gan_loss + opt.betapix * pixel_loss + opt.betavgg * vgg_loss 



        rev_secret_img = Rnet(container_img)  
        secret_imgv = Variable(secret_img)  
        errR_mse = opt.betamse  * mse_loss(rev_secret_img, secret_imgv)
        

        clean_rev_secret_img_A = Rnet(A_imgv)
        clean_imgv = Variable(clean_img)
        errR_clean_A = opt.betamse * mse_loss(clean_rev_secret_img_A, clean_imgv)
        clean_rev_secret_img_B = Rnet(cover_imgv)
        clean_imgv = Variable(clean_img)
        errR_clean_B = opt.betamse * mse_loss(clean_rev_secret_img_B, clean_imgv)
        errR_clean =opt.betacleanA * errR_clean_A + opt.betacleanB * errR_clean_B  

        half_batchsize = int(this_batch_size / 2)
        errR_consist = opt.betamse *  mse_loss(rev_secret_img[0:half_batchsize, :, : ,:], rev_secret_img[half_batchsize:this_batch_size, : ,: ,:]) 
        
        errR = errR_mse + opt.betacons * errR_consist +opt.betaclean * errR_clean
        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret 



        err_sum.backward()
        optimizerH.step()
        optimizerR.step()


        #  Train Discriminator
        Dnet.zero_grad()
        # Real loss
        pred_real = Dnet(cover_imgv)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = Dnet(container_img.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        errD = 10000 * 0.5 * (loss_real + loss_fake)
        
        errD.backward()
        optimizerD.step()


        Hlosses.update(errH.data, this_batch_size)  
        Rlosses.update(errR.data, this_batch_size) 
        R_mselosses.update(errR_mse.data, this_batch_size) 
        R_consistlosses.update(errR_consist.data, this_batch_size) 

        Dlosses.update(errD.data, this_batch_size)  
        FakeDlosses.update(loss_fake.data, this_batch_size)  
        RealDlosses.update(loss_real.data, this_batch_size)  
        Ganlosses.update(gan_loss.data, this_batch_size)
        Pixellosses.update(pixel_loss.data, this_batch_size) 
        Vgglosses.update(vgg_loss.data, this_batch_size)        
        SumLosses.update(err_sum.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_R_mse: %.4f Loss_R_consist: %.4f Loss_D: %.4f Loss_FakeD: %.4f Loss_RealD: %.4f Loss_Gan: %.4f Loss_Pixel: %.4f Loss_Vgg: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses.val, Rlosses.val, R_mselosses.val, R_consistlosses.val, Dlosses.val, FakeDlosses.val, RealDlosses.val, Ganlosses.val, Pixellosses.val, Vgglosses.val, SumLosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)


        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            diff =  50 * (container_img - cover_imgv)
            save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, 
            	secret_img, rev_secret_img.data, clean_rev_secret_img_A.data, clean_rev_secret_img_B.data, diff.data, epoch, i, opt.trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f     optimizerD_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr'], optimizerD.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_R_mseloss=%.6f\tepoch_R_consistloss=%.6f\tepoch_Dloss=%.6f\tepoch_FakeDloss=%.6f\tepoch_RealDloss=%.6f\tepoch_GanLoss=%.6fepoch_Pixelloss=%.6f\tepoch_Vggloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, R_mselosses.avg, R_consistlosses.avg, Dlosses.avg, FakeDlosses.avg, RealDlosses.avg, Ganlosses.avg, Pixellosses.avg, Vgglosses.avg, SumLosses.avg)


    print_log(epoch_log, logPath)


    writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/D_lr", optimizerD.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)

    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('train/R_consist_loss', R_consistlosses.avg, epoch)    
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/D_loss', Dlosses.avg, epoch)
    writer.add_scalar('train/FakeD_loss', FakeDlosses.avg, epoch) 
    writer.add_scalar('train/RealD_loss', RealDlosses.avg, epoch)   
    writer.add_scalar('train/Gan_loss', Ganlosses.avg, epoch)  
    writer.add_scalar('train/Pixel_loss', Pixellosses.avg, epoch)
    writer.add_scalar('train/Vgg_loss', Vgglosses.avg, epoch)    
    writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader,  epoch, Hnet, Rnet, Dnet):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Dnet.eval()
    Hlosses = AverageMeter()  
    Rlosses = AverageMeter() 
    R_mselosses = AverageMeter() 
    R_consistlosses = AverageMeter()   
    Dlosses = AverageMeter()  
    FakeDlosses = AverageMeter()
    RealDlosses = AverageMeter()
    Ganlosses = AverageMeter()
    Pixellosses = AverageMeter()
    Vgglosses = AverageMeter()    

    # Tensor type
    Tensor = torch.cuda.FloatTensor 
    with torch.no_grad(): 

        loader = transforms.Compose([trans.Grayscale(num_output_channels=1),transforms.ToTensor(),])
        clean_img = Image.open("../secret/clean.png")
        clean_img = loader(clean_img)  
        secret_img = Image.open("../secret/flower.png")
        secret_img = loader(secret_img)  
        
        for i, data in enumerate(val_loader, 0):

            Hnet.zero_grad()
            Rnet.zero_grad()
            Dnet.zero_grad()
            this_batch_size = int(data.size()[0])  
            cover_img = data[0:this_batch_size, :, :, :]  
            cover_img_A = cover_img[ :, :, 0:256, 0:256]
            cover_img_B = cover_img[ :, :, 0:256, 256:512]

            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
            secret_img = secret_img[0:this_batch_size, :, :, :]  
 
            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1) 
            clean_img = clean_img[0:this_batch_size, :, :, :]  

            if opt.cuda:
                cover_img = cover_img.cuda()
                cover_img_A = cover_img_A.cuda()
                cover_img_B = cover_img_B.cuda()             
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()

            concat_img = torch.cat([cover_img_B, secret_img], dim=1)
            concat_imgv = Variable(concat_img)  
            cover_imgv = Variable(cover_img_B)  
            container_img = Hnet(concat_imgv)  
            A_imgv = Variable(cover_img_A) 

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((cover_imgv.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((cover_imgv.size(0), *patch))), requires_grad=False)
            pred_fake = Dnet(container_img)
            gan_loss = criterion_GAN(pred_fake, valid) 

            pixel_loss = criterion_pixelwise(container_img, cover_imgv)
            container_img_rgb = container_img.repeat(1,3,1,1)
            cover_imgv_rgb = cover_imgv.repeat(1,3,1,1)
            cover_imgv_rgb.detach()
            vgg_loss = mse_loss(vgg(container_img_rgb).relu2_2, vgg(cover_imgv_rgb).relu2_2)        
            
            errH = opt.betamse * mse_loss(container_img, cover_imgv) +  opt.betagans * gan_loss + opt.betapix * pixel_loss + opt.betavgg * vgg_loss 

            #  Train Discriminator
            # Real loss
            pred_real = Dnet(cover_imgv)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = Dnet(container_img.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            errD = 10000 * 0.5 * (loss_real + loss_fake)

            rev_secret_img = Rnet(container_img)  
            secret_imgv = Variable(secret_img) 
            errR_mse = opt.betamse * mse_loss(rev_secret_img, secret_imgv)
            
            clean_rev_secret_img_A = Rnet(A_imgv)
            clean_imgv = Variable(clean_img)
            errR_clean_A = opt.betamse * mse_loss(clean_rev_secret_img_A, clean_imgv)
            clean_rev_secret_img_B = Rnet(cover_imgv)
            clean_imgv = Variable(clean_img)
            errR_clean_B = opt.betamse * mse_loss(clean_rev_secret_img_B, clean_imgv)
            errR_clean =opt.betacleanA * errR_clean_A + opt.betacleanB * errR_clean_B 

            half_batchsize = int(this_batch_size / 2)
            errR_consist = opt.betamse *  mse_loss(rev_secret_img[0:half_batchsize, :, : ,:], rev_secret_img[half_batchsize:half_batchsize * 2, : ,: ,:]) 
            
            errR = errR_mse + opt.betacons * errR_consist +opt.betaclean * errR_clean
            betaerrR_secret = opt.beta * errR
            err_sum = errH + betaerrR_secret 


            Hlosses.update(errH.data, this_batch_size)  
            Rlosses.update(errR.data, this_batch_size) 
            R_mselosses.update(errR_mse.data, this_batch_size)
            R_consistlosses.update(errR_consist.data, this_batch_size)
            Dlosses.update(errD.data, this_batch_size)  
            FakeDlosses.update(loss_fake.data, this_batch_size)  
            RealDlosses.update(loss_real.data, this_batch_size)  
            Ganlosses.update(gan_loss.data, this_batch_size) 
            Pixellosses.update(pixel_loss.data, this_batch_size) 
            Vgglosses.update(vgg_loss.data, this_batch_size) 


            if i % 50 == 0:
                diff =  50 * (container_img - cover_imgv)
                save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, 
                	secret_img, rev_secret_img.data, clean_rev_secret_img_A.data,clean_rev_secret_img_B.data, diff.data, epoch, i, opt.validationpics)    

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_r_mseloss = R_mselosses.avg
    val_r_consistloss = R_consistlosses.avg
    val_dloss = Dlosses.avg
    val_fakedloss = FakeDlosses.avg
    val_realdloss = RealDlosses.avg
    val_Ganlosses = Ganlosses.avg
    val_Pixellosses = Pixellosses.avg
    val_Vgglosses = Vgglosses.avg       
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_R_mseloss = %.6f\t val_R_consistloss = %.6f\t val_Dloss = %.6f\t val_FakeDloss = %.6f\t val_RealDloss = %.6f\t val_Ganlosses = %.6f\t val_Pixellosses = %.6f\t val_Vgglosses = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, val_Vgglosses, val_sumloss, val_time)

    print_log(val_log, logPath)


    writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('validation/R_consist_loss', R_consistlosses.avg, epoch)   
    writer.add_scalar('validation/D_loss_avg', Dlosses.avg, epoch)
    writer.add_scalar('validation/FakeD_loss_avg', FakeDlosses.avg, epoch)
    writer.add_scalar('validation/RealD_loss_avg', RealDlosses.avg, epoch)
    writer.add_scalar('validation/Gan_loss_avg', val_Ganlosses, epoch)
    writer.add_scalar('validation/Pixel_loss_avg', val_Pixellosses, epoch)
    writer.add_scalar('validation/Vgg_loss_avg', val_Vgglosses, epoch)    
    writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print("#################################################### validation end ########################################################")

    return val_hloss, val_rloss, val_r_mseloss, val_r_consistloss, val_dloss, val_fakedloss, val_realdloss, val_Ganlosses, val_Pixellosses, vgg_loss, val_sumloss


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, secretLabelv, RevSecImg,RevCleanImgA,RevCleanImgB, diff, epoch, i, save_path):

    originalFramesA = originalLabelvA.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    originalFramesB = originalLabelvB.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    container_allFrames = Container_allImg.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)

    secretFrames = secretLabelv.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    revCleanFramesA = RevCleanImgA.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    revCleanFramesB = RevCleanImgB.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)

    showResult = torch.cat([secretFrames,originalFramesA,revCleanFramesA, originalFramesB,revCleanFramesB, diff, container_allFrames,
            revSecFrames,], 0)
    
    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
