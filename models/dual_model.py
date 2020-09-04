import torch
import models.networks as networks
import util.util as util
from torch.autograd import Variable
import numpy as np
#from opensimplex import OpenSimplex

class DualModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD1, self.netD2, self.netE = self.initialize_networks(opt)


        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
                self.criterionVGGsurface = networks.VGG_surfaceLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
                self.criterionContent = networks.ContentLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, surface_image, color_image, input_image = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated_surface, generated_color = self.compute_generator_loss(
                input_semantics, surface_image, color_image, input_image)
            return g_loss, generated_surface, generated_color
        elif mode == 'surface_discriminator':
            d1_loss = self.compute_surface_discriminator_loss(
                input_semantics, surface_image, input_image)
            return d1_loss
        elif mode == 'color_discriminator':
            d2_loss = self.compute_color_discriminator_loss(
                input_semantics, color_image, input_image)
            return d2_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(input_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_surface_image, fake_color_image, _ = self.generate_fake(input_semantics, surface_image, input_image)
            return fake_surface_image, fake_color_image

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D1_params = list(self.netD1.parameters())
            D2_params = list(self.netD2.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr / 4

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D1 = torch.optim.Adam(D1_params, lr=D_lr, betas=(beta1, beta2))
        optimizer_D2 = torch.optim.Adam(D2_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D1, optimizer_D2

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD1, 'D1', epoch, self.opt)
        util.save_network(self.netD2, 'D2', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        opt.input_nc = 1
        netD1 = networks.define_D(opt) if opt.isTrain else None
        opt.input_nc = 3 + opt.label_nc 
        netD2 = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD1 = util.load_network(netD1, 'D1', opt.which_epoch, opt)
                netD2 = util.load_network(netD2, 'D2', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD1, netD2, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    '''
    def gen_perlin(self,w,h):
        fct = OpenSimplex(np.random.randint(1000000))
        heights = np.zeros((w,h))
        for i in range(w):
            for j in range(h):
                heights[i,j] = (fct.noise2d(i,j)+1)/2
        return heights
    '''
    def preprocess_input(self, data):
        # move to GPU and change data types

        #print(data['image'] - data['input'])
        #data['label'] = data['label'].long()

        #Gaussian noise
        #mean = 0.5; stddev = 0.1;
        #noise = Variable(data['label'].data.new(data['label'].size()).normal_(mean,stddev)).cuda()

        #Perlin noise
        #h,w = data['label'].size()[2], data['label'].size()[3]
        #noise = self.gen_perlin(w,h)
        #noise_tensor = torch.FloatTensor(noise).cuda()

        data['label'] = data['label'].long()

        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['surface'] = data['surface'].cuda()
            data['color'] = data['color'].cuda()
            data['input'] = data['input'].cuda()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        #input_semantics = (0.5 + noise_tensor/2) * input_semantics
        #print(input_semantics)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
        
        #print(data['surface'].shape)
        return input_semantics, data['surface'], data['color'], data['input']

    def compute_generator_loss(self, input_semantics, surface_image, color_image, input_image):
        G_losses = {}

        fake_surface_image, fake_color_image, KLD_loss = self.generate_fake(
            input_semantics, color_image, input_image, compute_kld_loss=self.opt.use_vae)



        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
            #G_losses['Content'] = self.criterionContent(fake_image, input_image)

        pred_fake_surface, pred_real_surface = self.surface_discriminate(
            input_semantics, fake_surface_image, surface_image, input_image)

        pred_fake_color, pred_real_color = self.color_discriminate(
                input_semantics, fake_color_image, color_image, input_image)        #Surface generator loss
        G_losses['GAN_surface'] = self.criterionGAN(pred_fake_surface, True,
                                            for_discriminator=False)
        #G_losses['surface_L1'] = self.criterionFeat(fake_surface_image, surface_image) * 10.0          #Include it in a variable
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake_surface)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake_surface[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake_surface[i][j], pred_real_surface[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat_surface'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG_surface'] = self.criterionVGGsurface(fake_surface_image, surface_image) \
                * self.opt.lambda_vgg


        #Color generator loss
        G_losses['GAN_color'] = self.criterionGAN(pred_fake_color, True, 
                for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake_color)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake_color[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake_color[i][j], pred_real_color[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat_color'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG_color'] = self.criterionVGG(fake_color_image, color_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_surface_image, fake_color_image

    def compute_surface_discriminator_loss(self, input_semantics, surface_image, input_image):
        D_losses = {}
        with torch.no_grad():
            fake_surface_image,_,_ = self.generate_fake(input_semantics, surface_image, input_image)
            fake_surface_image = fake_surface_image.detach()
            fake_surface_image.requires_grad_()

        pred_fake, pred_real = self.surface_discriminate(
            input_semantics, fake_surface_image, surface_image, input_image)

        D_losses['D1_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D1_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        return D_losses

    def compute_color_discriminator_loss(self, input_semantics, color_image, input_image):
        D_losses = {}
        with torch.no_grad():
            _, fake_color_image, _ = self.generate_fake(input_semantics, color_image, input_image)
            fake_color_image = fake_color_image.detach()
            fake_color_image.requires_grad_()

        pred_fake, pred_real = self.color_discriminate(
            input_semantics, fake_color_image, color_image, input_image)

        D_losses['D2_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D2_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, input_image):
        mu, logvar = self.netE(input_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, input_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(input_image)
            #print(z.shape)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld


        h, w = self.opt.crop_size, self.opt.crop_size
        #noise = self.gen_perlin(w , h)
        #noise_tensor = torch.FloatTensor(noise).cuda()

        #input_semantics = (0.5 + noise_tensor/2) * input_semantics
        #print(input_image.shape) 
        fake_surface_image, fake_color_image = self.netG(input_semantics, input_image, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_surface_image, fake_color_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def surface_discriminate(self, input_semantics, fake_image, real_image, input_image):

        fake_concat = torch.cat([fake_image], dim=1)
        real_concat = torch.cat([real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD1(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def color_discriminate(self, input_semantics, fake_image, real_image, input_image):
        
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD2(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real


    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.ByteTensor).cuda()
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).type(torch.ByteTensor).cuda()
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.ByteTensor).cuda()
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).type(torch.ByteTensor).cuda()
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


