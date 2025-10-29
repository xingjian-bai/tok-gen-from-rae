import torch
import torch.nn.functional as F



class LPIPSWithDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.perceptual_weight = None
        self.perceptual_loss = LPIPS().eval()


    def calculate_adaptive_weight(self, recon_loss, gan_loss, last_layer, max_d_weight: float = 1e4,):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, max_d_weight).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    



    def forward(self, imgs, recon, last_layer, use_lpips):
        recon_normed = recon * 2.0 - 1.0
        rec_loss = F.l1_loss(recon, imgs)

        if use_lpips:
            lpips_loss = self.perceptual_loss(imgs.contiguous(), recon.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * lpips_loss
        else:
            lpips_loss = torch.tensor([0.0])


        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.mean().detach(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/discriminator_weight".format(split): self.discriminator_weight
                   }
            return loss, log, batch_rec_loss

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                #    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   }
            return d_loss, log, batch_rec_loss
        
