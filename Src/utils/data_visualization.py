from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class TbSummary():
    def __init__(self, output_dir = None):
        if output_dir is not None:
            self.writer = SummaryWriter(output_dir)
        else:
            self.writer = SummaryWriter()

    def show_images(self, imgs, label_text, pred_text, n_iter, initial_title = ''):

        if len(initial_title) > 0:
            initial_title = initial_title + ' '
        imgs_to_show = imgs[:10,...]
        label_text_to_show = label_text[:10]
        pred_text_to_show = pred_text[:10]
        x = vutils.make_grid(imgs_to_show, nrow=1, normalize=True, scale_each=True)
        self.writer.add_image(initial_title + 'Image', x, n_iter)
        self.writer.add_text(initial_title + 'Label Text', "\n".join(label_text_to_show), n_iter)
        self.writer.add_text(initial_title + 'Pred Text', "\n".join(pred_text_to_show), n_iter)


    def get_writer(self):
        return self.writer
