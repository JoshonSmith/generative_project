import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
from os.path import splitext, join
import ntpath
from .util import tensor2im, save_image

class HTML:
    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, text):
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

    def save_images(self, visuals, image_path, aspect_ratio=1.0, width=256):
        image_dir = self.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = splitext(short_path)[0]

        self.add_header(name)
        ims, txts, links = [], [], []

        for label, im_data in visuals.items():
            im = tensor2im(im_data)
            image_name = '{}/{}.png'.format(label, name)
            os.makedirs(join(image_dir, label), exist_ok=True)
            save_path = join(image_dir, image_name)
            save_image(im, save_path, aspect_ratio=aspect_ratio)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        self.add_images(ims, txts, links, width=width)

