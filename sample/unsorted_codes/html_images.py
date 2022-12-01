import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
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
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
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


def display_sample_text(path, n):
    html = HTML(path, 'sample_text')
    html.add_header('Sample images with text prior')

    for i in range(2):
        ims, txts, links = [], [], []
        
        ims.append('original.jpg')
        txts.append('Original image, caption: A little girl holds up a big purple umbrella')
        links.append('original.jpg')

        ims.append('gray.jpg')
        txts.append('Gray image')
        links.append('gray.png')

        ims.append('prior.jpg')
        txts.append('Generated prior')
        links.append('prior.jpg')

        ims.append('seg_prior.jpg')
        txts.append('Segmented prior')
        links.append('seg_prior.jpg')

        ims.append('generated.jpg')
        txts.append('Generated image')
        links.append('generated.jpg')

        html.add_images(ims, txts, links)

    html.save()



if __name__ == '__main__':  # we show an example usage here.
    html = HTML('sample/sample_text/', 'sample_text')
    html.add_header('Sample images with text prior')

    for i in range(2):
        ims, txts, links = [], [], []
        
        ims.append('original.jpg')
        txts.append('Original image, caption: A little girl holds up a big purple umbrella')
        links.append('original.jpg')

        ims.append('gray.jpg')
        txts.append('Gray image')
        links.append('gray.png')

        ims.append('prior.jpg')
        txts.append('Generated prior')
        links.append('prior.jpg')

        ims.append('seg_prior.jpg')
        txts.append('Segmented prior')
        links.append('seg_prior.jpg')

        ims.append('generated.jpg')
        txts.append('Generated image')
        links.append('generated.jpg')

        html.add_images(ims, txts, links)

    html.save()