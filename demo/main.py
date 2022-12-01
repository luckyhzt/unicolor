import os, sys

sys.path.append('./sample')
from PIL import Image
import numpy as np
from io import BytesIO
import win32clipboard

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPen, QIcon, QImage, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QColorDialog, QRubberBand, QListView, QListWidgetItem, QShortcut

from ui import Ui_main
from stroke import Stroke, RectRegion

import cv2
import qimage2ndarray

from utils_func import *
from sample_func import *
from colorizer import Colorizer
from thread import SampleThread, LoadThread
from colorpicker.colorpicker import ColorPicker



class MainWindow(QMainWindow, Ui_main.Ui_MainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.current_path, os.pardir, 'finals')
        self.device = 'cuda:0'
        self.drag_flag = False
        self.save_path = os.path.abspath('./demo/data')
        self.img_size = [256, 256]
        self.colorizer = None
        self.default_img_path = 'C:\\Users\\lucky\\Desktop\\old_photo\\images2'
        self.default_exp_path = 'C:\\MyFiles\\Dataset\\coco\\val2017'
        self.setupUi(self)
        self.init_components()
        # Coco model
        self.load_filltran('C:/MyFiles/CondTran/finals/bert_coco/logs/bert/epoch=144-step=259999.ckpt')
        # Imagenet model
        #self.load_filltran('C:/MyFiles/CondTran/finals/bert_final/logs/bert/epoch=14-step=142124.ckpt')

    def mousePressEvent(self, e):
        self.origin = QPoint(e.pos())
        in_canvas = self.inwhich_canvas(self.origin.x(), self.origin.y())
        if in_canvas:
            x, y = self.get_canvas_coordinate(in_canvas, self.origin.x(), self.origin.y())
            self.in_canvas = in_canvas
            if e.button() == Qt.LeftButton:
                self.drag_flag = 'stroke'
                self.line_points = []
                self.line_points.append([y, x])
            elif e.button() == Qt.RightButton and self.in_canvas == 'output':
                self.drag_flag = 'select'
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()

    def mouseReleaseEvent(self, e):
        if self.drag_flag == 'select':
            x0, y0 = self.get_canvas_coordinate(self.in_canvas, self.origin.x(), self.origin.y())
            x1, y1 = self.get_canvas_coordinate(self.in_canvas, e.x(), e.y())
            self.output_regions.add([x0, y0, x1, y1])
            self.rubberBand.hide()
            self.draw_sample_regions()
            self.edit_output.setChecked(True)
        if self.drag_flag == 'stroke':
            self.stroke_mode.setChecked(True)
            if self.in_canvas == 'input':
                self.input_strokes.add(self.line_points, color=list(self.colorpicker.color))
                self.edit_input.setChecked(True)
            elif self.in_canvas == 'output':
                self.output_strokes.add(self.line_points, color=list(self.colorpicker.color))
                self.edit_output.setChecked(True)
        self.drag_flag = None

    def mouseMoveEvent(self, e):
        if self.drag_flag == 'stroke':
            x0, y0 = self.get_canvas_coordinate(self.in_canvas, self.origin.x(), self.origin.y())
            x1, y1 = self.get_canvas_coordinate(self.in_canvas, e.x(), e.y())
            painter = self.input_painter if self.in_canvas == 'input' else self.output_painter
            r, g, b = self.colorpicker.color
            painter.setPen( QPen(QColor(r, g, b), 10) )
            painter.drawLine(x0, y0, x1, y1)
            self.line_points.append([y1, x1])
            self.origin = QPoint(e.pos())
        
        elif self.drag_flag == 'select':
            self.rubberBand.setGeometry(QRect(self.origin, e.pos()).normalized())

        self.update()

    def draw_sample_regions(self):
        pw, ph = self.output_canvas.pixmap().width(), self.output_canvas.pixmap().height()
        original = np.array(self.output_image.resize([pw, ph]))
        highlight = 0.5 * original.copy()
        for c0, r0, c1, r1 in self.output_regions.rects:
            r0 = int(r0 / 16 * ph)
            c0 = int(c0 / 16 * pw)
            r1 = int(r1 / 16 * ph)
            c1 = int(c1 / 16 * pw)
            highlight[r0:r1, c0:c1, :] = cv2.cvtColor( cv2.cvtColor(original[r0:r1, c0:c1, :], cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB )
            self.display_output(highlight)

    def inwhich_canvas(self, x, y):
        for window in ['input', 'output']:
            sx, sy = self.get_canvas_coordinate(window, x, y, return_type='scale')
            if sx >= 0 and sx < 1 and sy >= 0 and sy < 1:
                return window
        return False

    def get_canvas_coordinate(self, window, x, y, return_type='clip'):
        canvas = self.input_canvas if window == 'input' else self.output_canvas
        if canvas.pixmap() == None:
            return -1, -1
        group = self.input_group if window == 'input' else self.output_group
        dx, dy = canvas.x()+group.x(), canvas.y()+group.y()
        cw, ch = canvas.width(), canvas.height()
        pw, ph = canvas.pixmap().width(), canvas.pixmap().height()
        p_ratio = pw / ph
        if p_ratio >= 1:
            dy += int( (ch - cw / p_ratio) / 2 )
        else:
            dx += int( (cw - ch * p_ratio) / 2 )
        x -= dx
        y -= dy
        if return_type == 'clip':
            x = np.clip(x, 0, pw-1)
            y = np.clip(y, 0, ph-1)
        elif return_type == 'scale':
            x /= pw
            y /= ph
        return x, y

    def init_components(self):
        self.info.setWordWrap(True)
        self.sampling_bar.hide()
        self.sample_btn.clicked.connect(self.sample)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.diverse_list.setSpacing(10)

        self.input_load_btn.clicked.connect(self.load_image)
        self.output_copy_btn.clicked.connect(self.copy_output)
        self.load_exp_btn.clicked.connect(self.load_exemplar)
        self.input_clear_btn.clicked.connect(self.clear_input)
        self.output_clear_btn.clicked.connect(self.clear_output)
        self.text_input.textChanged.connect(lambda: self.text_mode.setChecked(True))
        self.colorpicker = ColorPicker(self, rgb=(200,100,100))
        self.colorpicker.setGeometry(10, 542, 221, 209)
        self.diverse_list.itemDoubleClicked.connect(self.edit_result)

        self.update()

    def load_image(self):
        file_choose = QFileDialog.getOpenFileName(self, 'Choose image file', self.default_img_path)
        img_path = file_choose[0]
        if img_path != '':
            self.colorpicker.setRGB((200, 100, 100))
            self.default_img_path = os.path.join(img_path, os.pardir)
            output_image = Image.open(img_path).convert('RGB')
            self.output_image = limit_size(output_image, minsize=256, maxsize=1024)
            self.input_image = self.output_image.convert('L').convert('RGB')
            self.clear_input()
            self.clear_output()
            self.diverse_list.clear()
            self.results = []
            self.send_result(self.output_image)

    def copy_output(self):
        output = BytesIO()
        self.output_image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:]
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        
    def display_input(self, image):
        if hasattr(self, 'input_painter'):
            self.input_painter.end()
        qimage = qimage2ndarray.array2qimage(np.array(image))
        pix = QPixmap(qimage)
        if pix.width() >= pix.height():
            pix = pix.scaledToWidth(self.input_canvas.width())
        else:
            pix = pix.scaledToHeight(self.input_canvas.height())
        self.input_canvas.setPixmap(pix)
        pix = self.input_canvas.pixmap()
        self.input_painter = QPainter(pix)
    
    def display_output(self, image):
        if hasattr(self, 'output_painter'):
            self.output_painter.end()
        qimage = qimage2ndarray.array2qimage(np.array(image))
        pix = QPixmap(qimage)
        if pix.width() >= pix.height():
            pix = pix.scaledToWidth(self.output_canvas.width())
        else:
            pix = pix.scaledToHeight(self.output_canvas.height())
        self.output_canvas.setPixmap(pix)
        pix = self.output_canvas.pixmap()
        self.output_painter = QPainter(pix)
    
    def clear_input(self):
        if hasattr(self, 'input_image'):
            self.display_input(self.input_image)
            pix = self.input_canvas.pixmap()
            self.input_strokes = Stroke(img_size=[pix.height(), pix.width()])
            self.edit_input.setChecked(True)
    
    def clear_output(self):
        if hasattr(self, 'output_image'):
            self.display_output(self.output_image)
            pix = self.output_canvas.pixmap()
            self.output_strokes = Stroke(img_size=[pix.height(), pix.width()])
            self.output_regions = RectRegion(img_size=[pix.height(), pix.width()])
            self.edit_input.setChecked(True)
        
    def load_exemplar(self):
        file_choose = QFileDialog.getOpenFileName(self, 'Choose image file', self.default_exp_path)
        img_path = file_choose[0]
        if img_path != '':
            self.default_exp_path = os.path.join(img_path, os.pardir)
            self.exemplar_image = Image.open(img_path).convert('RGB')
            self.exemplar_image = limit_size(self.exemplar_image, minsize=256, maxsize=1024)
            self.exemplar_display.setPixmap(QPixmap(img_path))
            self.exemplar_display.setScaledContents(True)
            self.exemplar_mode.setChecked(True)
    
    def load_filltran(self, path):
        if path == '':
            path = QFileDialog.getOpenFileName(self, 'Choose model checkpoint file', self.model_path)
            path = path[0]
        if path != '':
            self.loader = LoadThread(self, path)
            self.loader.message.connect(self.message)
            self.loader.start()
      
    def message(self, text, color='black'):
        self.info.setText(f"<font color='{color}'>{text}</font>")
        self.info.repaint()
    
    def progress(self, percentage):
        if percentage > 0:
            self.sampling_bar.show()
            self.sampling_bar.setValue(percentage)
        else:
            self.sampling_bar.hide()
    
    def sample(self):
        edit_mode, cond_mode = self.get_mode()
        if self.colorizer == None:
            self.message('Model is not loaded!', 'red')
            return
        if not hasattr(self, 'input_image'):
            self.message('Image is not loaded!', 'red')
            return
        if 'text' in cond_mode and self.text_input.toPlainText() == '':
            self.message('Text prompt is not input!', 'red')
            return
        if 'exemplar' in cond_mode and not hasattr(self, 'exemplar_image'):
            self.message('Exemplar image is not loaded!', 'red')
            return
        self.sample_btn.setEnabled(False)
        # Sample
        self.sampler = SampleThread(self, edit_mode, cond_mode, int(self.topk.value()), int(self.sample_times.value()))
        self.sampler.message.connect(self.message)
        self.sampler.progress.connect(self.progress)
        self.sampler.result.connect(self.send_result)
        self.sampler.start()

    def send_result(self, image, show=False):
        self.results.append(image.copy())
        # Show in diverse list
        self.diverse_list.setSpacing(1)
        w = self.diverse_list.width() - 20
        h = int(self.diverse_list.height() / 4.3)
        item = QListWidgetItem(self.diverse_list)
        item.setSizeHint(QSize(w, h))
        imageWidget = ImageWidget(image, w, h, item)
        self.diverse_list.addItem(item)
        self.diverse_list.setItemWidget(item, imageWidget)

        self.update()
        if show:
            self.output_image = self.results[-1]
            self.clear_output()

    def edit_result(self):
        item = self.diverse_list.currentItem()
        if item != None:
            ind = self.diverse_list.row(item)
            self.output_image = self.results[ind]
            self.clear_output()
 
    def get_mode(self):
        cond_mode = []
        if self.stroke_mode.isChecked():
            cond_mode.append('stroke')
        if self.text_mode.isChecked():
            cond_mode.append('text')
        if self.exemplar_mode.isChecked():
            cond_mode.append('exemplar')

        if self.edit_input.isChecked():
            edit_mode = 'input'
        elif self.edit_output.isChecked():
            edit_mode = 'output'
        return edit_mode, cond_mode

    def get_pixmap_image(self, source):
        pixmap = self.input_canvas.pixmap() if source == 'input' else self.output_canvas.pixmap()
        ## Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()
        ## Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        byte_str = qimg.bits()
        byte_str.setsize(h*w*4)
        ## Using the np.frombuffer function to convert the byte string into an np array
        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize(self.input_image.size)
        return img


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, image, w, h, item):
        super(ImageWidget, self).__init__(None)
        self.item = item
        #self.textQVBoxLayout = QtWidgets.QVBoxLayout()
        #self.editButton = QtWidgets.QPushButton()
        #self.deleteButton = QtWidgets.QPushButton()
        
        #btn_size = QSize(25, 25)
        #icon_size = QSize(25, 25)
        #self.editButton.setIcon(self.getIcon('demo\\resources\\edit.png'))
        #self.deleteButton.setIcon(self.getIcon('demo\\resources\\delete.jpg'))
        #self.editButton.setIconSize(icon_size)
        #self.deleteButton.setIconSize(icon_size)
        #self.editButton.setFixedSize(btn_size)
        #self.deleteButton.setFixedSize(btn_size)

        #self.textQVBoxLayout.addWidget(self.editButton)
        #self.textQVBoxLayout.addWidget(self.deleteButton)
        self.allQHBoxLayout = QtWidgets.QHBoxLayout()
        self.iconQLabel = QtWidgets.QLabel()
        qimage = qimage2ndarray.array2qimage(np.array(image.resize([w, h-10])))
        pix = QPixmap(qimage).scaled(QSize(w, h-10))
        self.iconQLabel.setPixmap(pix)
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        #self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
        #self.allQHBoxLayout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.allQHBoxLayout)

        self.resize(w, h)

    def getIcon(self, path):
        pix = QPixmap(path)
        new_pix = QPixmap(pix.size())
        new_pix.fill(QtCore.Qt.transparent)
        painter = QPainter(new_pix)
        painter.setOpacity(0.6)
        painter.drawPixmap(QPoint(), pix)
        painter.end()
        icon = QIcon(pix)
        return icon


if __name__ == '__main__':
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())