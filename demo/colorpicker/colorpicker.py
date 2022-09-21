# ------------------------------------- #
#                                       #
# Modern Color Picker by Tom F.         #
# Version 1.3                           #
# made with Qt Creator & PyQt5          #
#                                       #
# ------------------------------------- #

import sys
import colorsys

from PyQt5.QtCore import (QPoint, Qt, pyqtSignal)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QColorDialog)


from .Ui_main import Ui_Form as Ui_Main


class ColorPicker(QWidget):

    colorChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):

        # Extract Initial Color out of kwargs
        rgb = kwargs.pop("rgb", None)
        hsv = kwargs.pop("hsv", None)
        hex = kwargs.pop("hex", None)

        super(ColorPicker, self).__init__(*args, **kwargs)

        # Call UI Builder function
        self.ui = Ui_Main()
        self.ui.setupUi(self)

        # Connect update functions
        self.ui.hue_frame.mouseMoveEvent = self.moveHueSelector
        self.ui.hue_frame.mousePressEvent = self.moveHueSelector

        # Connect selector moving function
        self.ui.black_overlay.mouseMoveEvent = self.moveSVSelector
        self.ui.black_overlay.mousePressEvent = self.moveSVSelector

        # Connect button
        self.ui.color_vis.clicked.connect(self.select_color)

        if rgb:
            self.setRGB(rgb)
        else:
            self.setRGB((0,0,0))


    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b = color.red(), color.green(), color.blue()
            self.setRGB((r, g, b))

    ## Update Functions ##
    def hsvChanged(self):
        h,s,v = (100 - self.ui.hue_selector.y() / 1.60, (self.ui.selector.x() + 6) / 1.5, (144 - self.ui.selector.y()) / 1.5)
        r,g,b = self.hsv2rgb(h,s,v)
        self.color = (r, g, b)
        self.ui.color_vis.setStyleSheet(f"background-color: rgb({r},{g},{b})")
        self.ui.color_view.setStyleSheet(f"border-radius: 5px;background-color: qlineargradient(x1:1, x2:0, stop:0 hsv({h*1.4}%,100%,50%), stop:1 #fff);")
        self.colorChanged.emit()

    ## external setting functions ##
    def setRGB(self, c):
        self.color = c
        r, g, b = c
        hsv = self.rgb2hsv(r, g, b)
        self.ui.hue_selector.move(7, (100 - hsv[0]) * 1.60)
        self.ui.color_view.setStyleSheet(f"border-radius: 5px;background-color: qlineargradient(x1:1, x2:0, stop:0 hsl({hsv[0]*1.4}%,100%,50%), stop:1 #fff);")
        self.ui.selector.move(hsv[1] * 1.5 - 6, (144 - hsv[2] * 1.5))
        self.ui.color_vis.setStyleSheet(f"background-color: rgb({r},{g},{b})")
        self.colorChanged.emit()


    ## Color Utility ##
    def hsv2rgb(self, h_or_color, s = 0, v = 0):
        if type(h_or_color).__name__ == "tuple": h,s,v = h_or_color
        else: h = h_or_color
        r,g,b = colorsys.hsv_to_rgb(h / 100.0, s / 100.0, v / 100.0)
        return self.clampRGB((r * 255, g * 255, b * 255))

    def rgb2hsv(self, r_or_color, g = 0, b = 0):
        if type(r_or_color).__name__ == "tuple": r,g,b = r_or_color
        else: r = r_or_color
        h,s,v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return (h * 100, s * 100, v * 100)

    def hex2rgb(self, hex):
        if len(hex) < 6: hex += "0"*(6-len(hex))
        elif len(hex) > 6: hex = hex[0:6]
        rgb = tuple(int(hex[i:i+2], 16) for i in (0,2,4))
        return rgb

    def rgb2hex(self, r_or_color, g = 0, b = 0):
        if type(r_or_color).__name__ == "tuple": r,g,b = r_or_color
        else: r = r_or_color
        hex = '%02x%02x%02x' % (int(r),int(g),int(b))
        return hex

    def hex2hsv(self, hex):
        return self.rgb2hsv(self.hex2rgb(hex))

    def hsv2hex(self, h_or_color, s = 0, v = 0):
        if type(h_or_color).__name__ == "tuple": h,s,v = h_or_color
        else: h = h_or_color
        return self.rgb2hex(self.hsv2rgb(h,s,v))


    # selector move function
    def moveSVSelector(self, event):
        if event.buttons() == Qt.LeftButton:
            pos = event.pos()
            if pos.x() < 0: pos.setX(0)
            if pos.y() < 0: pos.setY(0)
            if pos.x() > 150: pos.setX(150)
            if pos.y() > 150: pos.setY(150)
            self.ui.selector.move(pos - QPoint(6,6))
            self.hsvChanged()

    def moveHueSelector(self, event):
        if event.buttons() == Qt.LeftButton:
            pos = event.pos().y() - 6
            if pos < 0: pos = 0
            if pos > 160: pos = 160
            self.ui.hue_selector.move(QPoint(7,pos))
            self.hsvChanged()

    def i(self, text):
        try: return int(text)
        except: return 0

    def clampRGB(self, rgb):
        r,g,b = rgb
        if r<0.0001: r=0
        if g<0.0001: g=0
        if b<0.0001: b=0
        if r>255: r=255
        if g>255: g=255
        if b>255: b=255
        return (r,g,b)
