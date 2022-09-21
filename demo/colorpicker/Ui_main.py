# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\MyFiles\CondTran\demo\colorpicker\main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(221, 209)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 221, 209))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.hue_frame = QtWidgets.QFrame(self.groupBox)
        self.hue_frame.setGeometry(QtCore.QRect(177, 31, 40, 172))
        self.hue_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.hue_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.hue_frame.setObjectName("hue_frame")
        self.hue = QtWidgets.QFrame(self.hue_frame)
        self.hue.setGeometry(QtCore.QRect(7, 6, 26, 160))
        self.hue.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.hue.setFrameShadow(QtWidgets.QFrame.Raised)
        self.hue.setObjectName("hue")
        self.hue_bg = QtWidgets.QFrame(self.hue_frame)
        self.hue_bg.setGeometry(QtCore.QRect(7, 6, 26, 160))
        self.hue_bg.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 rgba(255, 0, 0, 255), stop:0.166 rgba(255, 255, 0, 255), stop:0.333 rgba(0, 255, 0, 255), stop:0.5 rgba(0, 255, 255, 255), stop:0.666 rgba(0, 0, 255, 255), stop:0.833 rgba(255, 0, 255, 255), stop:1 rgba(255, 0, 0, 255));\n"
"border-radius: 5px;")
        self.hue_bg.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.hue_bg.setFrameShadow(QtWidgets.QFrame.Raised)
        self.hue_bg.setObjectName("hue_bg")
        self.hue_selector = QtWidgets.QLabel(self.hue_frame)
        self.hue_selector.setGeometry(QtCore.QRect(6, 160, 26, 12))
        self.hue_selector.setStyleSheet("background-color: #222;\n"
"border-radius: 5px;")
        self.hue_selector.setText("")
        self.hue_selector.setObjectName("hue_selector")
        self.color_view = QtWidgets.QFrame(self.groupBox)
        self.color_view.setGeometry(QtCore.QRect(16, 54, 150, 150))
        self.color_view.setStyleSheet("/* ALL CHANGES HERE WILL BE OVERWRITTEN */;\n"
"background-color: qlineargradient(x1:1, x2:0, stop:0 hsl(0.5%,100%,50%), stop:1 rgba(255, 255, 255, 255));\n"
"\n"
"")
        self.color_view.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.color_view.setFrameShadow(QtWidgets.QFrame.Raised)
        self.color_view.setObjectName("color_view")
        self.black_overlay = QtWidgets.QFrame(self.color_view)
        self.black_overlay.setGeometry(QtCore.QRect(0, 0, 150, 150))
        self.black_overlay.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0, 0, 0, 0), stop:1 rgba(0, 0, 0, 255));\n"
"border-radius: 4px;\n"
"\n"
"")
        self.black_overlay.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.black_overlay.setFrameShadow(QtWidgets.QFrame.Raised)
        self.black_overlay.setObjectName("black_overlay")
        self.selector = QtWidgets.QFrame(self.black_overlay)
        self.selector.setGeometry(QtCore.QRect(-6, 144, 12, 12))
        self.selector.setStyleSheet("background-color:none;\n"
"border: 1px solid white;\n"
"border-radius: 5px;")
        self.selector.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.selector.setFrameShadow(QtWidgets.QFrame.Raised)
        self.selector.setObjectName("selector")
        self.black_ring = QtWidgets.QLabel(self.selector)
        self.black_ring.setGeometry(QtCore.QRect(1, 1, 10, 10))
        self.black_ring.setStyleSheet("background-color: none;\n"
"border: 1px solid black;\n"
"border-radius: 5px;")
        self.black_ring.setObjectName("black_ring")
        self.color_vis = QtWidgets.QPushButton(self.groupBox)
        self.color_vis.setGeometry(QtCore.QRect(120, 25, 41, 21))
        self.color_vis.setText("")
        self.color_vis.setObjectName("color_vis")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Stroke Color"))
        self.black_ring.setText(_translate("Form", "TextLabel"))

