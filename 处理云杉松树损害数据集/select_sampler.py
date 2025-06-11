import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QCheckBox, QLineEdit, QPushButton,
                           QScrollArea, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
import cv2
import shutil

class ImageSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_index = 0
        self.selected_images = set()
        self.image_data = []
        
        # 加载数据
        self.train_dir = "../data/ForestDamages/train2024"
        self.train_anno = "../data/ForestDamages/annotations/instances_train2024.json"
        self.load_data()
        
        self.initUI()
        
    def load_data(self):
        with open(self.train_anno, 'r') as f:
            self.anno_data = json.load(f)
            
        # 构建图片ID到标注的映射
        self.id_to_annos = {}
        for ann in self.anno_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_annos:
                self.id_to_annos[img_id] = []
            self.id_to_annos[img_id].append(ann)
            
        # 构建图片信息列表
        for img in self.anno_data['images']:
            self.image_data.append({
                'id': img['id'],
                'file_name': img['file_name'],
                'annotations': self.id_to_annos.get(img['id'], [])
            })
            
    def initUI(self):
        self.setWindowTitle('森林损伤图片选择器')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # 左侧图片显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # 图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('上一张')
        self.next_button = QPushButton('下一张')
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)
        
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=7)
        
        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # 场景类型选择
        self.scene_types = {
            'dense': '密集场景',
            'sparse': '稀疏场景',
            'single': '单体场景',
            'complex': '复杂场景'
        }
        
        self.checkboxes = {}
        for key, value in self.scene_types.items():
            self.checkboxes[key] = QCheckBox(value)
            right_layout.addWidget(self.checkboxes[key])
            
        # 自定义输出文件夹
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel('输出文件夹:'))
        self.folder_name = QLineEdit('selected_images')
        folder_layout.addWidget(self.folder_name)
        right_layout.addLayout(folder_layout)
        
        # 保存按钮
        self.save_button = QPushButton('保存选中图片')
        self.save_button.clicked.connect(self.save_selected_images)
        right_layout.addWidget(self.save_button)
        
        # 图片信息显示
        self.info_label = QLabel()
        right_layout.addWidget(self.info_label)
        
        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget, stretch=3)
        
        main_widget.setLayout(layout)
        
        # 显示第一张图片
        self.show_current_image()
        
    def show_current_image(self):
        if not self.image_data:
            return
            
        current_data = self.image_data[self.current_index]
        image_path = os.path.join(self.train_dir, current_data['file_name'])
        
        # 读取并显示图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制标注框
        for ann in current_data['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # 转换为QImage并显示
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 调整图片大小以适应窗口
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        
        # 更新信息显示
        self.info_label.setText(f'图片 {self.current_index + 1}/{len(self.image_data)}\n'
                              f'文件名: {current_data["file_name"]}\n'
                              f'标注框数量: {len(current_data["annotations"])}')
                              
    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            
    def show_next_image(self):
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            self.show_current_image()
            
    def save_selected_images(self):
        # 获取选中的场景类型
        selected_types = []
        for key, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                selected_types.append(self.scene_types[key])
                
        if not selected_types:
            QMessageBox.warning(self, '警告', '请至少选择一个场景类型！')
            return
            
        # 创建输出文件夹
        output_dir = self.folder_name.text()
        if not output_dir:
            QMessageBox.warning(self, '警告', '请输入输出文件夹名称！')
            return
            
        # 创建输出文件夹结构
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
        
        # 复制当前图片和对应的标注
        current_data = self.image_data[self.current_index]
        src_image = os.path.join(self.train_dir, current_data['file_name'])
        dst_image = os.path.join(output_dir, 'images', current_data['file_name'])
        shutil.copy2(src_image, dst_image)
        
        # 创建新的标注文件
        selected_anno = {
            'images': [img for img in self.anno_data['images'] if img['id'] == current_data['id']],
            'annotations': current_data['annotations'],
            'categories': self.anno_data['categories'],
            'scene_types': selected_types
        }
        
        # 保存标注文件
        anno_filename = f"instance_{os.path.splitext(current_data['file_name'])[0]}.json"
        with open(os.path.join(output_dir, 'annotations', anno_filename), 'w') as f:
            json.dump(selected_anno, f, indent=4)
            
        QMessageBox.information(self, '成功', f'已将图片保存到 {output_dir} 文件夹！')

def main():
    app = QApplication(sys.argv)
    ex = ImageSelector()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()