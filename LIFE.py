import os
import pandas as pd
import cv2
import numpy as np
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 데이터 로딩
data_directory = 'C:/Users/home/Desktop/Sheet'
annotations_path = os.path.join(data_directory, 'annotations')
images_path = os.path.join(data_directory, 'images')

# CSV 파일 생성
labels_list = []
for filename in os.listdir(annotations_path):
    labels_list.append({'filename': filename.replace('.xml', '.png'), 'class': 'safe'})

labels = pd.concat([pd.DataFrame(labels_list)], ignore_index=True)

# 2. 데이터 전처리
X = []  # 이미지를 저장할 리스트
y = labels['class']

# 이미지 불러와서 특성 추출
for filename in labels['filename']:
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # 이미지 크기를 조절
    features = np.ravel(image)  # 이미지를 1차원 배열로 변환
    X.append(features)

X = np.array(X)

# 3. 머신러닝 학습 모델 알고리즘
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 학습 모델의 성능 평가 수치 및 그래프
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 출력
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# 데이터 로딩
data_directory = 'C:/Users/home/Desktop/Sheet'
annotations_folder = 'annotations'
images_folder = 'images'

annotations_path = os.path.join(data_directory, annotations_folder)
images_path = os.path.join(data_directory, images_folder)

# 빈 DataFrame 생성
labels = pd.DataFrame(columns=['filename', 'class'])

# XML 파일 읽어와서 DataFrame에 추가
for filename in os.listdir(annotations_path):
    if filename.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_path, filename))
        root = tree.getroot()
        class_label = root.find('object/name').text
        image_filename = filename.replace('.xml', '.png')
        labels = pd.concat([labels, pd.DataFrame({'filename': [image_filename], 'class': [class_label]})], ignore_index=True)

# 5. 데이터 요약 정리 및 시각화 그래프
print("Data Summary:")
print(labels.head())

# 시각화 그래프 (클래스별 분포)
class_distribution = labels['class'].value_counts()
class_distribution.plot(kind='bar', rot=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()