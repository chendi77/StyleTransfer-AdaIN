# StyleTransfer-AdaIN

机器学习课程作业，使用AdaIN实现图像风格迁移

## 运行测试
python test.py -m ./decoder_160000.pth -c ./content/lenna.jpg -s ./style/brushstrokes.jpg -r ./result.png

## 训练
python train.py --content_dir ./content --style_dir ./style

## 安装
pip install -r requirements.txt

## 数据集

### 内容数据集

MS-COCO，test2017

http://images.cocodataset.org/zips/test2017.zip

### 风格数据集

wikiart

[https://storage.googleapis.com/kagglesdsdata/competitions/5127/868727/train_2.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589648393&Signature=BOXiw%2BqK9EjJu5NStRasE9W%2Fe7nFOVawPwMVOC%2B8OnsmWIdjMBxKqztI6rxi3oPid9kZGcbLALdD3ZI6fY8AOFg4PUisKXC6oiLWaet9tuvASGfGAnZqwfVCL](https://storage.googleapis.com/kagglesdsdata/competitions/5127/868727/train_2.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589648393&Signature=BOXiw%2BqK9EjJu5NStRasE9W%2Fe7nFOVawPwMVOC%2B8OnsmWIdjMBxKqztI6rxi3oPid9kZGcbLALdD3ZI6fY8AOFg4PUisKXC6oiLWaet9tuvASGfGAnZqwfVCLMGlFH6wudb1j%2FDVqkPCvIzV%2F9Ll8HF%2FGAM%2BtS2E8tfkHU74dpVREiL%2BfV759YmUHXFH0uxyZorNfmEPaC1AwfEvqzBUutrptGwjjJqh3FfwKYqkZ%2BhKntiZI4KqVGXv6sOXri6EhP4oYPqP7r08b7kjURHQG5eM3MsPBF%2FYTolc23ZstwF3dMcBzO%2BCAUbKn%2FE6TpLoR2cmsqjDXPfFtFDVsei0Gg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain_2.zip)