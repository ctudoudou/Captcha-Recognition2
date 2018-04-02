# Captcha-Recognition2
基於 “正方软件股份有限公司” 的教務管理平台提供驗證碼識別服務(公開版)

以Keras + tensorflow 使用神經網絡進行驗證碼識別服務

## 服務識別效果
字符識別基礎正確率為 95%+

## 倉庫文檔說明

```

├── LICENSE     授權文件
├── README.md   README.md
├── code        代碼文件夾
│   ├── TFtools.py      tfrecord文檔工具
│   ├── data_biaoji     已有標記的驗證碼圖片
│   │   ├── 004x.png
│   │   ├── 01ye.png
│   │   ├── ...
│   ├── dataset.py      數據處理文檔
│   ├── get_picture.py  獲取驗證碼代碼
│   ├── model.py        神經網絡模型
│   └── train.py        神經網絡訓練文檔
├── model       模型保存文件夾
│   └── net.h5          模型文件
└── requirements.txt    依賴庫文件
```

## 驗證碼服務演示
驗證碼圖片
![無標記驗證碼圖片](./model/0.gif)

識別效果

``` shell
python train.py
train or predict: predict
filename: ../model/0.gif
vf20
```

## 運行環境及時間
> 版本 MacBook Pro 2016
>
> 處理器 2.9 GHz Intel Core i5
>
> 內存 8 GB 2133 MHz LPDDR3
>
> 訓練時間 60s
>
> 預測時間 <1s

## 授權說明
授權具體說明見 [LICENSE](./LICENSE) 文檔

## 項目依賴庫
具體文件見 [requirements.txt](./requirements.txt) 文檔
