# 🤖 AI Text Detector

一個簡單的 AI vs Human 文章分類工具，使用機器學習來判斷文本是由 AI 生成還是人類撰寫。

## 📋 專案簡介

這個專案實現了一個文本分類器，能夠：

- 接受使用者輸入的文本
- 即時分析並顯示 AI% / Human% 的判斷結果
- 提供詳細的文本特徵分析
- 使用視覺化圖表展示結果

## 🛠️ 技術棧

- **UI Framework**: Streamlit
- **Machine Learning**: scikit-learn (TF-IDF + Naive Bayes)
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Language**: Python 3.8+

## 📁 專案結構

```
hw5/
├── app.py                   # Streamlit 主應用程式
├── train_model.py          # 模型訓練腳本
├── training_data.py        # 訓練資料生成器
├── requirements.txt        # Python 依賴套件
├── ai_detector_model.pkl   # 訓練好的模型（運行後生成）
└── README.md              # 專案說明文件
```

## 🚀 安裝與執行

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 訓練模型

首次使用前需要先訓練模型：

```bash
python train_model.py
```

這會生成 `ai_detector_model.pkl` 模型檔案。

### 3. 啟動應用程式

```bash
streamlit run app.py
```

應用程式會自動在瀏覽器中打開（預設: http://localhost:8501）

## 📖 使用方式

1. **輸入文本**: 在左側文本框中輸入或貼上要分析的文本
2. **選擇範例**: 可以從下拉選單選擇預設的範例文本
3. **分析**: 點擊「🔍 Analyze Text」按鈕
4. **查看結果**:
   - 右側會顯示主要判斷結果（AI-Generated 或 Human-Written）
   - 包含信心度百分比
   - 可切換不同標籤頁查看詳細分析

## 📊 功能特色

### 1. 即時分類

- 輸入文本後立即獲得 AI/Human 分類結果
- 顯示預測信心度

### 2. 視覺化分析

- **儀表板圖**: 直觀顯示 AI 機率
- **柱狀圖**: 比較 AI vs Human 機率
- **文字雲**: 視覺化文本中的關鍵詞

### 3. 特徵分析

- **基本統計**: 字數、平均詞長、平均句長
- **詞彙豐富度**: 獨特詞彙比例
- **標點符號使用**: 逗號、句號、驚嘆號、問號統計
- **寫作風格指標**:
  - 正式過渡詞（furthermore, moreover 等）
  - 非正式表達（I think, honestly, lol 等）

## 🔬 模型說明

### 特徵提取

- **TF-IDF**: 詞頻-逆文檔頻率特徵（500 個特徵）
- **N-grams**: 單詞和雙詞組合
- **自定義特徵**: 文本統計、標點符號、語言風格標記

### 分類算法

- **Naive Bayes (MultinomialNB)**: 適合文本分類的概率模型
- **交叉驗證**: 3-fold CV 確保模型穩定性

### 訓練資料

- 15 個 AI 生成文本範例（正式、結構化）
- 15 個人類撰寫文本範例（口語化、個人化）

## 📝 AI vs Human 文本特徵

### AI 文本特徵：

- 更正式和結構化
- 使用過渡詞（furthermore, moreover, consequently）
- 詞彙多樣化
- 句子結構完整
- 較少使用縮寫和口語

### Human 文本特徵：

- 更個人化和對話式
- 使用口語表達（I think, honestly, lol）
- 句子長度變化大
- 可能有不完整句子
- 使用縮寫和俚語

## 🎯 使用案例

1. **教育用途**: 檢測學生作業是否為 AI 生成
2. **內容審核**: 識別 AI 生成的評論或文章
3. **研究分析**: 研究 AI 與人類寫作風格差異
4. **自我檢測**: 作家檢查自己的寫作風格

## ⚠️ 限制與注意事項

- 模型僅在有限的訓練資料上訓練，準確度可能有限
- 最適合分析英文文本
- 較長的文本通常能得到更準確的結果
- 這是一個教育專案，結果僅供參考

## 🔧 進階設定

### 調整模型參數

編輯 `train_model.py` 中的參數：

```python
TfidfVectorizer(
    max_features=500,      # 特徵數量
    ngram_range=(1, 2),    # N-gram 範圍
    min_df=1,              # 最小文檔頻率
    max_df=0.8             # 最大文檔頻率
)

MultinomialNB(alpha=0.1)   # 平滑參數
```

### 添加訓練資料

在 `training_data.py` 中的列表添加更多範例：

- `ai_generated_samples`: AI 生成的文本
- `human_written_samples`: 人類撰寫的文本

## 📚 依賴套件

- streamlit==1.29.0
- scikit-learn==1.3.2
- pandas==2.1.4
- numpy==1.26.2
- matplotlib==3.8.2
- seaborn==0.13.0
- plotly==5.18.0
- wordcloud==1.9.3

## 🤝 貢獻

歡迎提出問題和改進建議！

## 📄 授權

本專案為教育用途，可自由使用和修改。

## 👨‍💻 作者

物聯網應用與資料分析 HW5 專案

---

**Happy Detecting! 🚀**
