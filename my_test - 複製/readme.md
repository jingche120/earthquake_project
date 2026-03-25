## 1. 專案目的

本專案包含兩個核心階段，旨在完成從資料處理到模型部署的完整驗證流程：

1. **資料前處理與視覺化**：從 CWA 原始 HDF5 資料庫中，精確提取特定地震事件的波形（P 波前後 5 秒），並將其轉換為可視化圖表與數值檔案，供模型輸入或人工檢查使用。
2. **模型一致性驗證**：驗證深度學習模型在 **Python (PyTorch)** 訓練環境與 **C++ (ONNX Runtime)** 部署環境中的推論結果是否完全一致（誤差需為 0）。

---

## 2. 檔案結構

請確保你的工作目錄包含以下檔案與資料夾結構：

```
Project_Root/
│
├── CWA_processed_data/     # [資料來源] 在nas裡面
│   ├── all.hdf5            # 原始波形資料庫
│   └── all_metadata.csv    # 地震事件列表
│
├── extracted_data/         # [輸出目錄] 執行 Step 1 後自動產生，存放波形圖與數值
│
├── extract_single_waveform.py # [Step 1 程式] 用於從 HDF5 抓取單一波形並畫圖
│
├── new_predict.py          # [Step 2 程式] Python 端：產生模型、輸入資料與標準答案
├── new_predict_c++.cpp     # [Step 3 程式] C++ 端：讀取 ONNX 模型與資料進行推論
├── CMakeLists.txt          # [編譯設定檔] 自動化 C++ 編譯與檔案搬運
├── epoch11.pth             # [權重檔] PyTorch 預訓練權重
├── TC_UNet.py              # [模型架構] PyTorch 模型定義檔
│
├── onnxruntime/            # [必要的 C++ 函式庫]
│   ├── include/            # 內含 onnxruntime_cxx_api.h
│   ├── lib/                # 內含 onnxruntime.lib
│   └── bin/                # 內含 onnxruntime.dll
│
└── build/                  # [編譯目錄] 執行 CMake 後自動產生

```

---

## 3. 操作步驟 (Step-by-Step)

### Step 1: 資料提取與波形檢查 (Python)

此步驟用於確認原始資料品質，並提取出標準的 10 秒波形片段。

在終端機 (Terminal) 執行：

```powershell
python extract_single_waveform.py

```

**執行後會發生什麼事？**

1. 程式會讀取 `CWA_processed_data` 中的資料。
2. 自動抓取一筆符合條件的地震波形（以 P 波為中心，前後各 5 秒）。
3. 在 `extracted_data/` 資料夾中產生三個檔案：
    - **`_waveform.png`**：三軸波形圖（含 P 波標記），用於人工檢查。
    - **`_10sec.csv`**：Excel 可讀的數值檔（含時間軸）。
    - **`_10sec.npy`**：Python NumPy 原始陣列檔（可作為後續模型的輸入）。

---

### Step 2: 產生 ONNX 模型與標準答案 (Python)

此步驟將 PyTorch 模型轉換為 ONNX 格式，並計算出一組「標準答案」。

在終端機執行：

```powershell
python new_predict.py

```

**執行後會發生什麼事？**

1. 若 `input_data.bin` 不存在，會自動產生一份固定輸入資料。
2. 匯出 `tc_unet.onnx` 模型檔。
3. **終端機顯示結果**：請複製或記下 `[Python] 預測結果` 的那串 List 數值（這是標準答案）。

---

### Step 3: 編譯 C++ 推論程式 (使用 CMake)

我們使用 CMake 來自動處理繁雜的路徑設定與編譯。

**前置作業**：
請務必使用 Windows 的 **「Developer Command Prompt for VS 2022」** (開發人員命令提示字元) 開啟 VS Code，以確保編譯器環境變數正確。
*(指令：在黑視窗進入專案資料夾後輸入 `code .`)*

在 VS Code 終端機中，依序輸入：

1. **產生建置設定** (建立 `build` 資料夾)：
    
    ```powershell
    cmake -B build
    
    ```
    
    *(若出現 `Configuring done` 代表成功)*
    
2. **開始編譯** (產生 .exe 並自動複製模型與資料)：
    
    ```powershell
    cmake --build build --config Release
    
    ```
    
    *(若出現 `Build finished` 代表成功)*
    

---

### Step 4: 執行 C++ 推論 (驗證結果)

編譯成功後，執行檔位於 `build/Release` 資料夾內。請直接執行：

```powershell
.\\build\\Release\\TestRunner.exe

```

**執行後會發生什麼事？**

1. C++ 程式讀取與 Python 相同的 `input_data.bin`。
2. 載入 `tc_unet.onnx` 模型。
3. **終端機顯示結果**：印出 `[C++] 預測結果` 的 List。

---

## 4. 最終驗證 (Verification)

請將 **Step 2 (Python)** 與 **Step 4 (C++)** 的輸出放在一起比對：

- **Python 輸出範例**：
`[0.12345678, 0.98765432, ...]`
- **C++ 輸出範例**：
`[0.12345678, 0.98765432, ...]`

**判定標準**：如果兩個 List 的數值連**小數點後 8 位都一模一樣**，代表模型移植成功，專案驗證完成！

---

## 5. 常見問題排除 (Troubleshooting)

- **Q: Step 1 執行失敗，顯示找不到 HDF5 檔案？**
    - **A**: 請確認您已將大型資料檔 `all.hdf5` 放入 `CWA_processed_data/` 資料夾中。該檔案通常不包含在程式碼庫中，需手動下載。
- **Q: `cmake` 指令無法辨識？**
    - **A**: 電腦未設定 CMake 路徑。請關閉 VS Code，搜尋並開啟 **"Developer Command Prompt for VS 2022"**，用該視窗進入資料夾並輸入 `code .` 重開 VS Code。
- **Q: C++ 輸出顯示亂碼 `?葫...`？**
    - **A**: 這是編碼問題。請在執行 .exe 前，於終端機輸入 `chcp 65001` 切換成 UTF-8 模式，或直接比對數值即可。
- **Q: 編譯時找不到 `onnxruntime_cxx_api.h`？**
    - **A**: 請檢查 `onnxruntime` 資料夾結構，確保打開 `onnxruntime/include` 就能直接看到 `.h` 檔案，路徑中間不要多夾一層版本號資料夾。