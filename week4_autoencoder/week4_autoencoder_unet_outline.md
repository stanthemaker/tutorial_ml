# Week 4 — Autoencoder → CNN Autoencoder → U-Net（教學大綱草案，待確認）

> 假設 CNN 是 `week3_cnn/`，這一週編號為 `week4_unet/`。若編號不同請告訴我。

## 這一週要回答的核心問題

到 Week 3 為止，模型都是「影像 → 一個標籤」，最後一層把整張圖壓成 10 個數字。
這週要問的是：**如果輸出也必須是一張影像呢？**

分類只需要「壓縮」，重建任務同時需要「壓縮」和「還原」。整週就是在補「還原」這一半，
而 skip connection 則是在修補「壓縮時丟掉的東西」。

## 承接與新概念

**沿用（不重講）**：`Conv2d`、`ReLU`、`MaxPool`、通道 (channel) 概念、
Week 1 Part 3 建立的標準訓練迴圈樣板。

**這週唯一的三個新概念**：

| 新概念 | 出現位置 | 一句話說明 |
|---|---|---|
| bottleneck（瓶頸層） | Part 1 | 強迫資訊通過低維空間，模型只好學到「重點」 |
| upsampling（上採樣） | Part 2 | 把 feature map 變大，`ConvTranspose2d` / `Upsample` |
| skip connection | Part 3 | 把 encoder 的高解析度特徵直接接到 decoder，補回細節 |

## 檔案規劃 `week4_unet/`

| 檔名 | 主題 | 產出的「啊哈」時刻 |
|---|---|---|
| `part1_mlp_autoencoder.py` | 用 MLP 做 MNIST 壓縮與重建 | 32 維就能還原一張 784 維的圖 |
| `part2_cnn_autoencoder.py` | 換成 Conv encoder + 轉置卷積 decoder | 參數量少很多，重建還更好 |
| `demo_why_skip_connections.py` | 去噪任務下 CNN AE 的極限 | 形狀對了，但邊緣和筆畫細節糊掉 |
| `part3_unet.py` | 加上 skip connection | 同樣的 bottleneck，細節回來了 |
| `part4_unet_task.py`（選配） | 換到比較真實的任務 | 同一套架構可以做 inpainting / 分割 |

---

## Part 1 — MLP Autoencoder

**任務**：MNIST 自我重建，input = target = 同一張圖（無標籤，第一次遇到 unsupervised）。

**架構**
```
Encoder: 784 → 128 → 32        (ReLU)
Decoder: 32 → 128 → 784        (ReLU, 最後接 Sigmoid)
Loss:    MSELoss(recon, x)
```

**要強調的點**
1. 訓練迴圈跟 Week 1 Part 3 一字不差，只有 `loss_fn(pred, y)` 變成 `loss_fn(recon, x)`。
2. 資料集不需要 label —「監督訊號」來自資料本身。
3. 最後一層 Sigmoid 的理由：像素值被正規化到 [0, 1]。

**實驗（建議做成迴圈掃過）**
- bottleneck 維度 = 128 / 32 / 8 / 2，並排比較重建結果。
- bottleneck = 2 時把 latent 畫成散點圖、依數字類別上色 → 可以直接呼應
  `week2_mlp/part4_mnist_pca_verify.py`，說明「這就是學出來的降維」。
- 補一句（不展開推導）：**移掉所有 ReLU 的線性 autoencoder ≈ PCA**，
  非線性版本則能把 PCA 分不開的類別分開。

**TODO 挖空建議**：`nn.Sequential` 的層定義、`loss_fn(recon, x)` 這一行。

---

## Part 2 — CNN Autoencoder

**動機**：Part 1 的第一步 `x.view(-1, 784)` 把二維結構丟了。Week 3 已經知道卷積能保留空間結構——decoder 該怎麼反過來做？

**架構**
```
Encoder: Conv(1→16, s=2) → Conv(16→32, s=2)      28×28 → 14×14 → 7×7
Decoder: ConvT(32→16, s=2) → ConvT(16→1, s=2)    7×7 → 14×14 → 28×28
```

**要強調的點**
1. 下採樣兩種寫法：`stride=2` 的卷積 vs `MaxPool`（這裡用 stride，之後 U-Net 用 pool，可對照）。
2. `ConvTranspose2d` 直觀說明：不是「反卷積」，而是把每個輸入像素攤開成一塊再疊加。
   `output_padding` / 尺寸怎麼算，給一張手算的圖。
3. **棋盤格 artifact (checkerboard)**：kernel size 不能被 stride 整除時會出現。
   替代寫法 `Upsample(scale_factor=2) + Conv2d`，兩種都跑一次讓學生看差別。
4. 跟 Part 1 比參數量與 loss ——這是「架構的 inductive bias 有用」的具體證據。

**TODO 挖空建議**：整個 decoder 的 `ConvTranspose2d` 參數。

---

## demo_why_skip_connections.py — U-Net 的動機

**任務改成去噪 (denoising)**：input = 加了高斯雜訊的圖，target = 乾淨原圖。
（這是第一個 input ≠ target 的重建任務，也讓 autoencoder 從「複製」變成真的有用。）

用 Part 2 的 CNN AE 直接跑，然後看三列對比圖：`乾淨 / 加噪 / 重建`。

**預期觀察**：整體形狀正確，但筆畫邊緣模糊、細節被抹平。
**解釋**：所有資訊都被迫擠過 7×7 的 bottleneck，高頻（邊緣、細節）在下採樣時就丟了，
decoder 沒辦法無中生有。
**提問收尾**：encoder 早期的 feature map 明明還有這些細節——為什麼不直接拿過來用？

---

## Part 3 — U-Net

**架構**（MNIST 尺寸，比原論文淺）
```
DoubleConv = Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU

enc1: DoubleConv(1→32)     28×28  ─────────────┐
      MaxPool                                   │ skip
enc2: DoubleConv(32→64)    14×14  ────────┐    │
      MaxPool                              │    │
bott: DoubleConv(64→128)    7×7            │    │
      ConvT(128→64)        14×14           │    │
dec2: DoubleConv(128→64)   ← cat(64, 64) ──┘    │
      ConvT(64→32)         28×28                │
dec1: DoubleConv(64→32)    ← cat(32, 32) ───────┘
head: Conv1×1(32→1) → Sigmoid
```

**要強調的點**
1. `torch.cat([up, skip], dim=1)` —— **沿 channel 維度串接，不是相加**。
   若之前提過 ResNet，這裡明確對比 concat vs add 的差別。
2. Decoder 每個 block 的輸入通道會加倍（64 → 128），這是學生最常寫錯的地方。
3. 尺寸對齊：`padding=1` 保持尺寸是最省事的做法
4. 「U」的形狀不是裝飾——左右對稱是為了讓每層 skip 的解析度對得上。

**驗證**：跟 demo 用同一個去噪任務、同樣的 epoch 數，並排比較。
另外做一個消融實驗：把 skip connection 註解掉再跑一次，證明差別來自 skip 而不是參數變多。

**TODO 挖空建議**：`DoubleConv` 的定義、`torch.cat` 那幾行、decoder 的輸入通道數。

---

## Part 4換到比較真實的任務

三選一，看時間：
- **Inpainting**：把圖中間挖掉一塊，讓模型補回來。程式碼改動最小。
- **Segmentation**：U-Net 的原始用途，但要換資料集（Oxford-IIIT Pet 約 800MB）。
---
## 共通的工程細節

- **視覺化**：統一寫一個 `show_grid(clean, noisy, recon)` 三列對比函式，各 part 共用。
- **評估指標**：主要看 MSE；PSNR 只用一行公式帶過，SSIM 建議略過（要多裝套件）。
- **資料集**：全週用 Cifar10 or Oxford-IIIT  for demoing skip connection
