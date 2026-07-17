week4_autoencoder/

demo_why_unsupervised.py
仿照前面兩週的「破除迷思」腳本:展示如果沒有 label(或 label 很貴、很少),CNN 分類器完全無法訓練,拋出問題——「能不能不靠 label 也學到有用的特徵?」
part1_ae_mnist.py
最基本的全連接(Linear)自編碼器,在 MNIST 上做重建。刻意先用 Linear 而非 Conv,呼應 Week1→Week2 的漸進邏輯(先簡單版本,建立「bottleneck / 壓縮表示」的核心概念)。
part2_conv_ae.py
換成卷積自編碼器(Conv2d + ConvTranspose2d),直接銜接 Week3 學到的卷積知識,在 MNIST 或 CIFAR-10 上比較重建品質。
part3_latent_space_visualize.py
呼應 Week2 的 part4_mnist_pca_verify.py:把 bottleneck 的低維表示(latent code)用 PCA 或 t-SNE 視覺化,讓學生「看見」自編碼器自動學到的分群結構——這是首尾呼應、也是 self-supervised learning 最直觀的證據(沒給 label,但相似的數字自然聚在一起)。
part4_pretrain_then_finetune.py(進階,選配)
把 Week4 的 encoder 拿出來,接一個小分類頭,只用少量 label 做 fine-tune,和 Week3 從頭訓練的 CNN 比較準確率。這一步是把「self-supervised pretraining」的實際價值講清楚的關鍵一步,也是整個教程從基礎走向現代深度學習實務的收尾。

