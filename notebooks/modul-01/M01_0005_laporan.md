# Laporan Praktikum M01 - Fondasi Jaringan Saraf, FNN, Aktivasi, dan Loss

## Identitas Praktikan

| Komponen | Isian |
|---|---|
| Nama | Efi Defiyati |
| NIM | 123450005 |
| Kelas | ISI KELAS |
| Modul | M01 - Fondasi Jaringan Saraf, FNN, Aktivasi, dan Loss |
| Tanggal praktikum | 2026-09-16 |
| Seed/varian individual | 1005 (=1000 + 4 digit terakhir NIM); latihan individual: sigmoid (digit terakhir NIM = 5) |
| Device | CPU |

## Ringkasan Singkat

Praktikum ini membandingkan tiga fungsi aktivasi (ReLU, Tanh, Sigmoid) pada dua ukuran hidden layer (4 dan 16) untuk klasifikasi biner dua bulan sabit (`make_moons`, n=600, noise=0.22) memakai FNN `2→h→1` yang dilatih dengan SGD (lr=0.05, batch=32, 200 epoch). Dari enam kombinasi, **Tanh dengan hidden size 16** memperoleh validation loss terendah (0,2453; val. accuracy 0,917), sedikit mengungguli ReLU h=16 (0,2558). Kedua konfigurasi Sigmoid berkinerja paling buruk dan hampir identik (val. loss ≈0,355), menunjukkan aktivasi ini menjadi bottleneck terlepas dari kapasitas hidden layer. Model final (Tanh h=16) dievaluasi satu kali pada test set dan memperoleh test loss 0,169 serta test accuracy 0,95. Forward pass manual NumPy dan implementasi PyTorch tercocokkan dengan selisih < 1e-6.

## 1. Tujuan dan Hipotesis

### 1.1 Tujuan
1. Membuktikan kesesuaian implementasi neuron/FNN manual (NumPy) dengan `nn.Sequential` PyTorch melalui forward pass dan BCE.
2. Membandingkan pengaruh fungsi aktivasi (ReLU, Tanh, Sigmoid) dan hidden size (4 vs 16) terhadap validation loss dan accuracy pada `make_moons`.
3. Memilih dan mengevaluasi satu model final memakai protokol train/validation/test tanpa kebocoran data.

### 1.2 Hipotesis sebelum eksperimen

| Perbandingan | Prediksi | Alasan teknis |
|---|---|---|
| ReLU vs Tanh/Sigmoid (h=16) | ReLU h=16 memberi validation loss terbaik | ReLU tidak jenuh pada nilai positif besar sehingga gradien tetap besar (Nair & Hinton, 2010); SGD polos konvergen lebih cepat dalam 200 epoch |
| Hidden size 16 vs 4 | Hidden size 16 memperbaiki validation loss dibanding 4 | Kapasitas representasi lebih tinggi untuk memisahkan pola non-linear `make_moons` |
| Individual: Sigmoid vs baseline ReLU (h=8) | Sigmoid berkonvergensi lebih lambat, val. loss lebih tinggi | Sigmoid mudah jenuh dan tidak berpusat di nol, sehingga dapat memperlambat pembelajaran gradien pada jaringan dalam (Glorot & Bengio, 2010) |

## 2. Data dan Protokol Eksperimen

### 2.1 Dataset dan split

| Komponen | Nilai |
|---|---|
| Dataset | `sklearn.datasets.make_moons`, n=600, noise=0.22 |
| Jumlah kelas | 2 (biner) |
| Train | 360 (60%) |
| Validation | 120 (20%) |
| Test | 120 (20%) |
| Cara split | `train_test_split` terstratifikasi dua tahap, `random_state=SEED=1005` |
| Praproses utama | `StandardScaler` (mean/std) |

Pencegahan kebocoran: `StandardScaler` di *fit* hanya pada train set, lalu dipakai untuk *transform* validation dan test set; test set baru dievaluasi satu kali setelah model final dipilih dari validation loss.

### 2.2 Konfigurasi yang dikendalikan

| Komponen | Nilai yang digunakan |
|---|---|
| Arsitektur dasar | FNN `2 → hidden → 1` (Linear-Aktivasi-Linear) |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | SGD |
| Learning rate | 0.05 |
| Batch size | 32 |
| Epoch | 200 |
| Seed | 1005 |
| Kriteria pemilihan model | Validation loss terendah |
| Batas komputasi | 6 konfigurasi × 200 epoch (~1 detik/run pada CPU) |

**Sama untuk seluruh run:** split data, scaler, optimizer, learning rate, batch size, epoch, seed.
**Sengaja diubah:** fungsi aktivasi (relu/tanh/sigmoid) dan hidden size (4/16).

### 2.3 Lingkungan eksekusi

```text
Python     : 3.12.3
PyTorch    : 2.14.0 (Paszke et al., 2019)
NumPy      : 2.4.4
Device     : CPU
Runtime    : Lokal (container Jupyter)
```
Dataset dihasilkan dan diproses memakai scikit-learn (Pedregosa et al., 2011).

## 3. Implementasi dan Pemeriksaan Kebenaran

| Pemeriksaan | Nilai diharapkan | Hasil aktual | Status |
|---|---:|---:|---|
| Selisih logit NumPy vs PyTorch | < 1e-6 | 0,0 (identik) | Lulus |
| Selisih BCE NumPy vs PyTorch | < 1e-6 | 0,0 (identik) | Lulus |
| Parameter FNN `2→8→1` | 4h+1 = 33 | 33 | Lulus |
| Rentang keluaran sigmoid pada uji neuron | (0,1) | 0,223–0,852 | Lulus |

**Temuan:** Forward pass manual (NumPy) dan `nn.Sequential` PyTorch menghasilkan logit (2,5), probabilitas (0,9241), dan BCE (0,0789) yang identik hingga presisi numerik, sehingga implementasi model FNN dapat dipercaya sebelum dipakai pada eksperimen skala penuh. Mekanisme pembelajaran parameter itu sendiri (backpropagation) mengikuti algoritma klasik Rumelhart, Hinton, dan Williams (1986) dan akan dibahas lebih rinci pada Modul 2.

## 4. Hasil Eksperimen

### 4.1 Tabel hasil utama (dari `M01_0005_metrics.csv`)

| Run | Aktivasi/hidden | Param. | Train loss | Val. loss | Val. accuracy | Waktu (detik) |
|---|---|---:|---:|---:|---:|---:|
| tanh_h16 (final) | tanh, h=16 | 65 | 0,187 | **0,2453** | 0,917 | 0,98 |
| relu_h16 | relu, h=16 | 65 | 0,201 | 0,2558 | 0,900 | 0,98 |
| tanh_h4 | tanh, h=4 | 17 | 0,229 | 0,2750 | 0,908 | 0,98 |
| sigmoid_h4 | sigmoid, h=4 | 17 | 0,330 | 0,3551 | 0,850 | 0,97 |
| sigmoid_h16 | sigmoid, h=16 | 65 | 0,323 | 0,3555 | 0,850 | 0,98 |
| relu_h4 | relu, h=4 | 17 | 0,303 | 0,3585 | 0,850 | 0,99 |

Model final (tanh_h16) pada **test set**: test loss = 0,169, test accuracy = 0,950 (dievaluasi satu kali).

### 4.2 Grafik utama

![Validation loss enam eksperimen](https://drive.google.com/drive/u/0/folders/1MHEeckIzVePdrsclSsLbuLf2fFTWW3va)

**Gambar 1.** Kurva validation loss (BCE) terhadap epoch untuk keenam kombinasi aktivasi × hidden size, pada validation set (n=120). Tanh (garis merah dan hijau) turun paling konsisten hingga akhir 200 epoch, sedangkan kedua kurva Sigmoid (ungu dan cokelat) melandai di atas 0,35 dan nyaris berhimpit.

**Temuan dari Gambar 1:** tanh_h16 (garis merah) mencapai validation loss terendah (0,2453) dan masih menurun di epoch ke-200, mengindikasikan model belum sepenuhnya konvergen; sebaliknya kedua kurva Sigmoid mendatar sejak sekitar epoch ke-100 pada level ≈0,355, konsisten dengan saturasi gradien pada aktivasi tersebut.

![Decision boundary model final](https://drive.google.com/drive/u/0/folders/1MHEeckIzVePdrsclSsLbuLf2fFTWW3va)

**Gambar 2.** Decision boundary model final (tanh_h16) pada test set (n=120); kontur warna menunjukkan probabilitas kelas 1, garis hitam adalah ambang 0,5.

**Temuan dari Gambar 2:** batas keputusan mengikuti bentuk lengkung kedua bulan sabit dengan cukup baik; sebagian besar titik salah klasifikasi (test accuracy 0,95, artinya 6 dari 120 titik salah) berada tepat di sekitar garis hitam, pada zona tumpang tindih akibat noise=0,22.

## 5. Analisis dan Pembahasan

| Unsur | Isi |
|---|---|
| Klaim | Tanh dengan hidden size besar (16) memberi validation loss terbaik, mengalahkan ReLU pada anggaran epoch yang sama |
| Bukti | Tabel 4.1: tanh_h16 val_loss=0,2453 vs relu_h16 val_loss=0,2558; Gambar 1 menunjukkan kurva tanh_h16 masih menurun di akhir training |
| Penalaran | Tanh berpusat di nol sehingga rata-rata aktivasi mendekati nol, yang secara empiris sering mempercepat optimisasi berbasis SGD polos dibanding ReLU pada jaringan sekecil ini; keunggulan teoritis ReLU (tidak mudah jenuh) yang diperkenalkan oleh Nair dan Hinton (2010) tidak selalu menang pada epoch terbatas dan learning rate tetap |

1. **Perbandingan dengan hipotesis:** Sebagian ditolak, karena hipotesis memprediksi ReLU h=16 terbaik, tetapi hasil aktual menunjukkan Tanh h=16 unggul tipis (val_loss 0,2453 vs 0,2558). Hipotesis hidden size 16 lebih baik dari 4 terbukti benar untuk ReLU dan Tanh. Hipotesis Sigmoid individual lebih lambat dan lebih buruk dari baseline ReLU **terbukti benar** (val_loss sigmoid_h8 0,3557 dibandingkan relu_h8 baseline 0,3559, keduanya jauh di atas Tanh).
2. **Run terbaik dan terburuk:** Terbaik tanh_h16 (val_loss 0,2453); terburuk relu_h4 (val_loss 0,3585). Perbedaan utama adalah kombinasi aktivasi yang mudah jenuh/kapasitas kecil (relu_h4 hanya 17 parameter) memberi kapasitas dan gradien yang kurang optimal dibanding tanh_h16 (65 parameter, aktivasi berpusat nol).
3. **Trade-off:** tanh_h16 memakai 65 parameter dan waktu 0,98 detik, hampir sama dengan relu_h16 dan sigmoid_h16 (memakai jumlah parameter dan waktu yang identik); tidak ada trade-off biaya komputasi yang berarti di antara ketiga aktivasi pada hidden size sama, sehingga pemilihan aktivasi murni berdasarkan akurasi/loss.
4. **Anomali:** Tidak ada run yang gagal (semua BCE finite, tidak ada NaN). Sedikit anomali: peningkatan hidden size pada Sigmoid (h=4 menjadi 16) nyaris tidak mengubah val_loss (0,3551 menjadi 0,3555), menandakan model terjebak pada wilayah saturasi aktivasi terlepas dari kapasitas tambahan, sejalan dengan temuan Glorot dan Bengio (2010) bahwa aktivasi sigmoid dapat mendorong unit ke wilayah jenuh pada inisialisasi acak standar.
5. **Generalisasi:** Train-val gap tanh_h16 kecil (0,187 vs 0,245, selisih 0,058) dan test loss (0,169) justru lebih rendah dari validation loss, menunjukkan model tidak overfitting dan generalisasi cukup baik pada dataset sintetis ini.

## 6. Jawaban Pertanyaan Modul

1. **Mengapa output terakhir tidak diberi sigmoid di dalam model?**
   Karena `BCEWithLogitsLoss`/`binary_cross_entropy_with_logits` menggabungkan sigmoid dan BCE dalam satu operasi yang stabil secara numerik (log-sum-exp), sehingga logit mentah harus diberikan langsung ke loss; menambah sigmoid eksplisit pada model akan membuat probabilitas dihitung dua kali dan berisiko menimbulkan gradien tidak stabil pada logit ekstrem.

2. **Berapa parameter FNN `2→8→1`?**
   `Linear(2,8)`: 2×8+8=24; `Linear(8,1)`: 8×1+1=9. Total = 33, sesuai rumus `4h+1 = 4(8)+1 = 33` dan sudah dibuktikan lewat `assert count_parameters(model_check) == 33` (Bagian 3).

3. **Bukti apa yang harus ditunjukkan sebelum menyatakan satu aktivasi lebih baik?**
   Validation loss dan accuracy pada protokol split dan anggaran (epoch, lr, batch size) yang identik untuk semua aktivasi yang dibandingkan (Tabel 4.1), idealnya diulang pada beberapa seed untuk memastikan perbedaan bukan sekadar variasi inisialisasi acak, dan bukan hanya train loss atau hasil dari satu kali run tunggal.

## 7. Kesimpulan dan Keterbatasan

### 7.1 Kesimpulan
Praktikum berhasil membuktikan kecocokan implementasi NumPy dan PyTorch (selisih <1e-6) serta membandingkan enam kombinasi aktivasi/hidden size pada `make_moons`. Model final **Tanh, hidden size 16** dipilih karena validation loss terendah (0,2453) dan validation accuracy tertinggi (0,917), lalu diverifikasi pada test set dengan hasil konsisten (test loss 0,169, test accuracy 0,950), menunjukkan model dapat menggeneralisasi dengan baik tanpa tanda overfitting berarti.

### 7.2 Keterbatasan
- Hanya satu seed (1005) per konfigurasi digunakan; hasil val_loss antar aktivasi bisa dipengaruhi variasi inisialisasi acak, bukan murni perbedaan aktivasi.
- Epoch tetap 200 dan learning rate tetap 0,05 untuk semua konfigurasi; konfigurasi seperti ReLU mungkin memerlukan learning rate/epoch berbeda untuk mencapai performa optimalnya, sehingga perbandingan antar aktivasi belum tentu adil pada anggaran hyperparameter yang sama persis.

### 7.3 Tindak lanjut
Mengulang keenam konfigurasi dengan 5 seed berbeda dan melaporkan validation loss rata-rata ± standar deviasi, untuk memastikan keunggulan Tanh h=16 atas ReLU h=16 bukan kebetulan akibat satu inisialisasi bobot tertentu.

## Referensi
1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
2. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*, 807–814.
3. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 9, 249–256. https://proceedings.mlr.press/v9/glorot10a.html
4. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8024–8035. https://arxiv.org/abs/1912.01703
5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
6. Dokumentasi resmi PyTorch, versi 2.14.0. `torch.nn`, `torch.nn.BCEWithLogitsLoss`, `torch.optim.SGD`. https://pytorch.org/docs/stable/
7. Dokumentasi resmi scikit-learn, versi 1.8.0. `sklearn.datasets.make_moons`, `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.train_test_split`. https://scikit-learn.org/stable/

## Pernyataan Orisinalitas
Saya menyatakan bahwa kode, eksperimen, analisis, dan laporan ini merupakan pekerjaan individual. Semua sumber eksternal, termasuk potongan kode, telah dicantumkan. Saya memahami bahwa kemiripan hasil akibat seed atau data yang sama tidak membenarkan penyalinan notebook maupun analisis.

**Nama:** Efi Defiyati
**Tanggal:** 2026-09-15
