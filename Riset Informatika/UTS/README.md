# Proposal Penelitian: Analisis Perbandingan ConvNeXt dan Data-efficient Image Transformer (DeiT) untuk klasifikasi Fine-Grained Varietas Beras Berdasarkan Citra Biji
## 1. Proposal Penelitian

### 1.1 Latar Belakang
Beras adalah komoditas pangan pokok, dan mutu serta kemurnian varietasnya memengaruhi harga, proses pengelolahan, dan hasil budidaya. Identifikasi varietas secara manual bergantung pada pengamatan visual oleh tenaga ahli, sehingga bersifat subjektif, lambat, dan melelahkan. Klasifikasi otomatis berbasis citra dan deep learning menjadi alternatif yang konsisten dan dapat diskalakan.

Masalahnya, varietas beras berasal dari spesies yang sama sehingga perbedaan antar kelas sangat tipis (bentuk, ukuran, tekstur, dan warna yang mirip), sementara variasi di dalam satu kelas juga ada. Kondisi ini termasuk dalam masalah **klasifikasi fine-grained**.

Dua keluarga arsitektur yang dominan saat ini adalah:
* **ConvNeXt** adalah CNN yang dimodernisasi dengan prinsip desain Trnasformer namun tetap sepenuhnya konvolusional. Bias induktif konvolusi (lookalitas dan translasi) cocok untuk menangkap detail lokal.
* **DeiT (DData-efficient Image Transformer)** adalah Vision Transformer yang dilatih secara efisien pada data terbatas dengan resep augmentasi dan distilasi sehingga tidak membutuhkan dataset berskala sangat besar.

Penelitian klasifikasi beras sebelumnya banyak menguji CNN atau ViT, tetapi umumnya pada dataset dengan jumlah kelas sedikit (5 sampai 13 kelas), dan belum membandingkan CNN modern dengan ViT efisien-data dalam satu protokol yang sama beserta analisis efisiensi komputasi. Penelitian ini memakai dataset 38 varietas beras Bangladesh yang dirilis pada 2024 (Tahsin dkk., 2024), yang memberi kesempatan menguji kedua arsitektur pada kasus fine-grained yang lebih menantang dan terbuka.

### 1.2 Rumusan Masalah
1. Bagaimana performa ConvNeXt dan DeiT dalam mengklasifikasikan 38 varietas beras berdasarkan citra biji?
2. Bagaimana perbandingan keduanya dari sisi jumlah parameter, FLOPs, ukuran file, dan waktu inferensi?
3. Pasangan varietas mana yang paling sering tertukar oleh model, dan apa penyebab visualnya?

### 1.3 Tujuan Penelitian
1. Mengukur accuracy dan macro F1-score ConvNeXt-Tiny dan DeiT-Small pada dataset 38 varietas beras.
2. Membandingkan efisiensi keduanya (parameter, FLOPs, ukuran file, waktu inferensi).
3. Menentukan model yang paling seimbang antara akurasi dan efisiensi.
4. Menganalisis varietas yang sering salah diklasifikasikan.

### 1.4 Manfaat
* Akademis: bukti empiris perbandingan CNN modern dan transformer efisien-data pada klasifikasi fine-grained beras, dengan protokol yang dapat direproduksi.
* Praktis: bahan pertimbangan pemilihan model untuk sistem identifikasi varietas beras dengan memperhatikan akurasi dan efisiensi.
* Bagi penulis: latihan menyusun riset dari identifikasi gap sampai rencana evaluasi.

### 1.5 Hipotesis Awal 
ConvNeXt-Tiny diperkirakan setara atau sedikit lebih akurat daripada DeiT-Small karena bias induktif konvolusi cocok untuk detail lokal biji beras, sedangkan DeiT-Small tetap kompetitif dengan parameter lebih sedikit. (Ini hipotesis yang akan diuji, bukan hasil.)

### 1.6 Batasan Penelitian 
* Hanya klasifikasi satu varietas per citra, pada dataset yang dijelaskan di bagian 3.
* Hanya dua model utama (ConvNeXt-Tiny dan DeiT-Small), dengan bobot awal ImageNet-1K.
* Penelitian tidak membuat aplikasi; pembuatan aplikasi menjadi pengembangan lanjutan.

## 2. Research Gap

Penelitian klasifikasi beras sudah banyak, tetapi perbandingan langsung CNN modern dan transformer efisien-data pada banyak kelas fine-grained masih jarang.
* ConvNeXt (Liu dkk., 2022) memodernisasi ResNet dengan desain terinspirasi Transformer dan dilaporkan menyamai atau melampaui Swin Transformer pada ImageNet.
* DeiT (Touvron dkk., 2021) menunjukkan bahwa ViT dapat dilatih efektif hanya dengan ImageNet-1K melalui strategi pelatihan dan distilasi.
* Koklu dkk. (2021) mengklasifikasikan 5 varietas beras Turki dengan 75.000 citra dan CNN mencapai akurasi 100%. Jumlah kelas yang sedikit membuat tugasnya relatif mudah, sehingga belum mewakili kasus fine-grained.
* Rajalakshmi dkk. (2024) membangun RiceSeedNet berbasis ViT untuk 13 varietas beras India (13.000 citra) dengan akurasi 97%. Modelnya dirancang sendiri dan bukan perbandingan langsung ConvNeXt dan DeiT.
* Zhang dkk. (2025) menyusun dataset 36 varietas biji beras dan mengajukan RSCD-Net. Mereka mencatat bahwa sedikit penelitian meneliti lebih dari 15 varietas. ConvNeXt dan Swin Transformer hanya menjadi pembanding, dan DeiT tidak termasuk model pembanding yang disebut pada abstraknya.
* Tahsin dkk. (2024) merilis dataset 38 varietas beras Bangladesh (19.000 citra asli, 500 per kelas) yang diambil dengan kamera mikroskop. Makalah ini berfokus pada penyediaan dataset sebagai tolok ukur, bukan pada perbandingan arsitektur.
* Gap utama: penelitian beras terdahulu umumnya menguji satu jenis arsitektur pada jumlah kelas kecil. Belum banyak yang membandingkan CNN modern (ConvNeXt) dan ViT efisien-data (DeiT) dalam satu protokol yang sama pada puluhan varietas, lengkap dengan analisis efisiensi komputasi.

## 3. Dataset Riset Kecil 

* Nama: An Extensive Image Dataset for Classifying Rice Varieties in Bangladesh
* Sumber: Mendeley Data, dibuat oleh Tahsin dkk. (East West University), terbit 2024. Artikel pendampingnya terbit di Data in Brief vol. 57.
* Alamat: data.mendeley.com/datasets/2fgv99854n/1 (DOI 10.17632/2fgv99854n.1)
* Isi: 19.000 citra asli berformat JPG, 640×480 piksel, 38 kelas varietas, masing-masing 500 citra (seimbang). Citra diambil dengan kamera mikroskop digital pada 15 Januari sampai 28 Februari 2024, dari varietas koleksi BINA dan BRRI.
* Daftar kelas (38):
    * Seri BD (18): BD30, BD33, BD39, BD49, BD51, BD52, BD56, BD57, BD70, BD72, BD75, BD76, BD79, BD85, BD87, BD91, BD93, BD95
    * Seri Binadhan (15): Binadhan7, 8, 10, 11, 12, 14, 16, 17, 19, 20, 21, 23, 24, 25, 26
    * Seri BR dan BRRI (5): BR22, BR23, BRRI67, BRRI74, BRRI102
* Catatan penting:
    ◦ Dataset juga menyertakan folder Augmented. Penelitian ini hanya memakai folder Original, karena citra augmentasi berasal dari citra asli yang sama dan bisa bocor antara data latih dan data uji.
    ◦ Angka di atas (19.000 citra, 38 kelas, 500 per kelas, 640×480) berasal dari deskripsi dataset. Setelah unduhan selesai, jumlah per folder dan ukuran citra akan dihitung ulang dan dicatat di laporan akhir.
    ◦ Cek citra duplikat atau hampir sama sebelum membagi data supaya tidak ada gambar yang sama di data latih dan data uji.
    ◦ Cek lisensi dan ketentuan penggunaan di halaman Mendeley sebelum dataset dipakai atau disebarkan.
    ◦ Citra diambil dengan mikroskop dalam kondisi terkontrol, sehingga hasil belum tentu berlaku untuk foto dari kamera ponsel.
• Pembagian data: stratified 70% latih, 15% validasi, 15% uji, yaitu 350 / 75 / 75 citra per kelas (13.300 / 2.850 / 2.850 citra). Pembagian dilakukan sebelum augmentasi apa pun, dengan seed tetap dan daftar berkas tiap split disimpan dalam berkas CSV agar bisa diulang.

## 4. Research Mind Map

Penelitian ini mengisi gap: ConvNeXt vs DeiT pada 38 varietas beras

<img width="700" height="1000" alt="Diagram Biji Beras drawio" src="https://github.com/user-attachments/assets/8b3fbf97-941d-4b65-a52a-0258c8bf6a2b" />

Peta dibaca dari kiri ke kanan: masalah dan studi terdahulu mengarah ke research gap, yang dijawab lewat dataset, metode, dan evaluasi

## 5. Rencana Evaluation Metric

Semua metrik dihitung pada data uji (data yang tidak dipakai saat pelatihan). Data validasi hanya dipakai untuk memilih epoch terbaik dan menyetel hiperparameter. Eksperimen diulang dengan 3 seed berbeda per model, lalu dilaporkan rata-rata dan standar deviasinya.

### 5.1 Metrik Kualitas Klasifikasi 
* Accuracy: persentase prediksi yang benar dari seluruh citra uji. Cocok sebagai gambaran umum karena dataset seimbang (500 citra per kelas).
* Precision (macro): dari semua citra yang ditebak sebagai suatu varietas, berapa yang benar. Dihitung per kelas lalu dirata-rata.
* Recall (macro): dari semua citra suatu varietas yang sebenarnya ada, berapa yang berhasil ditemukan. Dihitung per kelas lalu dirata-rata.
* F1-score (macro): rata-rata harmonis precision dan recall, dirata-rata antar kelas. Metrik utama untuk membandingkan model karena memperlakukan semua kelas sama.
* Top-3 accuracy: prediksi dianggap benar bila jawaban benar ada di tiga tebakan teratas. Berguna untuk 38 kelas yang mirip.
* Confusion matrix dan F1 per kelas: menunjukkan varietas mana yang sering tertukar, sebagai dasar analisis kesalahan.

### 5.2 Metrik Efisiensi 
* Jumlah parameter dan FLOPs: diukur sendiri pada input 224×224, bukan hanya dikutip dari makalah.
* Ukuran file model (MB): ukuran model setelah disimpan.
* Waktu inferensi (ms per citra): rata-rata waktu menebak satu citra setelah pemanasan (warm-up), pada perangkat dan pengaturan yang sama untuk kedua model (CPU dan GPU).
* Waktu pelatihan: total waktu melatih tiap model pada lingkungan yang sama (Google Colab atau Kaggle).

### 5.3 Uji Statistik 
* McNemar's test pada prediksi data uji kedua model, untuk menilai apakah selisih kesalahan keduanya nyata atau hanya kebetulan.
* Bootstrap 95% confidence interval untuk macro F1 tiap model, agar selisih kecil tidak ditafsirkan berlebihan.

### 5.4 Cara Membandingkan Model 
* Model terbaik dipilih berdasarkan macro F1 lebih dulu, lalu dipertimbangkan efisiensinya (parameter, FLOPs, ukuran, dan waktu inferensi).
* Hasil disajikan berdampingan untuk kedua model, lalu dibahas selisih akurasi dibanding selisih ukuran dan kecepatan.
* Analisis kesalahan: tampilkan pasangan varietas yang paling sering tertukar dan peta perhatian Grad-CAM untuk menjelaskan penyebab visualnya (misalnya kemiripan bentuk atau warna biji).

## 6. Rancangan Metode Singkat 

Dua model dilatih lewat transfer learning dari bobot ImageNet-1K dengan pipeline yang sama, lalu dibandingkan pada data uji yang sama.

### 6.1 Model 

### Perbandingan Model

| Model | Jenis | Parameter (±) | GFLOPs (±) | Input | Bobot Awal
|---|---|---|---|---|---|
| ConvNeXt-Tiny | CNN modern | 28,6 juta | 4,5 | 224x224 | ImageNet-1K
| DeiT-Small (patch16) | ViT efisien-data | 22,1 juta | 4,6 | 224x224 | ImageNet-1K | 

Ukuran kedua model sengaja dipilih sebanding agar perbandingan adil. Varian DeiT (biasa atau distilled) ditetapkan dan dilaporkan secara eksplisit di laporan akhir. Jumlah parameter dan FLOPs akan diukur ulang sendiri, bukan hanya dikutip.

### 6.2 Protokol Pelatihan (rencana awal, disesuaikan setelah uji coba) 

| Komponen | Rencana | 
|---|---|
| Inisialisasi | Pretrained ImageNet-1K | 
| Head | Diganti menjadi 38 Kelas | 
| Optimizer | AdamW |
| Learning rate | Dicari pada rentang kecil (misalnya 1e-4 sampai 5e-5) dengan scheduler cosine, dipilih berdasarkan validasi, bukan data uji |
| Loss | Cross-entropy (opsional smoothing |
| Augmentasi (data latih saja) | Random resized crop ringan, flip, rotasi 90°; tanpa perubahan warna ektrem karena warna adalah ciri varietas |
| Resize | 640x480 diubah ke 224x224; pilihan (squash, padding, atau crop) diuji di pilot dan dipakai sama untuk kedua model |
| Epoch | Ditetapkan lewat early stopping pada validasi |
| Seed | 3 seed berbeda per model |
| Lingkungan | Google Colab atau Kaggle (GPU) |

Kedua model memakai pipeline, augmentasi, optimizer, anggaran penyetelan hiperparameter, dan jumlah epoch maksimum yang sama.

### 6.3 Langkah Kerja 
1. Unduh dataset, periksa jumlah citra per kelas, dan buang duplikat.
2. Bagi data 70/15/15 secara stratified dengan seed tetap, sebelum augmentasi apa pun.
3. Praproses: ubah ukuran citra ke 224×224 dan normalisasi sesuai ImageNet. Augmentasi hanya pada data latih.
4. Latih ConvNeXt-Tiny dan DeiT-Small (fine-tuning penuh) dengan 3 seed masing-masing.
5. Evaluasi pada data uji memakai metrik di bagian 5, lalu analisis kesalahan dan Grad-CAM.

#### 6.4 Eksperimen Opsional 
* Pengaruh resolusi input (224 dibanding 384).
* Penambahan Swin Transformer sebagai pembanding ketiga (pengembangan, tidak mengubah judul).

## 7. Refrensi 
1. Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187.
2. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2201.03545.
3. Rajalakshmi, R., Faizal, S., Sivasankaran, S., & Geetha, R. (2024). RiceSeedNet: Rice seed variety identification using deep neural network. Journal of Agriculture and Food Research, 16, 101062.
4. Tahsin, M., Matin, M. M. H., Khandaker, M., Reemu, R. S., Arnab, M. I., Rashid, M. R. A., Rasel, M. M. K., Islam, M. M., Islam, M., & Ali, M. S. (2024). An extensive image dataset for deep learning-based classification of rice kernel varieties in Bangladesh. Data in Brief, 57, 111109. https://doi.org/10.1016/j.dib.2024.111109
5. Tahsin, M., dkk. (2024). An Extensive Image Dataset for Classifying Rice Varieties in Bangladesh [Data set]. Mendeley Data, V1. https://doi.org/10.17632/2fgv99854n.1
6. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). Training data-efficient image transformers & distillation through attention. Proceedings of the 38th International Conference on Machine Learning (ICML). arXiv:2012.12877.
7. Zhang, T., Zhang, C., Yang, Z., dkk. (2025). Multi-class rice seed recognition based on deep space and channel residual network combined with double attention mechanism. PLOS ONE. https://doi.org/10.1371/journal.pone.0322699
