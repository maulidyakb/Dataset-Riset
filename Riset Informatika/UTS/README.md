# Proposal Penelitian

## Analisis Perbandingan ConvNeXt dan Dataefficient Image Transformer (DeiT) untuk Klasifikasi Penyakit Daun Padi dengan Interpretasi Grad-CAM dan Uji Ketahanan terhadap Degradasi Citra

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Padi (*Oryza sativa* L.) adalah komoditas pangan utama Indonesia. Pada 2025, luas panen padi nasional mencapai sekitar 11,32 juta hektare dengan produksi 60,21 juta ton gabah kering giling (GKG), setara 34,69 juta ton beras untuk konsumsi penduduk [1]. Besarnya angka tersebut menunjukkan bahwa gangguan kecil pada produktivitas per hektare dapat berdampak besar pada ketahanan pangan nasional.

Salah satu ancaman utama produktivitas padi adalah serangan patogen dan hama. Survei pakar global memperkirakan kehilangan hasil padi akibat patogen dan hama rata-rata 30,0%, dengan rentang 24,6–40,9% antarwilayah [2]. Penyakit daun seperti hawar daun bakteri (*Xanthomonas oryzae* pv. *oryzae*), blas (*Pyricularia oryzae*), bercak cokelat (*Bipolaris oryzae*), dan tungro (virus yang ditularkan wereng hijau) termasuk penyakit penting di lahan sawah Asia, termasuk Indonesia.

Diagnosis penyakit di lapangan umumnya masih mengandalkan pengamatan visual oleh petani atau penyuluh. Cara ini lambat, bergantung pada pengalaman, dan rawan salah karena gejala beberapa penyakit mirip, misalnya bercak cokelat dan blas pada fase awal. Karena itu, klasifikasi otomatis berbasis citra daun menjadi alternatif yang banyak diteliti [3], [4].

Penelitian klasifikasi penyakit daun padi selama ini didominasi *Convolutional Neural Network* (CNN). Sethy et al. [5] membandingkan 13 model CNN pada 5.932 citra empat penyakit daun padi dan mendapatkan F1-score 0,9838 dengan fitur ResNet50 dan SVM. Sejak hadirnya *Vision Transformer* (ViT) [6], dua arah arsitektur berkembang bersamaan. ConvNeXt memodernisasi CNN murni dengan prinsip desain transformer dan menyaingi Swin Transformer pada ImageNet [7]. DeiT membuat transformer dapat dilatih efisien hanya dengan ImageNet-1K tanpa data eksternal berskala besar [8]. Keduanya berada pada skala komputasi sebanding (sekitar 4,5 GFLOPs untuk varian Tiny/Small), sehingga layak dibandingkan secara adil.

Namun, hasil perbandingan CNN dan transformer pada penyakit daun padi belum konsisten. Sebagian studi melaporkan transformer atau model hibrida lebih unggul, sementara studi lain menemukan ResNet50 tetap terbaik pada dataset padi tertentu [9], [10]. Selain itu, sebagian besar penelitian hanya melaporkan akurasi pada citra uji yang bersih. Padahal, citra yang diambil petani dengan ponsel di sawah sering mengalami *blur*, *noise* sensor, kompresi, dan pencahayaan yang tidak ideal. Hendrycks dan Dietterich [11] menunjukkan bahwa model dengan akurasi tinggi pada citra bersih dapat turun tajam pada citra yang terdegradasi.

Aspek kepercayaan juga penting. Model dengan akurasi tinggi belum tentu memusatkan perhatian pada lesi penyakit; model dapat bertumpu pada latar belakang atau artefak dataset. *Gradient-weighted Class Activation Mapping* (Grad-CAM) [12] dapat memvisualisasikan area citra yang memengaruhi keputusan model. Akan tetapi, pada penelitian penyakit daun padi Grad-CAM umumnya hanya ditampilkan secara kualitatif, tanpa diukur kesesuaiannya dengan area lesi dan tanpa diuji kestabilannya ketika citra terdegradasi.

Berdasarkan uraian tersebut, penelitian ini membandingkan ConvNeXt-Tiny dan DeiT-Small untuk klasifikasi empat penyakit daun padi pada dataset publik Sethy et al. [5]. Perbandingan dilakukan dari tiga sisi: kinerja pada citra bersih, ketahanan terhadap degradasi citra sintetis pada beberapa tingkat keparahan, dan kualitas interpretasi Grad-CAM yang diukur terhadap *mask* lesi. Hasilnya diharapkan memberi dasar pemilihan arsitektur yang tidak hanya akurat, tetapi juga tangguh dan dapat dipertanggungjawabkan untuk kondisi lapangan.

## 1.2 Rumusan Masalah

1. Bagaimana perbandingan kinerja ConvNeXt-Tiny dan DeiT-Small dalam mengklasifikasikan empat penyakit daun padi (hawar daun bakteri, blas, bercak cokelat, dan tungro) pada citra uji bersih, berdasarkan akurasi, presisi, *recall*, dan F1-score makro?
2. Bagaimana ketahanan kedua model terhadap degradasi citra berupa *noise*, *blur*, kompresi JPEG, dan perubahan pencahayaan pada lima tingkat keparahan?
3. Bagaimana kesesuaian area fokus Grad-CAM kedua model dengan area lesi penyakit, serta seberapa stabil peta Grad-CAM tersebut ketika citra mengalami degradasi?

## 1.3 Tujuan Penelitian

1. Mengukur dan membandingkan kinerja klasifikasi ConvNeXt-Tiny dan DeiT-Small pada citra uji bersih menggunakan akurasi, presisi, *recall*, dan F1-score makro, disertai uji signifikansi McNemar.
2. Mengukur penurunan kinerja kedua model pada citra terdegradasi dan menentukan model yang lebih tangguh menggunakan *mean corruption* F1 dan *relative robustness*.
3. Mengevaluasi interpretasi Grad-CAM secara kuantitatif terhadap *mask* lesi (*energy-based pointing game* dan IoU) serta mengukur stabilitas peta Grad-CAM antara citra bersih dan citra terdegradasi.

## 1.4 Manfaat Penelitian

**Manfaat teoretis.** Penelitian ini menambah bukti empiris tentang perbedaan bias induktif CNN modern dan transformer pada citra pertanian, khususnya dalam hal ketahanan terhadap degradasi dan perilaku interpretasinya. Protokol evaluasi yang menggabungkan uji ketahanan dan evaluasi Grad-CAM kuantitatif dapat dipakai ulang untuk komoditas lain.

**Manfaat praktis.** Hasil penelitian dapat menjadi acuan bagi pengembang sistem diagnosis penyakit tanaman berbasis ponsel dalam memilih arsitektur yang tetap andal pada foto lapangan berkualitas rendah. Bagi penyuluh pertanian, visualisasi Grad-CAM yang sesuai dengan lesi dapat meningkatkan kepercayaan terhadap hasil prediksi model.

## 1.5 Batasan Masalah

1. Dataset yang digunakan adalah *Rice Leaf Disease Image Samples* [5] sebanyak 5.932 citra dengan empat kelas penyakit; kelas daun sehat tidak tercakup karena tidak tersedia pada dataset tersebut.
2. Model yang dibandingkan adalah ConvNeXt-Tiny dan DeiT-Small (tanpa token distilasi) dengan bobot *pretrained* ImageNet-1K dan resolusi masukan 224 × 224 piksel.
3. Degradasi citra bersifat sintetis dan hanya diterapkan pada data uji, meliputi Gaussian *noise*, *defocus blur*, *motion blur*, kompresi JPEG, peningkatan kecerahan, dan kondisi gelap, masing-masing lima tingkat keparahan.
4. Metode interpretasi yang digunakan hanya Grad-CAM; evaluasi kuantitatifnya memakai *mask* segmentasi lesi RiceSeg-5932.
5. Penelitian berfokus pada pemodelan dan evaluasi *machine learning*, tidak mencakup pembuatan aplikasi atau implementasi pada perangkat lapangan.

# BAB II TINJAUAN PUSTAKA

## 2.1 Penelitian Terdahulu

Penelitian terdahulu menunjukkan akurasi klasifikasi penyakit daun padi sudah tinggi (umumnya di atas 90%), tetapi hampir seluruhnya diuji pada citra bersih dan hanya menampilkan Grad-CAM secara visual. Tabel 2.1 merangkum studi yang paling relevan.

**Tabel 2.1** Ringkasan penelitian terdahulu

| No | Peneliti | Dataset | Metode | Hasil Utama | Keterbatasan terhadap Penelitian Ini |
| --- | --- | --- | --- | --- | --- |
| 1 | Lu et al. [13] | 500 citra, 10 penyakit padi | CNN dilatih dari awal | Akurasi 95,48% | Dataset kecil; tanpa uji ketahanan dan interpretasi |
| 2 | Sethy et al. [5] | 5.932 citra, 4 penyakit daun padi (Odisha, India) | 13 CNN; *transfer learning* dan fitur CNN + SVM | ResNet50 + SVM terbaik, F1 0,9838 | Hanya CNN; citra uji bersih; tanpa Grad-CAM |
| 3 | Rahman et al. [14] | 1.426 citra, 9 kelas penyakit dan hama padi (Bangladesh) | VGG16, InceptionV3, *Simple CNN* dua tahap | CNN ringan dua tahap mencapai akurasi 93,3% | Tanpa transformer; tanpa uji degradasi |
| 4 | Wang et al. [15] | 4.523 citra, 7 kelas (6 penyakit + sehat) | ECA-ConvNeXt (ConvNeXt + *channel attention*) | Akurasi 94,82% (ConvNeXt tanpa ECA 94,12%) | Tanpa pembanding transformer, uji ketahanan, dan Grad-CAM |
| 5 | Petchiammal et al. [16] | Paddy Doctor: 16.225 citra, 13 kelas | ResNet34, VGG16, MobileNet, Xception, dll. | ResNet34 terbaik, F1 97,50% | Fokus *benchmark* akurasi pada citra bersih |
| 6 | Mehnaz dan Islam [10] | Dhan-Shomadhan, 5 penyakit padi | ResNet50, ResNet152, InceptionV3, MaxViT-L, SVM | ResNet50 91,50% vs MaxViT 75,00% | Menunjukkan transformer tidak selalu unggul; tanpa XAI dan uji degradasi |
| 7 | Al-Falluji dan Albahar [9] | 4.684 citra (3 kelas) dan 10.165 citra (5 kelas) | ResNet50, ViT, *Hybrid* ConvNeXt + XAI | ViT dan *Hybrid* ConvNeXt 100% pada dataset 3 kelas | Evaluasi hanya pada citra bersih; XAI kualitatif |
| 8 | Rahaman et al. [17] | 11.790 citra (9 kelas) dan 11.250 citra (12 kelas) | Fusi dua aliran Swin Transformer + ConvNeXt, Grad-CAM | Akurasi 99,55% dan 97,06%; lintas dataset 90,80% | Grad-CAM hanya visual tanpa metrik lokalisasi; tanpa uji degradasi |
| 9 | de Andrade et al. [18] | MangoLeafDB-C: 19 jenis korupsi × 5 tingkat | ResNet-50/101, VGG-16, Xception, LCNN | LCNN paling tangguh (mCE terendah); ResNet-101 turun signifikan | Komoditas mangga; tanpa transformer |

Dari tabel tersebut terlihat tiga pola. Pertama, studi padi berfokus pada akurasi citra bersih. Kedua, perbandingan CNN dan transformer memberi hasil yang berlawanan antarstudi (no. 6 dan 7). Ketiga, uji ketahanan terhadap korupsi citra baru dilakukan pada komoditas lain (no. 9), belum pada penyakit daun padi.

## 2.2 Landasan Teori

### 2.2.1 Penyakit Daun Padi

Empat penyakit yang menjadi kelas pada penelitian ini memiliki penyebab dan gejala visual yang berbeda, tetapi sebagian gejalanya mirip pada fase awal. Kemiripan inilah yang membuat klasifikasi berbasis citra menantang.

**Tabel 2.2** Karakteristik visual penyakit daun padi

| Penyakit | Penyebab | Gejala Visual pada Daun |
| --- | --- | --- |
| Hawar daun bakteri (*bacterial blight*) | Bakteri *Xanthomonas oryzae* pv. *oryzae* | Lesi memanjang dari ujung atau tepi daun, kuning lalu putih keabu-abuan dan mengering |
| Blas (*blast*) | Jamur *Pyricularia oryzae* | Bercak berbentuk belah ketupat/elips, pusat abu-abu keputihan dengan tepi cokelat |
| Bercak cokelat (*brown spot*) | Jamur *Bipolaris oryzae* | Bercak oval kecil berwarna cokelat, sering dengan pusat keabu-abuan, tersebar di permukaan daun |
| Tungro | Virus RTBV dan RTSV, ditularkan wereng hijau (*Nephotettix virescens*) | Daun menguning hingga oranye dari ujung, tanaman kerdil, jumlah anakan berkurang |

### 2.2.2 Citra Digital dan Degradasi Citra

Citra digital berwarna adalah matriks piksel tiga kanal (R, G, B) dengan intensitas 0–255. Citra yang diambil di lapangan jarang ideal: kamera ponsel dapat menghasilkan *noise* sensor pada cahaya rendah, *blur* karena fokus meleset atau tangan bergerak, artefak blok akibat kompresi JPEG, serta paparan cahaya berlebih atau kurang. Secara umum, degradasi dimodelkan sebagai berikut, dengan *f* citra asli, *h* fungsi sebaran titik (*point spread function*) penyebab *blur*, dan *n* *noise* aditif.

```text
g(x, y) = h(x, y) ∗ f(x, y) + n(x, y)
```

Uji ketahanan model dilakukan dengan menerapkan degradasi terkontrol pada citra uji. Hendrycks dan Dietterich [11] membakukan pendekatan ini melalui ImageNet-C, yaitu 15 jenis korupsi dalam kelompok *noise*, *blur*, cuaca, dan digital, masing-masing dengan lima tingkat keparahan.

### 2.2.3 Convolutional Neural Network (CNN)

CNN mengekstraksi fitur citra melalui lapisan konvolusi yang menggeser filter kecil pada citra. Dua bias induktif utamanya adalah lokalitas (setiap neuron melihat area kecil) dan *translation equivariance* (pola yang sama dikenali di posisi mana pun). Bias ini membuat CNN efisien data, tetapi CNN yang dilatih pada ImageNet cenderung mengandalkan tekstur daripada bentuk [19]. Kecenderungan ini relevan karena *blur* dan *noise* justru merusak informasi tekstur.

### 2.2.4 ConvNeXt

ConvNeXt adalah CNN murni yang dirancang ulang secara bertahap dari ResNet dengan meniru keputusan desain Swin Transformer [7]. Perubahan utamanya: *stem* "patchify" 4 × 4 dengan *stride* 4, rasio blok per tahap 3:3:9:3, konvolusi *depthwise* berukuran 7 × 7, blok *inverted bottleneck*, fungsi aktivasi GELU, dan *Layer Normalization* menggantikan *Batch Normalization*. ConvNeXt-Tiny memiliki sekitar 28 juta parameter dan 4,5 GFLOPs, dengan akurasi top-1 ImageNet-1K 82,1%.

### 2.2.5 Transformer dan Vision Transformer (ViT)

ViT memotong citra menjadi *patch* 16 × 16 piksel, memproyeksikan tiap *patch* menjadi vektor, lalu menambahkan *position embedding* dan token kelas [CLS] [6]. Urutan token diproses oleh lapisan *multi-head self-attention* (MHSA), sehingga setiap *patch* dapat berinteraksi dengan seluruh *patch* lain sejak lapisan pertama. Inti mekanisme tersebut adalah *scaled dot-product attention*:

```text
Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
```

Karena tidak memiliki bias lokalitas bawaan, ViT umumnya membutuhkan data pralatih sangat besar. Beberapa studi melaporkan transformer lebih tahan terhadap sebagian jenis korupsi dibanding CNN klasik [20], [21], meskipun hasilnya bergantung pada jenis korupsi dan resep pelatihan.

### 2.2.6 Data-efficient Image Transformer (DeiT)

DeiT menggunakan arsitektur ViT yang sama, tetapi dengan resep pelatihan yang memungkinkan model dilatih hanya pada ImageNet-1K: augmentasi kuat (RandAugment, Mixup, CutMix, *random erasing*), regularisasi, dan opsi distilasi melalui token tambahan [8]. DeiT-Small memiliki sekitar 22 juta parameter dan 4,6 GFLOPs dengan akurasi top-1 ImageNet-1K 79,8%. Skala komputasinya sebanding dengan ConvNeXt-Tiny, sehingga keduanya dapat dibandingkan secara adil. Penelitian ini memakai varian tanpa token distilasi agar perbandingan murni antara arsitektur konvolusi dan *attention*.

### 2.2.7 Transfer Learning dan Fine-tuning

*Transfer learning* memanfaatkan bobot model yang telah dilatih pada dataset besar (ImageNet) sebagai titik awal untuk tugas baru dengan data terbatas. Pada *fine-tuning*, lapisan klasifikasi akhir diganti sesuai jumlah kelas target, lalu seluruh bobot dilatih ulang dengan laju pembelajaran kecil. Pendekatan ini terbukti efektif pada klasifikasi penyakit tanaman [3], [22].

### 2.2.8 Gradient-weighted Class Activation Mapping (Grad-CAM)

Grad-CAM menghasilkan peta panas yang menunjukkan area citra yang paling memengaruhi prediksi kelas *c* [12]. Bobot setiap kanal fitur *k* dihitung dari rata-rata gradien skor kelas terhadap peta fitur *Aᵏ*, lalu peta fitur dijumlahkan berbobot dan dilewatkan ReLU:

```text
αₖᶜ = (1/Z) · Σᵢ Σⱼ ∂yᶜ / ∂Aᵏᵢⱼ
L_Grad-CAMᶜ = ReLU( Σₖ αₖᶜ · Aᵏ )
```

Pada ConvNeXt, peta fitur diambil dari blok terakhir tahap keempat. Pada DeiT, keluaran token *patch* blok transformer terakhir (tanpa token [CLS]) disusun ulang menjadi grid 14 × 14 agar dapat diperlakukan seperti peta fitur konvolusi. Karena peta saliensi dapat tampak meyakinkan meskipun tidak setia pada model [23], penelitian ini mengevaluasi Grad-CAM secara kuantitatif, bukan hanya visual.

## 2.3 Research Gap

Celah utama yang diisi penelitian ini adalah belum adanya evaluasi CNN modern dan transformer pada penyakit daun padi yang sekaligus mengukur ketahanan terhadap degradasi citra dan kualitas Grad-CAM secara kuantitatif. Tabel 2.3 merinci empat celah dan kontribusi yang ditawarkan.

**Tabel 2.3** Research gap dan kontribusi penelitian

| Kode | Celah Penelitian | Bukti dari Literatur | Kontribusi Penelitian Ini |
| --- | --- | --- | --- |
| G1 | Perbandingan CNN dan transformer pada padi tidak setara skala dan hasilnya kontradiktif | ResNet50 mengalahkan MaxViT-L [10], sedangkan ViT mencapai 100% [9]; model yang dibandingkan berbeda jauh ukurannya | Membandingkan ConvNeXt-Tiny (±28 juta parameter, 4,5 GFLOPs) dan DeiT-Small (±22 juta, 4,6 GFLOPs) dengan data, praproses, dan resep pelatihan yang sama, diulang tiga *seed* dan diuji McNemar |
| G2 | Evaluasi hanya pada citra uji bersih | Studi no. 1–8 pada Tabel 2.1 tidak menguji korupsi citra; uji korupsi multi-tingkat baru ada pada daun mangga [18] | Uji ketahanan pada 6 jenis degradasi × 5 tingkat keparahan (30 kondisi) dengan metrik *mean corruption* F1 dan *relative robustness* |
| G3 | Grad-CAM hanya ditampilkan secara visual | Rahaman et al. [17] menyatakan tidak ada metrik lokalisasi kuantitatif; *mask* lesi piksel untuk dataset Sethy baru tersedia melalui RiceSeg-5932 [24] | Mengukur kesesuaian Grad-CAM dengan *mask* lesi memakai *energy-based pointing game* dan IoU |
| G4 | Hubungan ketahanan prediksi dan stabilitas penjelasan belum diteliti | Belum ditemukan studi penyakit daun padi yang membandingkan Grad-CAM citra bersih dan citra terdegradasi | Mengukur stabilitas Grad-CAM (korelasi Pearson dan SSIM antara peta bersih dan terdegradasi) untuk melihat apakah model tetap "melihat" lesi saat kualitas citra turun |

Dengan demikian, kebaruan penelitian ini terletak pada protokol evaluasi tiga sisi — akurasi, ketahanan, dan interpretasi kuantitatif — yang diterapkan secara setara pada dua keluarga arsitektur.

## 2.4 Kerangka Pemikiran

<img width="714" height="421" alt="kerangka-pemikiran riset" src="https://github.com/user-attachments/assets/e2c3e7ad-0e29-448b-b3f9-07263e261762" />

**Gambar 2.1** Kerangka pemikiran penelitian. Masalah lapangan melahirkan empat celah (G1–G4 pada Tabel 2.3); G1 dijawab perbandingan setara, G2 dijawab uji ketahanan, sedangkan G3 dan G4 dijawab evaluasi Grad-CAM kuantitatif. Ketiganya bermuara pada satu rekomendasi arsitektur.

# BAB III METODOLOGI PENELITIAN

## 3.1 Pendekatan Penelitian

Penelitian ini adalah penelitian kuantitatif eksperimental dengan desain komparatif. Variabel bebasnya adalah arsitektur model (ConvNeXt-Tiny dan DeiT-Small) serta jenis dan tingkat degradasi citra. Variabel terikatnya adalah metrik klasifikasi, metrik ketahanan, dan metrik interpretasi Grad-CAM. Seluruh faktor lain — dataset, pembagian data, praproses, augmentasi, *optimizer*, jumlah *epoch*, dan *seed* — dibuat sama untuk kedua model agar perbedaan hasil dapat diatribusikan pada arsitektur.

## 3.2 Alur Penelitian

<img width="713" height="493" alt="alur-penelitian" src="https://github.com/user-attachments/assets/801ff665-84ac-4691-93e0-fcf6aa99c322" />

**Gambar 3.1** Alur penelitian. Tahap 1–3 menyiapkan data yang sama untuk kedua model; tahap 4 melatih ConvNeXt-Tiny dan DeiT-Small secara paralel; tahap 5–6 menguji *checkpoint* terbaik dari tiga sisi; tahap 7 menyimpulkan model yang paling layak untuk kondisi lapangan.

## 3.3 Dataset

Penelitian menggunakan dataset publik *Rice Leaf Disease Image Samples* [5], [25] yang tersedia di [Mendeley Data](https://data.mendeley.com/datasets/fwcj7stb8r) (DOI 10.17632/fwcj7stb8r). Dataset berisi 5.932 citra daun padi berukuran 300 × 300 piksel yang diambil di lahan sawah Odisha, India. Dataset ini dipilih karena terpublikasi di jurnal bereputasi (*Computers and Electronics in Agriculture*), banyak dipakai sebagai pembanding, dan memiliki *mask* segmentasi lesi resmi tingkat piksel ([RiceSeg-5932](https://data.mendeley.com/datasets/92jc6w6mcy/1), DOI 10.17632/92jc6w6mcy.1) [24] yang dibutuhkan untuk evaluasi Grad-CAM.

**Tabel 3.1** Distribusi kelas dan rencana pembagian data (stratifikasi 70:15:15)

| Kelas | Jumlah Citra | Proporsi | Latih | Validasi | Uji |
| --- | --- | --- | --- | --- | --- |
| Hawar daun bakteri (*Bacterial blight*) | 1.584 | 26,7% | 1.108 | 238 | 238 |
| Blas (*Blast*) | 1.440 | 24,3% | 1.008 | 216 | 216 |
| Bercak cokelat (*Brown spot*) | 1.600 | 27,0% | 1.120 | 240 | 240 |
| Tungro | 1.308 | 22,0% | 916 | 196 | 196 |
| **Total** | **5.932** | **100%** | **4.152** | **890** | **890** |

Distribusi kelas relatif seimbang (rasio kelas terbesar dan terkecil 1,22), sehingga tidak diperlukan *oversampling*. Angka pembagian dapat sedikit bergeser setelah tahap pembersihan duplikat pada subbab 3.4.

## 3.4 Praproses, Pembagian Data, dan Augmentasi

1. **Pemeriksaan kualitas data.** Citra rusak atau tidak dapat dibaca dihapus. Kemiripan antarcitra diperiksa dengan *perceptual hash* (pHash); pasangan dengan jarak Hamming kecil dianggap hampir duplikat dan dikelompokkan.
2. **Pembagian data.** Data dibagi secara stratifikasi 70% latih, 15% validasi, dan 15% uji. Satu kelompok hampir duplikat selalu ditempatkan pada subset yang sama untuk mencegah kebocoran data (*data leakage*).
3. **Penyesuaian ukuran dan normalisasi.** Citra diubah ukurannya menjadi 224 × 224 piksel dan dinormalisasi dengan rerata dan simpangan baku ImageNet, sesuai bobot *pretrained* kedua model.
4. **Augmentasi (hanya data latih).** *Random resized crop* (skala 0,8–1,0), *flip* horizontal dan vertikal, rotasi ±15°, dan *color jitter* ringan (kecerahan dan kontras ±10%).
5. **Pemisahan augmentasi dan uji ketahanan.** *Blur*, *noise*, dan kompresi JPEG sengaja tidak dipakai sebagai augmentasi, agar uji ketahanan pada subbab 3.7 benar-benar mengukur kemampuan model menghadapi degradasi yang belum pernah dilihat saat pelatihan.

## 3.5 Arsitektur Model dan Konfigurasi Pelatihan

Kedua model dimuat dari pustaka `timm` dengan bobot ImageNet-1K, lalu lapisan klasifikasi akhirnya diganti dengan lapisan linear 4 keluaran. Seluruh lapisan dilatih ulang (*full fine-tuning*). Laju pembelajaran dipilih per model dari tiga kandidat berdasarkan F1 makro validasi, karena CNN dan transformer dikenal memiliki laju optimal yang berbeda; parameter lain dibuat identik.

**Tabel 3.2** Konfigurasi model dan pelatihan

| Parameter | ConvNeXt-Tiny | DeiT-Small |
| --- | --- | --- |
| Bobot *pretrained* (`timm`) | `convnext_tiny.fb_in1k` | `deit_small_patch16_224.fb_in1k` |
| Jumlah parameter / GFLOPs | ±28 juta / 4,5 | ±22 juta / 4,6 |
| Ukuran masukan | 224 × 224 | 224 × 224 (196 *patch* 16 × 16) |
| *Optimizer* | AdamW, *weight decay* 0,05 | AdamW, *weight decay* 0,05 |
| Kandidat laju pembelajaran | 5e-5; 1e-4; 3e-4 | 5e-5; 1e-4; 3e-4 |
| Penjadwal laju | *Cosine annealing*, *warm-up* linear 5 *epoch* | sama |
| *Batch size* | 32 | 32 |
| *Epoch* maksimum | 50, *early stopping* 10 *epoch* (F1 makro validasi) | sama |
| Fungsi *loss* | *Cross-entropy* dengan *label smoothing* 0,1 | sama |
| Regularisasi | *Stochastic depth* 0,1 | *Stochastic depth* 0,1 |
| Presisi | *Mixed precision* (AMP) | sama |
| *Seed* | 42, 123, 2026 | 42, 123, 2026 |

## 3.6 Skenario Eksperimen

1. **Skenario 1 — Klasifikasi citra bersih.** Setiap model dilatih tiga kali dengan *seed* berbeda. *Checkpoint* terbaik berdasarkan F1 makro validasi diuji pada 890 citra uji bersih. Hasil dilaporkan sebagai rerata ± simpangan baku, dan prediksi kedua model dibandingkan dengan uji McNemar.
2. **Skenario 2 — Uji ketahanan.** *Checkpoint* yang sama diuji tanpa pelatihan ulang pada 30 versi data uji terdegradasi (6 jenis × 5 tingkat; subbab 3.7).
3. **Skenario 3 — Evaluasi Grad-CAM.** Peta Grad-CAM dihitung pada citra uji bersih dan dibandingkan dengan *mask* lesi RiceSeg-5932. Stabilitas peta diukur pada citra terdegradasi tingkat 1, 3, dan 5 (subbab 3.8).
4. **Skenario 4 — Efisiensi (pelengkap).** Jumlah parameter, GFLOPs, dan waktu inferensi per citra pada GPU dan CPU dicatat sebagai pertimbangan penerapan.

## 3.7 Uji Ketahanan terhadap Degradasi Citra

Uji ketahanan memakai enam jenis degradasi yang mewakili masalah umum foto ponsel di sawah, masing-masing dengan lima tingkat keparahan. Degradasi diterapkan setelah citra diubah ke 224 × 224 piksel, sehingga parameter ImageNet-C [11] tetap berlaku. Lima jenis memakai implementasi pustaka `imagecorruptions` [26]; kondisi gelap diimplementasikan sendiri dengan koreksi gamma.

**Tabel 3.3** Jenis dan parameter degradasi citra

| Jenis Degradasi | Kelompok | Parameter Tingkat 1 → 5 | Kondisi Lapangan yang Diwakili |
| --- | --- | --- | --- |
| Gaussian *noise* | *Noise* | σ = 0,08; 0,12; 0,18; 0,26; 0,38 | *Noise* sensor kamera saat cahaya kurang |
| *Defocus blur* | *Blur* | radius = 3; 4; 6; 8; 10 | Fokus kamera meleset |
| *Motion blur* | *Blur* | (radius, σ) = (10,3); (15,5); (15,8); (15,12); (20,15) | Tangan atau daun bergerak tertiup angin |
| Kompresi JPEG | Digital | kualitas = 25; 18; 15; 10; 7 | Foto dikompres saat dikirim lewat aplikasi pesan |
| Peningkatan kecerahan | Pencahayaan | +0,1; +0,2; +0,3; +0,4; +0,5 (kanal V pada HSV) | Terik matahari, *overexposure* |
| Kondisi gelap | Pencahayaan | γ = 1,5; 2,0; 2,5; 3,0; 3,5 | Pagi/sore hari, area teduh |

Koreksi gamma untuk kondisi gelap dirumuskan sebagai berikut, dengan *I* intensitas piksel ternormalisasi 0–1; nilai γ > 1 membuat citra lebih gelap.

```text
I_gelap = I^γ ,   γ ∈ {1,5; 2,0; 2,5; 3,0; 3,5}
```

Setiap kombinasi jenis dan tingkat menghasilkan satu salinan data uji (890 citra), sehingga total terdapat 30 × 890 = 26.700 citra uji terdegradasi per model per *seed*. Seluruh salinan dibuat sekali dengan *seed* tetap dan disimpan, agar kedua model diuji pada citra yang persis sama.

## 3.8 Interpretasi dan Evaluasi Grad-CAM

Grad-CAM dihitung dengan pustaka `pytorch-grad-cam` untuk kelas yang diprediksi model. Lapisan target ConvNeXt-Tiny adalah blok terakhir tahap keempat (`stages[-1]`). Lapisan target DeiT-Small adalah `norm1` pada blok transformer terakhir, dengan *reshape transform* yang membuang token [CLS] dan menyusun 196 token *patch* menjadi grid 14 × 14. Peta panas diperbesar ke 224 × 224 dan dinormalisasi min–maks ke rentang 0–1. *Mask* RiceSeg-5932 diperkecil ke 224 × 224 dengan interpolasi *nearest neighbor* agar tetap biner.

Evaluasi dilakukan dalam tiga bagian.

1. **Lokalisasi — *energy-based pointing game*.** Proporsi energi peta Grad-CAM *L* yang jatuh di dalam *mask* lesi *M* [27]. Nilai mendekati 1 berarti perhatian model terpusat pada lesi.
2. **Lokalisasi — IoU.** Peta Grad-CAM dibinerkan pada ambang 0,5, lalu dihitung *Intersection over Union* dengan *mask* lesi.
3. **Stabilitas di bawah degradasi.** Peta Grad-CAM citra bersih dibandingkan dengan peta citra terdegradasi tingkat 1, 3, dan 5 menggunakan korelasi Pearson dan SSIM. Nilai tinggi berarti model tetap menyoroti area yang sama meskipun kualitas citra turun.

```text
Proporsi Energi = Σ₍ᵢ,ⱼ₎∈M Lᵢⱼ / Σ₍ᵢ,ⱼ₎ Lᵢⱼ
IoU = |B ∩ M| / |B ∪ M|
```

Pada rumus di atas, *B* adalah peta Grad-CAM yang telah dibinerkan. Metrik lokalisasi dihitung pada seluruh citra uji dan dilaporkan terpisah untuk prediksi benar dan salah. Sebagai pelengkap, contoh visual per kelas — termasuk kasus salah klasifikasi — dianalisis secara kualitatif.

## 3.9 Rencana Metrik Evaluasi

Metrik utama penelitian ini adalah F1-score makro, karena metrik ini memberi bobot sama pada setiap kelas dan menyeimbangkan presisi serta *recall* [28]. Tabel 3.4 merangkum seluruh metrik sesuai rumusan masalah.

**Tabel 3.4** Rencana metrik evaluasi

| Aspek (Rumusan Masalah) | Metrik | Keterangan |
| --- | --- | --- |
| Klasifikasi citra bersih (RM 1) | Akurasi, presisi makro, *recall* makro, **F1 makro**, *confusion matrix* | Dilaporkan rerata ± simpangan baku dari 3 *seed*; per kelas untuk analisis kesalahan |
| Signifikansi perbedaan (RM 1) | Uji McNemar, α = 0,05 | Membandingkan prediksi kedua model pada citra uji yang sama [29] |
| Ketahanan (RM 2) | F1 per jenis dan tingkat, *mean corruption* F1 (mCF1), *relative robustness* (RR), penurunan F1 (ΔF1) | RR mendekati 1 berarti performa hampir tidak turun |
| Interpretasi (RM 3) | Proporsi energi (*energy-based pointing game*), IoU | Kesesuaian Grad-CAM dengan *mask* lesi |
| Stabilitas interpretasi (RM 3) | Korelasi Pearson, SSIM | Kemiripan peta Grad-CAM bersih dan terdegradasi |
| Efisiensi (pelengkap) | Parameter, GFLOPs, waktu inferensi (ms/citra) | Diukur pada GPU dan CPU |

Rumus metrik klasifikasi per kelas *k* (dengan *K* = 4 kelas) dan rerata makronya adalah:

```text
Pₖ = TPₖ / (TPₖ + FPₖ)
Rₖ = TPₖ / (TPₖ + FNₖ)
F1ₖ = (2 · Pₖ · Rₖ) / (Pₖ + Rₖ)
F1_makro = (1/K) · Σₖ F1ₖ
```

Metrik ketahanan dihitung dari F1 makro pada setiap jenis degradasi *c* (*C* = 6) dan tingkat *s* (*S* = 5):

```text
mCF1 = (1 / (C · S)) · Σ_c Σ_s F1_c,s
RR = mCF1 / F1_bersih
ΔF1_c,s = F1_bersih − F1_c,s
```

Uji McNemar memakai *b* (citra yang benar oleh ConvNeXt tetapi salah oleh DeiT) dan *c* (sebaliknya), dengan koreksi kontinuitas:

```text
χ² = ( |b − c| − 1 )² / (b + c)
```

## 3.10 Lingkungan Penelitian

| Komponen | Spesifikasi |
| --- | --- |
| Perangkat keras | GPU NVIDIA T4 16 GB (Google Colab/Kaggle) atau setara; CPU untuk pengukuran waktu inferensi |
| Bahasa dan *framework* | Python 3.11, PyTorch 2.x, `timm` |
| Pustaka pendukung | `pytorch-grad-cam`, `imagecorruptions`, `scikit-learn`, `scikit-image` (SSIM), `ImageHash`, `fvcore` (GFLOPs) |
| Manajemen kode | GitHub (folder Riset Informatika/UTS) |

# DAFTAR PUSTAKA

[1] Badan Pusat Statistik, "Luas panen padi pada tahun 2025 mencapai sekitar 11,32 juta hektare dengan produksi padi sebanyak 60,21 juta ton gabah kering giling (GKG)," *Berita Resmi Statistik*, 2 Feb. 2026. [Online]. Available: <https://www.bps.go.id/id/pressrelease/2026/02/02/2545/>

[2] S. Savary, L. Willocquet, S. J. Pethybridge, P. Esker, N. McRoberts, and A. Nelson, "The global burden of pathogens and pests on major food crops," *Nat. Ecol. Evol.*, vol. 3, no. 3, pp. 430–439, 2019, doi: 10.1038/s41559-018-0793-y.

[3] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Front. Plant Sci.*, vol. 7, Art. no. 1419, 2016.

[4] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Comput. Electron. Agric.*, vol. 145, pp. 311–318, 2018.

[5] P. K. Sethy, N. K. Barpanda, A. K. Rath, and S. K. Behera, "Deep feature based rice leaf disease identification using support vector machine," *Comput. Electron. Agric.*, vol. 175, Art. no. 105527, 2020, doi: 10.1016/j.compag.2020.105527.

[6] A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[7] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.

[8] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training data-efficient image transformers & distillation through attention," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 139, 2021, pp. 10347–10357.

[9] R. A. Al-Falluji and M. A. Albahar, "Enhancing green AI through explainable deep learning-based multi-model for automated rice leaf disease classification," *Discover Comput.*, vol. 28, no. 1, 2025, doi: 10.1007/s10791-025-09799-5.

[10] S. Mehnaz and M. T. Islam, "Rice leaf disease detection: A comparative study between CNN, transformer and non-neural network architectures," 2025, *arXiv:2501.06740*.

[11] D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and perturbations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[12] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[13] Y. Lu, S. Yi, N. Zeng, Y. Liu, and Y. Zhang, "Identification of rice diseases using deep convolutional neural networks," *Neurocomputing*, vol. 267, pp. 378–384, 2017.

[14] C. R. Rahman *et al.*, "Identification and recognition of rice diseases and pests using convolutional neural networks," *Biosyst. Eng.*, vol. 194, pp. 112–120, 2020.

[15] X. Wang, Y. Wang, J. Zhao, and J. Niu, "ECA-ConvNeXt: A rice leaf disease identification model based on ConvNeXt," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2023.

[16] Petchiammal, B. Kiruba, Murugan, and P. Arjunan, "Paddy Doctor: A visual image dataset for automated paddy disease classification and benchmarking," in *Proc. 6th Joint Int. Conf. Data Sci. Manage. Data (CODS-COMAD)*, 2023, pp. 203–207.

[17] M. N. Rahaman *et al.*, "Global-local feature-based rice leaf disease classification using two-stream deep neural network feature fusion," *Computers*, vol. 15, no. 8, Art. no. 528, 2026, doi: 10.3390/computers15080528.

[18] G. V. de Andrade, S. R. dos Santos, I. P. C. A. da Silva, E. A. M. Pereira, and E. de A. Barboza, "From code to field: Evaluating the robustness of convolutional neural networks for disease diagnosis in mango leaves," 2025, *arXiv:2512.13641*.

[19] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel, "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[20] S. Bhojanapalli, A. Chakrabarti, D. Glasner, D. Li, T. Unterthiner, and A. Veit, "Understanding robustness of transformers for image classification," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10231–10241.

[21] S. Paul and P.-Y. Chen, "Vision transformers are robust learners," in *Proc. AAAI Conf. Artif. Intell.*, vol. 36, no. 2, 2022, pp. 2071–2081.

[22] J. G. A. Barbedo, "Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification," *Comput. Electron. Agric.*, vol. 153, pp. 46–53, 2018.

[23] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018.

[24] K. M. S. Zinsou and V. R. Houndji, "RiceSeg-5932: Complete pixel-level segmentation masks for rice leaf disease images samples," Mendeley Data, V1, 2025, doi: 10.17632/92jc6w6mcy.1.

[25] P. K. Sethy, "Rice leaf disease image samples," Mendeley Data, V1, 2020, doi: 10.17632/fwcj7stb8r.1.

[26] C. Michaelis *et al.*, "Benchmarking robustness in object detection: Autonomous driving when winter is coming," 2019, *arXiv:1907.07484*.

[27] H. Wang *et al.*, "Score-CAM: Score-weighted visual explanations for convolutional neural networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 24–25.

[28] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manage.*, vol. 45, no. 4, pp. 427–437, 2009.

[29] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998.
