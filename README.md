## 1. Rencana Topik
Klasifikasi Makanan Indonesia dengan Transfer Learning Ringan: Perbandingan Akurasi, Ukuran Model, dan Kecepatan Inferensi. Penelitian ini akan membandingkan beberapa arsitektur CNN ringan (MobileNetV2, MobileNetV3, EfficientNet-B0) pada dataset citra makanan Indonesia untuk menentukan model yang paling seimbang antara akurasi dan efisiensi bagi aplikasi mobile.
## 2. Formulasi Masalah
- Seberapa baik MobileNetV2, MobileNetV3, dan EfficientNet-B0 mengklasifikasikan citra makanan Indonesia?
- Bagaimana trade-off antara akurasi, jumlah parameter, ukuran file model, dan waktu inferensi (CPU), sebelum dan sesudah konversi ke TensorFlow Lite?
- Pasangan makanan mana yang paling sering tertukar, dan apa penyebab visualnya?
## 3. Riset Gap Penelitian Terdahulu
- Rasyidi dkk. (2024) membangun dataset 24.427 gambar dari 160 jenis makanan Indonesia dan membandingkan 67 model dari 16 arsitektur. Akurasi terbaik 85,44% dicapai EfficientNetV2-L yang berukuran sekitar 118,5 juta parameter. Penelitian ini membahas *trade-off* antara ukuran model dan akurasi, tetapi hasilnya belum melaporkan kecepatan inferensi maupun ukuran model untuk perangkat mobile.
- Wibisono dkk. (2020) membuat dataset 34 jenis makanan tradisional (1.644 gambar) yang difoto di studio dengan kondisi terkontrol. Citra yang seragam membuat model berisiko kurang mampu mengenali foto dengan pencahayaan dan latar yang beragam (hal ini juga disebut oleh Rasyidi dkk.).
- Benarkah dkk. (2025) mengembangkan aplikasi mobile pengenal jajanan tradisional (16 kelas, 3.240 gambar) dengan MobileNetV2 sebagai model terbaik. Akurasinya sangat tinggi pada dataset yang dikumpulkan sendiri, tetapi ketahanan model pada data dari sumber lain belum diuji.
- Penelitian klasifikasi masakan Padang pada dataset Kaggle (9 kelas) masih terbatas pada satu daerah dan jumlah kelas yang sedikit.
- Secara umum, penelitian makanan Indonesia lebih menekankan akurasi, sedangkan perbandingan model ringan yang mempertimbangkan akurasi, ukuran model, dan kecepatan inferensi sekaligus masih jarang dilakukan.
## 4. Peluang Pengembangan
- **Menambah jenis makanan.** Dataset yang dipakai baru memuat sedikit jenis makanan. Penelitian lanjutan bisa menambah makanan khas daerah lain (misalnya dari Sumatra, Sulawesi, atau Papua) supaya model mengenali lebih banyak hidangan Indonesia.
- **Menguji dengan foto sendiri.** Foto di dataset biasanya dari internet. Model bisa diuji dengan foto yang diambil langsung memakai ponsel untuk melihat apakah tetap akurat di kondisi nyata, seperti cahaya redup atau piring yang berantakan.
- **Menampilkan informasi gizi.** Setelah makanan dikenali, aplikasi bisa menambahkan perkiraan kalori dan kandungan gizinya, sehingga berguna untuk mencatat asupan makan.
- **Mengenali beberapa makanan dalam satu foto.** Penelitian ini mengenali satu jenis makanan per foto. Pengembangan berikutnya bisa mengenali beberapa lauk dalam satu piring, misalnya nasi campur.
- **Memperkecil model lagi.** Ukuran model masih bisa dikurangi dengan teknik kompresi, sehingga berjalan lebih cepat di ponsel murah tanpa menurunkan akurasi terlalu banyak.
- ## 5. Sumber Data & Refrensi
  **a. Nama repository dataset**
    - Indonesian Food Dataset (Kaggle), berisi 6.500 gambar dari 13 kelas makanan, masing-masing 500 gambar.

  **b. Alamat dataset**
    - https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification

  **c. Referensi jurnal**
    1. Rasyidi, M. A., Mardhiyyah, Y. S., Nasution, Z., & Wijaya, C. H. (2024). Performance comparison of state-of-the-art deep learning model architectures in Indonesian food image classification. *Bulletin of Electrical Engineering and Informatics*, 13(5), 3355-3368. https://doi.org/10.11591/eei.v13i5.7996
    2. Wibisono, A., Wisesa, H. A., Rahmadhani, Z. P., Fahira, P. K., Mursanto, P., & Jatmiko, W. (2020). Traditional food knowledge of Indonesia: a new high-quality food dataset and automatic recognition system. *Journal of Big Data*, 7, 69. https://doi.org/10.1186/s40537-020-00342-5
    3. Benarkah, N., Siswantoro, J., & Ikhsan, M. (2025). Development of a mobile application using convolutional neural networks for recognizing Indonesian traditional snacks. *Teknika*, 14(2). https://garuda.kemdiktisaintek.go.id/documents/detail/6479641
    4. Afrinanto, F. F. (2022). Padang Cuisine (Indonesian Food Image Dataset) [Data set]. Kaggle.
