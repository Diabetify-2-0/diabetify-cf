# TESTING.md

Dokumen ini merangkum implementasi dan hasil pengujian modul counterfactual produksi pada `diabetify-cf` sebagai bagian dari integrasi end-to-end Diabetify 2.0.

Pengujian yang dijelaskan di sini berfokus pada **service counterfactual yang sudah diproduksikan**, bukan pada pipeline eksperimen riset. Karena itu, seluruh metrik di bawah ini mengukur apakah sistem produksi:

- mematuhi constraint yang ditentukan user,
- menghasilkan outcome yang benar,
- terintegrasi dengan backend dan mobile secara benar,
- stabil,
- dan cukup cepat untuk penggunaan nyata.

## Ruang Lingkup

Arsitektur yang diuji adalah:

`diabetify-mobile -> diabetify-be -> RabbitMQ -> diabetify-cf -> diabetify-be -> diabetify-mobile`

Repository yang terlibat dalam pengujian:

- `diabetify-cf`
  Fokus utama evaluasi, verifier, scenario runner, dan agregasi metrik.
- `diabetify-be`
  Diuji sebagai komponen integrasi nyata melalui endpoint backend dan lifecycle job asynchronous.
- `diabetify-mobile`
  Diuji pada level unit test JVM, Compose instrumentation test untuk flow input dan hasil counterfactual, serta end-to-end on-device test (`CounterfactualE2ETest`) yang menjadi sumber utama metrik 5 dan 8 pada level mobile->mobile.

Hal yang **tidak** menjadi fokus dokumen ini:

- benchmark eksperimen di folder `experiments/`
- perbandingan engine riset seperti DiCE
- evaluasi ilmiah riset di luar service produksi

## Tujuan Pengujian

Tujuan pengujian adalah membuktikan bahwa implementasi service counterfactual produksi:

1. tidak melanggar fitur immutable,
2. tidak mengubah fitur di luar mutable yang diizinkan user,
3. menghasilkan kandidat yang benar-benar memenuhi target saat service menyatakan solusi ditemukan,
4. menyatakan `infeasible` secara benar saat solusi memang tidak ditemukan,
5. terintegrasi secara end-to-end dengan backend dan mobile,
6. hanya mengembalikan kandidat yang lolos verifikasi LOF eksternal,
7. stabil dan konsisten pada input yang sama,
8. responsif untuk penggunaan nyata,
9. mematuhi seluruh gate operasional produksi: rentang wajar, direction, dan transition.

## Pendekatan Pengujian

Pengujian dibangun dalam 3 lapisan utama:

### 1. Service-Level Verification di `diabetify-cf`

Lapisan ini memverifikasi perilaku service dan kandidat counterfactual yang returned dari production engine `NN`.

Komponen utama:

- `src/diabetify_cf/verification/external.py`
- `src/diabetify_cf/verification/runner.py`
- `src/diabetify_cf/verification/fixtures.py`
- `src/diabetify_cf/verification/reporting.py`

Peran lapisan ini:

- memvalidasi kandidat secara independen di luar flow `generate()`,
- menghitung ulang target satisfaction,
- menghitung ulang LOF eksternal,
- mengecek constraint gate,
- merangkum hasil ke report terstruktur.

### 2. Backend Integration / End-to-End Verification

Lapisan ini menjalankan pengujian melalui backend yang asli, bukan memanggil engine langsung.

Komponen utama:

- `src/diabetify_cf/verification/backend.py`
- `src/diabetify_cf/verification/suites.py`
- `src/diabetify_cf/verification/launcher.py`
- `src/diabetify_cf/verification/run_backend_suite_from_config.py`

Peran lapisan ini:

- submit request ke backend asli,
- memicu job asynchronous melalui RabbitMQ,
- mengambil hasil dari endpoint backend,
- mengirim hasil tersebut ke verifier eksternal,
- menghitung metrik akhir dari flow produksi nyata.

### 3. Mobile Verification

Lapisan ini memastikan flow counterfactual di mobile benar, mulai dari input,
hasil, sampai integrasi ujung-ke-ujung yang sebenarnya dialami pengguna.

Cakupan mobile:

- unit test JVM untuk repository, async use case, polling manager, dan flow helper,
- Compose instrumentation test untuk screen input counterfactual,
- Compose instrumentation test untuk result screen counterfactual,
- **End-to-End on-device test** (`CounterfactualE2ETest`) yang menembus jalur
  produksi nyata `mobile -> diabetify-be -> RabbitMQ -> diabetify-cf -> diabetify-be -> mobile`.
  Test ini memakai `@UninstallModules(TestAppModule::class)` sehingga jalur
  counterfactual (API service, job manager, repository) berjalan dengan stack
  produksi sesungguhnya, bukan mock. Lapisan inilah yang menjadi **sumber utama
  metrik 5 (End-to-End Scenario Pass Rate) dan metrik 8 (End-to-End Latency)**
  pada level mobile->mobile.

## Artefak Pengujian

Artefak utama hasil evaluasi produksi ada di:

- [configs/verification/backend_suite_reports/index.json](D:\Perkuliahan\Tugas%20Akhir%20-%20The%20Last%20Chapter%2FProgram%2Fdiabetify-cf%2Fconfigs%2Fverification%2Fbackend_suite_reports%2Findex.json:1)
- [configs/verification/backend_suite_reports/feasible_core.json](D:\Perkuliahan\Tugas%20Akhir%20-%20The%20Last%20Chapter%2FProgram%2Fdiabetify-cf%2Fconfigs%2Fverification%2Fbackend_suite_reports%2Ffeasible_core.json:1)
- [configs/verification/backend_suite_reports/infeasible_core.json](D:\Perkuliahan\Tugas%20Akhir%20-%20The%20Last%20Chapter%2FProgram%2Fdiabetify-cf%2Fconfigs%2Fverification%2Fbackend_suite_reports%2Finfeasible_core.json:1)
- [configs/verification/backend_suite_reports/repeatability_core.json](D:\Perkuliahan\Tugas%20Akhir%20-%20The%20Last%20Chapter%2FProgram%2Fdiabetify-cf%2Fconfigs%2Fverification%2Fbackend_suite_reports%2Frepeatability_core.json:1)

## Definisi 9 Metrik Keberhasilan

### 1. Immutable Violation Rate

Definisi:

Persentase kandidat yang mengubah minimal satu fitur immutable.

Rumus:

`(# kandidat melanggar immutable / # total kandidat returned) x 100%`

Target:

`0%`

### 2. Mutable Violation Rate

Definisi:

Persentase kandidat yang mengubah fitur di luar `mutable_allowed`.

Rumus:

`(# kandidat melanggar mutable / # total kandidat returned) x 100%`

Target:

`0%`

### 3. Externally Verified Target Satisfaction Rate

Definisi:

Persentase kandidat feasible yang, setelah diverifikasi ulang secara independen oleh evaluator eksternal, benar-benar memenuhi target yang diminta user.

Rumus:

`(# kandidat feasible yang lolos verifikasi target eksternal / # total kandidat feasible returned) x 100%`

Target:

`100%`

Catatan:

Metrik ini sengaja diverifikasi ulang di luar flow internal service agar tidak menjadi tautologi.

### 4. Infeasible Handling Accuracy

Definisi:

Persentase skenario tanpa solusi yang dikembalikan service sebagai `INFEASIBLE` dengan status dan `reason_code` yang benar.

Rumus:

`(# skenario no-solution yang ditangani benar / # total skenario no-solution) x 100%`

Target:

`100%`

### 5. End-to-End Scenario Pass Rate

Definisi:

Persentase skenario user end-to-end yang lulus penuh dari submit di mobile
sampai hasil kembali tervalidasi dan tampil di mobile.

Rumus:

`(# skenario E2E lulus / # total skenario E2E) x 100%`

Target:

`100%`

Sumber pengukuran:

- **Utama (mobile->mobile):** `CounterfactualE2ETest` on-device. Skenario
  dianggap lulus bila alur penuh dari UI mencapai state `Feasible`
  (`CounterfactualResultState_Feasible`) melalui stack produksi nyata.
- **Pendukung (lapis layanan):** backend suite (`ScenarioRunner`) yang menilai
  `end_to_end_scenario_pass_rate` pada jalur `diabetify-be -> RabbitMQ -> diabetify-cf`.

### 6. External LOF Verification Accuracy

Definisi:

Persentase kandidat returned yang, setelah dihitung ulang oleh verifier eksternal, tetap memiliki `LOF` di bawah atau sama dengan threshold plausibilitas service.

Rumus:

`(# kandidat returned dengan external LOF <= threshold / # total kandidat returned) x 100%`

Target:

`100%`

### 7. Repeatability Rate

Definisi:

Persentase eksekusi berulang pada input identik yang menghasilkan outcome konsisten.

Konsistensi minimum yang dicek:

- status sama,
- `reason_code` sama,
- kandidat utama konsisten,
- hasil verifikasi eksternal konsisten.

Rumus:

`(# eksekusi ulang konsisten / # total eksekusi ulang) x 100%`

Target:

`100%`

### 8. End-to-End Latency

Definisi:

Waktu total yang dialami pengguna, dari saat request di-submit di mobile sampai
hasil siap ditampilkan di mobile. Latency ini sudah mencakup overhead polling
status dan render UI, bukan hanya round-trip layanan.

Ukuran yang dilaporkan:

- `average latency`
- `p95 latency`

Target:

- `average < 5 detik`
- `p95 < 5 detik`

Sumber pengukuran:

- **Utama (mobile->mobile):** `CounterfactualE2ETest::serviceBackedLatencyStaysUnderFiveSeconds`
  mengukur latency dari klik tombol "Cari Skenario" sampai state hasil tampil,
  diulang `3` iterasi, lalu meng-assert `average < 5000 ms` dan `p95 < 5000 ms`
  (`verifyLatencyWithinTarget`). Nilai presisi per iterasi tercatat di logcat
  (`Counterfactual E2E iteration <n>: ... latencyMs=<...>`).
- **Pendukung (lapis layanan):** backend suite melaporkan latency round-trip
  `diabetify-be -> diabetify-cf` sebagai komponen dominan dari latency mobile.

### 9. Constraint-Gate Compliance Rate

Definisi:

Persentase kandidat yang lolos seluruh gate operasional produksi:

- rentang wajar (`_medical_ok`)
- direction sesuai (`_directional_ok`)
- transition wajar (`_transition_ok`)

Rumus:

`(# kandidat lolos seluruh gate / # total kandidat returned) x 100%`

Target:

`100%`

## Teknis Implementasi Pengujian

### A. Skenario Produksi

Skenario produksi yang dipakai disimpan di `configs/verification/` dan dikelompokkan menjadi tiga suite:

#### `feasible_core`

Menguji skenario yang memang harus berhasil menghasilkan outcome feasible.

Skenario:

- `feasible_bmi_activity`
- `feasible_bmi_activity_repeatability`
- `feasible_target_already_satisfied`

#### `infeasible_core`

Menguji skenario yang memang harus berakhir infeasible.

Skenario:

- `infeasible_no_mutable`
- `infeasible_target_unreachable_bmi_only`
- `infeasible_medical_rule_only_high_target`

#### `repeatability_core`

Menguji kestabilan pada input yang sama secara berulang.

Skenario:

- `feasible_bmi_activity_repeatability`

### B. Jalur Eksekusi

Jalur utama pengujian end-to-end backend:

1. Runner memuat skenario JSON.
2. Runner melakukan preflight ke endpoint health backend.
3. Runner login memakai user uji.
4. Runner submit request ke backend asli.
5. Backend membuat job dan publish ke RabbitMQ.
6. `diabetify-cf` memproses request menggunakan production engine `NN`.
7. Backend menerima response dan menyimpan hasil.
8. Runner mengambil hasil dari backend.
9. Verifier eksternal memeriksa ulang kandidat dan menghitung metrik.
10. Reporter menghasilkan JSON suite report dan `index.json`.

### C. Verifikasi Eksternal

Verifier eksternal mengukur ulang:

- immutable compliance,
- mutable compliance,
- target satisfaction,
- external LOF,
- gate compliance.

Dengan cara ini, metrik tidak hanya bergantung pada klaim internal service.

## Perintah Pengujian yang Digunakan

### Suite service produksi

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Hasil:

- `72 passed`

### Backend suite report

```powershell
python -m diabetify_cf.verification.run_backend_suite_from_config --config configs/verification/backend_suite_launcher.example.json
```

Hasil:

- menghasilkan `index.json`
- menghasilkan `feasible_core.json`
- menghasilkan `infeasible_core.json`
- menghasilkan `repeatability_core.json`

### Mobile instrumentation tests

Test class yang dijalankan:

- `com.itb.diabetify.presentation.home.counterfactual.CounterfactualScreenContentTest`
- `com.itb.diabetify.presentation.home.counterfactual.CounterfactualResultContentTest`

Perintah:

```powershell
adb shell am instrument -w -e class com.itb.diabetify.presentation.home.counterfactual.CounterfactualScreenContentTest,com.itb.diabetify.presentation.home.counterfactual.CounterfactualResultContentTest com.itb.diabetify.test/com.itb.diabetify.HiltTestRunner
```

Hasil:

- `OK (6 tests)`

### Mobile end-to-end test (mobile->mobile)

Test class yang dijalankan:

- `com.itb.diabetify.e2e.presentation.counterfactual.CounterfactualE2ETest`

Test ini menjalankan stack produksi nyata pada device (login asli, submit asli,
RabbitMQ asli, service `diabetify-cf` asli) dan menjadi sumber utama metrik 5
dan 8 pada level mobile->mobile. Mencakup dua kasus:

- `counterfactualFlow_serviceBackedScenarioCompletesOnDevice` — memvalidasi
  metrik 5 (alur penuh mencapai state `Feasible`).
- `counterfactualFlow_serviceBackedLatencyStaysUnderFiveSeconds` — memvalidasi
  metrik 8 (`3` iterasi, assert `average < 5000 ms` dan `p95 < 5000 ms`).

Perintah:

```powershell
adb shell am instrument -w -e class com.itb.diabetify.e2e.presentation.counterfactual.CounterfactualE2ETest com.itb.diabetify.test/com.itb.diabetify.HiltTestRunner
```

Hasil:

- kedua kasus lulus di device nyata,
- nilai latency per iterasi tercatat di logcat (lihat catatan pada tabel hasil).

## Hasil Aktual Pengujian

Metrik kualitas kandidat (1, 2, 3, 4, 6, 7, 9) dan metrik E2E pada **lapis
layanan** (5 dan 8 versi `diabetify-be -> diabetify-cf`) diambil dari
`overall_summary` pada `backend_suite_reports/index.json`. Metrik 5 dan 8 versi
final **mobile->mobile** diambil dari `CounterfactualE2ETest` (lihat catatan di
bawah tabel).

| Metrik | Target | Hasil Aktual | Sumber | Status |
|---|---:|---|---|---|
| Immutable Violation Rate | `0%` | `0.0` | backend suite | Lulus |
| Mutable Violation Rate | `0%` | `0.0` | backend suite | Lulus |
| Externally Verified Target Satisfaction Rate | `100%` | `1.0` | backend suite | Lulus |
| Infeasible Handling Accuracy | `100%` | `1.0` | backend suite | Lulus |
| End-to-End Scenario Pass Rate | `100%` | `1.0` (mobile->mobile, state `Feasible` tercapai); `1.0` (lapis layanan) | `CounterfactualE2ETest` + backend suite | Lulus |
| External LOF Verification Accuracy | `100%` | `1.0` | backend suite | Lulus |
| Repeatability Rate | `100%` | `1.0` | backend suite | Lulus |
| End-to-End Latency | `average < 5 detik`, `p95 < 5 detik` | mobile->mobile: `average < 5000 ms` dan `p95 < 5000 ms` (assert lulus, `3` iterasi); referensi lapis layanan: `average 943.73 ms`, `p95 1059.5 ms` | `CounterfactualE2ETest` + backend suite | Lulus |
| Constraint-Gate Compliance Rate | `100%` | `1.0` | backend suite | Lulus |

Catatan metrik 5 dan 8:

- Definisi final keduanya adalah ujung-ke-ujung **mobile->mobile**, sehingga
  sumber kebenarannya adalah `CounterfactualE2ETest` on-device. Backend suite
  berperan sebagai bukti pendukung pada lapis layanan.
- Latency mobile divalidasi melalui hard assertion `< 5000 ms` (average dan p95)
  di `verifyLatencyWithinTarget`. Nilai numerik presisi per iterasi tidak
  disimpan sebagai artefak JSON, melainkan tercatat di logcat saat run
  (`Counterfactual E2E iteration <n>: state=..., latencyMs=...`). Untuk laporan,
  nilai presisi dapat dikutip langsung dari output logcat tersebut.

Ringkasan volume pengujian (lapis layanan, backend suite):

- total suite: `3`
- total scenario: `7`
- total run: `11`
- total kandidat tervalidasi: `7`

## Temuan Utama per Suite

### 1. Feasible Suite

Ringkasan:

- `scenario_count = 3`
- `total_runs = 5`
- `passed = true`
- `average_latency_ms = 1035.8`

Temuan:

- Semua kandidat feasible yang dikembalikan service lolos verifikasi immutable.
- Semua kandidat hanya mengubah fitur mutable yang diizinkan.
- Semua kandidat feasible memenuhi target setelah diverifikasi ulang secara eksternal.
- Semua kandidat feasible lolos external LOF verification.
- Semua kandidat feasible lolos gate `medical`, `directional`, dan `transition`.
- Skenario `TARGET_ALREADY_SATISFIED` berhasil menghasilkan `FEASIBLE` tanpa kandidat tambahan, sesuai desain.

### 2. Infeasible Suite

Ringkasan:

- `scenario_count = 3`
- `total_runs = 3`
- `passed = true`
- `average_latency_ms = 699.67`

Temuan:

- `NO_MUTABLE_FEATURE` berhasil ditangani benar.
- `TARGET_UNREACHABLE_UNDER_CONSTRAINTS` berhasil ditangani benar.
- `MEDICAL_RULE_VIOLATION_ONLY` berhasil ditangani benar.
- Seluruh skenario no-solution menghasilkan `INFEASIBLE` dengan `reason_code` yang sesuai.

Ini penting karena metrik infeasible merupakan salah satu metrik yang paling kuat secara engineering: service tidak mengarang solusi saat solusi memang tidak ada.

### 3. Repeatability Suite

Ringkasan:

- `scenario_count = 1`
- `total_runs = 3`
- `passed = true`
- `repeatability_rate = 1.0`

Temuan:

- Input yang sama menghasilkan outcome yang konsisten pada tiga pengulangan.
- `candidate_id`, target satisfaction, external LOF, dan constraint gate status konsisten.

## Hasil Mobile Verification

Pengujian mobile dilakukan agar klaim end-to-end tidak berhenti di backend dan service saja, tetapi juga menutup jalur UI utama yang dipakai user.

### Input screen

Menguji:

- opsi mutable dirender benar,
- aksi run button dapat dipicu,
- helper text berubah sesuai kondisi target,
- state target-already-satisfied dan state search-needed ditampilkan benar.

### Result screen

Menguji:

- state feasible,
- state target already satisfied,
- state no scenario,
- tombol aksi utama pada result screen.

Hasil:

- kedua test class lulus di device nyata,
- total `OK (6 tests)`.

### End-to-end on-device (sumber metrik 5 dan 8)

`CounterfactualE2ETest` menjalankan jalur produksi nyata
`mobile -> diabetify-be -> RabbitMQ -> diabetify-cf -> diabetify-be -> mobile`
di device, tanpa mock pada jalur counterfactual.

Menguji:

- alur penuh dari Home -> layar counterfactual -> set target -> run -> hasil,
- metrik 5: skenario lulus bila state hasil yang tampil adalah `Feasible`,
- metrik 8: latency user-perceived (klik tombol sampai hasil tampil) pada `3`
  iterasi dengan assert `average < 5000 ms` dan `p95 < 5000 ms`.

Hasil:

- `counterfactualFlow_serviceBackedScenarioCompletesOnDevice` lulus
  (state `Feasible` tercapai end-to-end),
- `counterfactualFlow_serviceBackedLatencyStaysUnderFiveSeconds` lulus
  (average dan p95 di bawah 5 detik),
- nilai latency presisi per iterasi tercatat di logcat.

## Interpretasi Keseluruhan

Berdasarkan hasil pengujian:

1. service produksi hanya mengembalikan kandidat yang patuh terhadap immutable, mutable, target, LOF, dan gate operasional,
2. service mampu menyatakan `infeasible` dengan benar pada skenario tanpa solusi,
3. integrasi backend-service berjalan benar pada flow asynchronous nyata,
4. perilaku service stabil pada input yang sama,
5. latency ujung-ke-ujung mobile->mobile berada di bawah target `5 detik` (average dan p95), tervalidasi on-device,
6. mobile counterfactual flow tervalidasi otomatis pada input screen, result screen, dan jalur end-to-end produksi nyata.

Dengan demikian, implementasi engineering modul counterfactual Diabetify 2.0 dapat dinyatakan **berhasil** berdasarkan 9 metrik yang telah didefinisikan.

## Batasan dan Catatan

- Skenario `TIMEOUT_NO_FEASIBLE_SOLUTION` tidak dijadikan bagian inti suite produksi, dan diposisikan sebagai jalur robustness service-level, bukan bagian utama dari 9 metrik produksi.
- Engine eksperimen seperti DiCE tidak menjadi bagian dari evaluasi keberhasilan produksi.
- Seluruh hasil pada dokumen ini merujuk pada production engine `NN` sebagai satu-satunya engine produksi service.
- Metrik 5 dan 8 versi mobile->mobile divalidasi via hard assertion pada `CounterfactualE2ETest`; nilai latency presisi belum disimpan sebagai artefak file dan harus dikutip dari output logcat saat run.
- Volume run produksi pada backend suite relatif kecil (`7` skenario, `11` run, `7` kandidat tervalidasi). Kepercayaan terhadap metrik ditopang juga oleh unit test yang membuktikan setiap gate mampu mendeteksi pelanggaran (rate bukan nol), bukan hanya kasus lulus.

## Kesimpulan

Kesimpulan akhir pengujian adalah:

- implementasi pengujian untuk 9 metrik telah selesai,
- seluruh 9 metrik telah terpenuhi pada run aktual,
- integrasi `mobile -> backend -> RabbitMQ -> diabetify-cf -> backend -> mobile` tervalidasi,
- dan service counterfactual produksi siap diklaim berhasil secara engineering.

