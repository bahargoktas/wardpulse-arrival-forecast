# WardPulse — Türkçe çalışma notları

1. `python -m pip install -e ".[dev]"` ile bağımlılıkları kur.
2. `python -m pytest -q` ile testleri çalıştır.
3. `python -m wardpulse all --output-dir reports` ile sentetik tahmin deneyini üret.
4. `reports/metrics.json` içindeki MAE değerlerini günlük ve haftalık baseline ile karşılaştır.
5. `interval.empirical_coverage` ile %80 nominal kapsama arasındaki farkı incele.
6. `src/wardpulse/pipeline.py` içinde `supervised`, `chronological_split` ve `absolute_residual_radius` fonksiyonlarını açıklayabilecek düzeyde çalış.

Önemli formül: MAE = (1/n) toplam |y - y_tahmin|. Split-conformal yarıçapı kalibrasyon mutlak hata dağılımının uygun kantilinden alınır. Bu garanti değişen zaman serisi dağılımlarında otomatik olarak geçerli değildir.

Tüm veriler sentetiktir; klinik kullanım veya personel planlaması için geçerli değildir.
