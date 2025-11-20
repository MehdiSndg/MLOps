# mlops_full_pipeline

Modüler, genişletilebilir ve bulut tabanlı bir MLOps referans projesi. AutoGluon ile otomatik modelleme, ZenML ile orkestrasyon, DVC + AWS S3 ile veri/artefakt versiyonlama, MLflow ile deney izleme, Docker ile paketleme ve Jenkins ile CI/CD entegrasyonu içerir.

## Mimari
- **DVC + S3**: Ham ve işlenmiş verilerin sürüm kontrolü, uzaktan S3 deposu.
- **ZenML**: Veri hazırlama, eğitim ve deploy pipeline’larının orkestrasyonu.
- **AutoGluon**: LightGBM, CatBoost, XGBoost dahil otomatik model araması.
- **MLflow**: Parametre/metric/artefakt kaydı ve model registry.
- **Docker**: Çalıştırılabilir imaj.
- **Jenkins**: CI/CD süreci (checkout, bağımlılıklar, DVC pull, build, pipeline çalıştırma, artefakt arşivi).

## Pipeline Akışı (metin diyagramı)
```
data_pipeline: load_data -> preprocess -> data/processed/processed.csv
train_pipeline: load processed -> train AutoGluon -> evaluate -> MLflow register
deploy_pipeline: load predictor -> seçilen en iyi modeli registry'ye kopyala (artifacts/registry)
```

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## AWS S3 DVC Remote
```bash
dvc init
dvc remote add -d myremote s3://mlops-bucket-mehdi/data
dvc remote modify myremote endpointurl https://s3.amazonaws.com
```
`.env` dosyasına örnek değerleri girin:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-central-1
```

## Pipeline Çalıştırma
- Tüm akış: `python run_pipelines.py`
- Sadece veri: `python run_pipelines.py --pipeline data`
- Eğitim: `python run_pipelines.py --pipeline train`
- Deploy: `python run_pipelines.py --pipeline deploy`

## Jenkins
`jenkins/Jenkinsfile` içindeki repo URL’sini kendi Git adresinizle değiştirin. Aşamalar: Checkout → Install Dependencies → DVC Pull → Docker Build → Run Pipelines → Archive Artifacts.

## Docker
```bash
docker build -t mlops_full_pipeline -f docker/Dockerfile .
docker run --rm -it mlops_full_pipeline
```

## Modüler Yapı
- `src/steps/*`: Her adım ZenML step olarak izole.
- `src/pipelines/*`: Veri, eğitim, deploy pipeline tanımları.
- `src/utils/*`: Ortak logger, mlflow, dvc yardımcıları.
- `src/training/*`: AutoGluon eğitim ve değerlendirme modülleri.

## Örnek Çıktılar
- `data/processed/processed.csv`: İşlenmiş veri.
- `artifacts/models/leaderboard.csv`: AutoGluon leaderboard.
- `artifacts/models/feature_importance.csv`: Özellik önemleri.
- MLflow run’larında metrikler ve parametreler kaydedilir, model registry’ye versiyon eklenir.
