# Архитектура конвейера

```mermaid
flowchart TD
    Input["Источник · data/telemetry-xs.tar.gz<br/>Суточные JSONL-разделы синтетической телеметрии"]
    Config["Настройки · config.json<br/>Признаки, seed, разбиение, параметры моделей и порогов"]
    Load["load_data · чтение JSONL<br/>Проверка разделов и обязательных полей"]
    Prep["preprocess · подготовка<br/>Дедупликация, признаки и разбиение по узлам"]
    Train["train_model · обучение на train<br/>StandardScaler + LogisticRegression, XGBoost"]
    Threshold["select_threshold · выбор порога<br/>Прогнозы моделей на validation"]
    Eval["evaluate · качество на test<br/>Метрики при 0.5 и выбранном пороге"]
    Save["Хранилище · artifacts/&lt;ключ запуска&gt;/<br/>prepared.parquet, logistic_regression.joblib, xgboost.json,<br/>metrics.json и report.md"]

    Input --> Load --> Prep
    Config --> Prep
    Config --> Train
    Config --> Threshold
    Prep -->|train| Train
    Prep -->|validation| Threshold
    Prep -->|test| Eval
    Train --> Threshold
    Train --> Eval
    Threshold --> Eval
    Prep -->|данные и обзор| Save
    Train -->|модели| Save
    Eval -->|метрики| Save
```

JSONL-файлы читаются из архива по порядку дат. Все стадии выполняются одним процессом. Логи идут в стандартный поток ошибок. Конвейер формирует весь комплект во временной папке и публикует его одним переименованием в пределах выходной директории. Каталог с тем же ключом используется повторно.
