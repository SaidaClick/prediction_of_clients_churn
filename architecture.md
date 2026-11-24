# Архитектура конвейера прогноза оттока

```mermaid
flowchart TD
    Source["Данные · data/churn-xs.tar.gz"]
    Raw["Наблюдения · CDR, обращения, телеметрия"]
    Truth["Разметка · reference/churn_truth.csv и passport.json"]
    Config["Настройки · config.json"]
    Load["load_data · чтение разделов до среза и проверка входа"]
    Features["preprocess · очистка и агрегация по абоненту"]
    Label["Метка · уход в окне прогноза"]
    Split["Разбиение по абонентам · train / validation / test"]
    Train["train_model · логистическая регрессия и XGBoost"]
    Threshold["Выбор порога на validation"]
    Eval["evaluate · расчёт метрик на test"]
    Output["artifacts/ключ запуска/<br/>prepared.parquet, модели, metrics.json, report.md"]

    Source --> Raw --> Load --> Features
    Source --> Truth --> Label
    Config --> Load
    Config --> Label
    Config --> Train
    Features --> Split
    Label --> Split
    Split -->|train| Train
    Split -->|validation| Threshold
    Split -->|test| Eval
    Train --> Threshold --> Eval
    Train --> Eval
    Split --> Output
    Train --> Output
    Eval --> Output
```

В признаках используются события до даты прогноза и CDR, уже полученные к этой дате. Даты ухода используются на стадии разметки. Все стадии выполняются в одном процессе, а комплект результатов публикуется после сохранения всех файлов.
