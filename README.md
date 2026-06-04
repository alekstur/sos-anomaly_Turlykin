# Поиск аномальных респондентов — SoS

## Запуск

```
pip install -r requirements.txt
python solution_Турлыкин_ШАД-111.py
```

Скрипт ищет все parquet-файлы в своей папке и во всех подпапках
(поддерживается структура month=ГГГГ-ММ-ДД/). Папки с данными
должны лежать рядом со скриптом в одной директории. Папка output/
создаётся автоматически.

## Структура файлов

```
папка_проекта/
├── month=2025-06-01/
│   └── part-*.parquet
├── month=2025-07-01/
│   └── part-*.parquet
├── ...
├── solution_ФАМИЛИЯ_ГРУППА.py
├── DESCRIPTION.md
├── README.md
└── requirements.txt
```

## Выходные файлы

После запуска создаётся папка output/ со следующим содержимым:

| Файл | Описание |
|------|----------|
| output/anomalies.csv | Пары (SubjectID, researchdate) к удалению |
| output/anomaly_reasons.csv | Бренд, score, threshold и причина для каждого триггера |
| output/plots/total_ots_before_after.png | OTS по дням до и после очистки |
| output/plots/category_ots_change.png | Изменение OTS по категориям, % |
| output/plots/daily_anomaly_count.png | Число аномальных респондентов по дням |

## Аналитические функции

Все функции доступны для импорта из скрипта:

```python
from solution_ФАМИЛИЯ_ГРУППА import (
    load_data, compute_daily_ots, compute_modified_zscore,
    detect_anomalies, build_output_tables,
    analytics_respondent_profile,
    analytics_resource_profile,
    analytics_brand_ots_by_day,
    analytics_query_text,
)

df = load_data(".")
daily = compute_daily_ots(df)
daily = compute_modified_zscore(daily)
flagged = detect_anomalies(daily)
anomalies, anomaly_reasons = build_output_tables(flagged)

# До/после по социально-демографическим характеристикам
analytics_respondent_profile(df, anomalies, 'Пол')
analytics_respondent_profile(df, anomalies, 'Возраст')
analytics_respondent_profile(df, anomalies, 'Регион')
analytics_respondent_profile(df, anomalies, 'Федеральный_округ')
analytics_respondent_profile(df, anomalies, 'Количество_детей')
analytics_respondent_profile(df, anomalies, 'Занятость')
analytics_respondent_profile(df, anomalies, 'Доход')

# До/после по ресурсам
analytics_resource_profile(df, anomalies, 'ResourceName')
analytics_resource_profile(df, anomalies, 'ResourceType')
analytics_resource_profile(df, anomalies, 'Platform')
analytics_resource_profile(df, anomalies, 'UseType')

# До/после по уровням категорий
analytics_respondent_profile(df, anomalies, 'Category1')
analytics_respondent_profile(df, anomalies, 'Category2')
analytics_respondent_profile(df, anomalies, 'Category3')

# OTS конкретного бренда до/после по дням
analytics_brand_ots_by_day(df, anomalies, brand_id=207286)

# Все запросы аномального респондента за конкретный день
analytics_query_text(df, anomalies,
                     subject_id=1585271561337038249,
                     research_date='2025-07-01')
```

## Описание алгоритма

Подробное обоснование всех решений и числовых параметров — в файле DESCRIPTION.md.
