
# --- README.md ---
readme_md = '''
# Система прогнозування попиту

Цей додаток на Streamlit дозволяє прогнозувати продажі товарів за історичними даними.

## Інструкції

1. Клонуйте репозиторій:

```bash
git clone <URL репозиторію>
cd forecast_app
```

2. Встановіть залежності:

```bash
pip install -r requirements.txt
```

3. Запустіть додаток локально (необов'язково, можна використовувати Streamlit Cloud):

```bash
streamlit run app.py
```

4. Або відкрийте на [https://share.streamlit.io](https://share.streamlit.io) через Android:

- Увійдіть через GitHub.
- Натисніть **New app**, оберіть репозиторій і гілку.
- Натисніть **Deploy**.
- Додаток готовий до використання у браузері.
'''

# --- Запис файлів ---
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_py)

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_txt)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_md)

print("Готові файли створено: app.py, requirements.txt, README.md")
