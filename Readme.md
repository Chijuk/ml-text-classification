## Machine learning tools
[![Generic badge](https://img.shields.io/badge/python-3.8.7-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/wfastcgi-3.0.0-blue.svg)](https://shields.io/)
### Зміст
____

1. [User guide](#User-guide)
   1. [Requirements](#Requirements)
   2. [Install](#Install)
   3. [How To](#How-To)
   4. [Settings json](#Settings-json)
3. [Service deployment](#Service-deployment)
   1. [Deployment requirements](#Deployment-requirements)
   2. [Deployment install](#Deployment)
### Опис
____
Репозиторій містить набір готових інструментів для навчання deep learning моделі 
для розв'язання наступних задач:
- [x] аналіз тексту і його класифікація згідно з певною ознакою (класа)

Структура папок проекту:

| Папка            | Призначення                                                                     |
|------------------|---------------------------------------------------------------------------------|
|`app\preprocessor`|Інструменти для попередньої обробки тексту                                       |
|`app\service `    |Веб сервіс, що відповідає за обробку запитів від клієнтів для передбачення ознаки|
|`app\trainer`     |Інструменти для навчання і візуалізації моделі                                   |
|`app\utils`       |Функції, що використовуються в модулях                                           |
|`notebooks`       |Робочі Jupyter notebooks. Використовується для роботи з інструментом онлайн      |
|`resources`       |Ресурси, що використовуються під час роботи                                      |

### User guide
____
#### Requirements
+ Python 3.8.7
+ Pipenv 2020.11.15
#### Install
+ Клонувати репозиторій локально в $PROJECT_DIR
+ Встановити pipenv
```
> pip install pipenv
```  
+ Якщо потрібно використовувати віртуальне середовище в паці з проектом — створити папку`.venv` в $PROJECT_DIR
+ Встановити всі додаткові пакети
```
> cd $PROJECT_DIR
> pipenv install --dev
```
#### How To
Інструментом можна користуватись двома способами:
1. Запуск готових скриптів з файлом налаштувань
2. Робота з модулями інструменту через свій Jupyter notebook

##### Запуск готових скриптів напряму з файлом налаштувань
1. Створити в зручному місці робочі папки, що будуть містити файли даних, логи та результати роботи навчання. Наприклад,
```
- \експеримент 1
   - \data
      - \clean
      - \trained
   - \logs
   - \custom_stop_words
   - preprocessor_config.json
   - trainer_config.json
   - service_config.json
```
2. Заповнити `json` файли налаштувань ([Settings json](#Settings json))
3. Запустити скрипт обробки даних
```
> python $PROJECT_DIR\app\preprocessor\preprocess.py preprocessor_config.json
```
4. Запустити скрипт тренування моделі
```
> python $PROJECT_DIR\app\trainer\train.py trainer_config.json
```
❗**NOTICE**: Скрипти будуть використовувати алгоритми очищення та тренування за замовчанням. Якщо потрібно використовувати специфічні алгоритми — використовувати Jupyter notebook
5. Запустити скрипт сервісу<br/>
```
> python $PROJECT_DIR\app\service\ml_service.py service_config.json
```
❗**NOTICE**: запускати таким чином тільки для тестування моделі!<br/>
Сервіс має наступні REST endpoints (http://127.0.0.1:5000):

- POST `/predict`<br/>
Тіло:
```json
{
   "text": "текст_для_передбачення"
}
```
Відповідь:
```json
{
   "predictions": [
   {
      "class_id": 123456,
      "class_name": "class_name",
      "probability": 100
   }
  ]
}
```
Якщо передбачення відсутні:
```json
{
   "predictions": null
}
```
- GET `/show_model`<br/>
  Показати повний опис моделі
- GET `/health`<br/>
Статус роботи сервісу

#### Settings json
##### - preprocessor_config.json
Файл містить налаштування для очистки текстових даних в сирому датасеті<br/>
Параметри:<br/>
`'name'`<br/>
Назва налаштування. Використовується для іменування лог файлу<br/> 
`'input_data_path'`<br/>
Абсолютний шлях до файла з сирим датасетом<br/>
`'output_data_path'`<br/>
Абсолютний шлях до файла куди буде збережено очищений датасет<br/>
`'log_path'`<br/>
Абсолютний шлях до папки куди буде збережено лог<br/>
`'drop_all_duplicates'`<br/>
Видаляти дублікати стрічок з текстом. Значення: `true` або `false`<br/>
`'drop_duplicates_class_list'`<br/>
Масив класів. Якщо масив не пустий (`[]`): дублікати будуть видалятись лише у вказаних класах<br/>
`'min_words_count'`<br/>
Мінімальна кількість допустимих слів. Якщо кількість менша за вказану — рядок видаляється. Значення: `0` - налаштування ігнорується<br/>
`'min_word_len'`<br/>
Мінімальна кількість символів у слові. Якщо кількість менша за вказану — слово видаляється. Значення: `0` - налаштування
ігнорується<br/>
`'max_word_len'`<br/>
Максимальна кількість символів у слові. Якщо кількість більша за вказану — слово видаляється. Значення: `0` -
налаштування ігнорується<br/>
`'clean_stop_words'`<br/>
Використовувати механізм видалення стоп слів. Значення: `true` або `false`<br/>
`'stop_words_settings'|'use_uk_stop_words'`<br/>
Використовувати передзаповнений словник українських стоп слів. Значення: `true` або `false`<br/>
`'stop_words_settings'|'use_ru_stop_words'`<br/>
Використовувати передзаповнений словник російських стоп слів. Значення: `true` або `false`<br/>
`'stop_words_settings'|'alt_stop_words_file'`<br/>
Абсолютний шлях для файлу в форматі `txt`, де знаходяться альтернативні стоп слова (очистка по токенам). Значення: "" -
налаштування ігнорується<br/>
`'stop_words_settings'|'custom_stop_words_path'`<br/>
Абсолютний шлях для папки, де знаходяться файли з користувацькими стоп словами в форматі `txt`. Значення: "" -
налаштування ігнорується<br/>
`'stop_words_settings'|'use_file_cleanup'`<br/>
Масив імен файлів зі стоп словами з розширенням. Якщо масив не пустий (`[]`) - всі стоп слова будуть попередньо очищені
по наступному алгоритму:
- очищення цифри
- лематизація слів згідно налаштувань

`'clean_email'`<br/>
Виконувати очистку email ознак в тексті. Значення: `true` або `false`<br/>
`'email_setting'|'signatures'`<br/>
Перелік підписів email через роздільник: `;`. Якщо підпис знайдений в тексті — все, що після нього видаляється. Значення: `""` - налаштування ігнорується<br/>
`'email_setting'|'clean_address'`<br/>
Видаляти email адреси в тексті по regexp. Значення: `true` або `false`<br/>
`'clean_html'`<br/>
Видаляти html теги в тексті. Значення: `true` або `false`<br/>
`'clean_urls'`<br/>
Видаляти url в тексті по regexp. Значення: `true` або `false`<br/>
`'use_words_lemmatization'`<br/>
Використовувати лематизацію слів. Значення: `true` або `false`<br/>
`'words_lemmatization_setting'|'russian'`<br/>
Використовувати визначення російської мови для лематизації слів. Значення: `true` або `false`<br/>
`'words_lemmatization_setting'|'ukrainian'`<br/>
Використовувати визначення української мови для лематизації слів. Значення: `true` або `false`<br/>

##### - trainer_config.json
Файл містить налаштування для тренування моделі<br/>
Параметри:<br/>
`'name'`<br/>
Назва налаштування. Використовується для іменування лог файлу<br/> 
`'input_data_path'`<br/>
Абсолютний шлях до файла з очищеним датасетом<br/>
`'output_data_path'`<br/>
Абсолютний шлях до папки куди буде збережено модель і суміжні файли<br/>
`'log_path'`<br/>
Абсолютний шлях до папки куди буде збережено лог<br/>
`'use_data_balancing'`<br/>
Використовувати балансування даних перед навчанням. Значення: `true` або `false`<br/>
`'data_balancing_setting'|'min_class_data'`<br/>
Класи, що містять кількість зразків менше ніж вказано — видаляються. Значення: `0` - налаштування ігнорується<br/>
`'data_balancing_setting'|'over_sampling_value'`<br/>
Збільшує кількість зразків всх класів на вказане значення, якщо їх менше. Зразки копіюються з існуючих в класі випадковим чином. Значення: `0` - налаштування ігнорується<br/>
`'data_balancing_setting'|'over_sampling_ratio'`<br/>
Словник: "ключ": значення. Містить назву класу і значення кількості зразків, до якого потрібно привести. Зразки копіюються з існуючих в класі випадковим чином. Якщо словник пустий (`{}`) або використовується `'over_sampling_value'`- налаштування ігнорується<br/>
`'intermediate_data_path'`<br/>
Абсолютний шлях до папки куди будуть збережені train, test, validation датасети. Значення: `""` - налаштування ігнорується<br/>
`'model_name'`<br/>
Ім'я моделі. Значення: `""` - налаштування ігнорується<br/>
`'train_data_size'`<br/>
Співвідношення даних для навчання до загальної кількості даних в датасеті.<br/>
Наприклад, розмір датасету - 100. Якщо `"train_data_size":0.8`, то розмір тренувальних даних - 80, 20 - тестування + валідація <br/>
`'validation_data_size'`<br/>
Співвідношення даних для валідації до даних, що залишились після розподілу в `'train_data_size'`<br/>
`'use_class_weight'`<br/>
Використовуються ваги класів для навчання. Значення: `true` або `false`<br/>
`'use_sample_weight'`<br/>
Використовуються ваги для зразків. Значення: `true` або `false`<br/>
`'dict_num_words'`<br/>
Кількість слів, що використовуються в словнику. Значення: `0` - використовується увесь словник<br/>
`'max_sequence_length'`<br/>
Довжина послідовності слів в зразку. Відсутні слова заповнюються нулями з кінця<br/>
`'epochs'`<br/>
Кількість епох при навчанні моделі<br/>
`'batch_size'`<br/>
Розмір batch при навчанні моделі<br/>

##### - service_config.json
Файл містить налаштування для роботи сервісу машинного навчання<br/>
Параметри:<br/>
`'name'`<br/>
Назва налаштування. Використовується для іменування лог файлу<br/> 
`'log_path'`<br/>
Абсолютний шлях до папки куди буде збережено лог<br/>
`'model_path'`<br/>
Абсолютний шлях до файлу з моделлю<br/>
`'dict_path'`<br/>
Абсолютний шлях до файлу з словником<br/>
`'classes_path'`<br/>
Абсолютний шлях до файлу з описом класів<br/>
`'preprocessor_settings_path'`<br/>
Абсолютний шлях до файлу з налаштуваннями очищення тексту<br/>
`'top_n_predictions'`<br/>
Число передбачень у відповіді клієнту на виклик POST `/predict`

### Service deployment

____

#### Deployment requirements

+ Python 3.8.7
+ IIS

#### Deployment

Для прикладу `machine-learning` назва папки, де знаходиться актуальний дистрибутив.

1. Скопіювати останню версію дистрибутиву сервісу на сервер в папку `C:\inetpub\wwwroot\machine-learning`
2. Скачати готову модель для розпізнавання мови
   за [посиланням](https://omniwayukraine.sharepoint.com/sites/owu/Shared%20Documents/Install/Machine%20learning/fasttext/lid.176.bin)
   в папку `C:\inetpub\wwwroot\machine-learning\resources\bin`
3. Скачати бібліотеки для роботи сервісу
   за [посиланням](https://omniwayukraine.sharepoint.com/sites/owu/Shared%20Documents/Install/Machine%20learning/Virtual%20environment/.venv.zip)
   в папку з проектом `C:\inetpub\wwwroot\machine-learning\.venv`
4. Скопіювати в зручне місце файли, що використовуються для роботи сервісу:
   - модель
   - словник
   - класи
   - користувацькі стоп слова (якщо використовуються)
   - актуальне для моделі налаштування preprocessor_config.json
5. Створити налаштування service_config.json і прописати актуальні значення шляхів до файлів
6. Інсталювати IIS Application Initialization feature на сервері стандартним способом

##### Інсталяція через FastCGI handler

7. Інсталювати IIS CGI feature на сервері стандартним способом
8. В IIS на рівні сервера додати налаштування FastCGI:
   - Full Path: шлях до python.exe `C:\inetpub\wwwroot\machine-learning\.venv\Scripts\python.exe`
   - Arguments: шлях до wfastcgi.py `C:\inetpub\wwwroot\machine-learning\.venv\Lib\site-packages\wfastcgi.py`
   - Activity Timeout: 60 секкнд
   - Idle Timeout: 600 секунд
9. В IIS додати сайт з довільним портом і physical path - папка дистрибутиву `C:\inetpub\wwwroot\machine-learning`
10. В налаштуваннях сайта змінити значення Preload Enabled на True
11. В налаштуваннях відповідного application pool змінити значення Start Mode на AlwaysRunning
12. Додати в папку з дистрибутивом файл web.config з наступними налаштуваннями

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <handlers accessPolicy="Read, Script">
            <remove name="CGI-exe" />
            <add name="FlaskHandler" path="*" verb="*" modules="FastCgiModule" 
			scriptProcessor="C:\inetpub\wwwroot\machine-learning\.venv\Scripts\python.exe|C:\inetpub\wwwroot\machine-learning\.venv\Lib\site-packages\wfastcgi.py" 
			resourceType="Unspecified" requireAccess="Script" />
        </handlers>
    </system.webServer>
	
	<appSettings>
    <!-- Required settings -->
    <add key="WSGI_HANDLER" value="ml_service.app" />
    <add key="PYTHONPATH" value="C:\inetpub\wwwroot\machine-learning\app\service" />
    <add key="ML_SERVICE_SETTINGS" value="C:\inetpub\wwwroot\machine-learning\service_config.json" />

    <!-- Optional settings -->
    <add key="WSGI_LOG" value="C:\inetpub\wwwroot\machine-learning\wsgi-logs\ml_service.log" />
  </appSettings>
</configuration>
```

13. Створити папку для логів WCGI в папці з дистрибутивом `C:\inetpub\wwwroot\machine-learning\wsgi-logs\ml_service.log`
14. Надати права для користувача від імені якого працює сайт права на читання і редагування папки з дистрибутивом
15. Перезавантажити сервер IIS

##### Інсталяція через HttpPlatform handler

7. Інсталювати `httpPlatformHandler_amd64.msi` на сервері
8. В IIS додати сайт з довільним портом і physical path - папка дистрибутиву `C:\inetpub\wwwroot\machine-learning`
9. В налаштуваннях сайта змінити значення Preload Enabled на True
10. В налаштуваннях відповідного application pool змінити значення Start Mode на AlwaysRunning
11. Додати в папку з дистрибутивом файл web.config з наступними налаштуваннями

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="C:\inetpub\wwwroot\machine-learning\.venv\Scripts\python.exe"
                  arguments="C:\inetpub\wwwroot\machine-learning-alt\app\service\ml_service.py"
                  stdoutLogEnabled="true"
                  stdoutLogFile="C:\inetpub\wwwroot\machine-learning-alt\http-logs\http_service.log"
                  startupTimeLimit="60"
                  processesPerApplication="1">
      <environmentVariables>
        <environmentVariable name="SERVER_PORT" value="%HTTP_PLATFORM_PORT%" />
		<environmentVariable name="ML_SERVICE_SETTINGS" value="C:\inetpub\wwwroot\machine-learning-alt\service_config.json" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
```

12. Створити папку для логів http в папці з
    дистрибутивом `C:\inetpub\wwwroot\machine-learning\http-logs\http_service.log`
13. Надати права для користувача від імені якого працює сайт права на читання і редагування папки з дистрибутивом
14. Перезавантажити сервер IIS