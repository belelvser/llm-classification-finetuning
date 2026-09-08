# Воспроизведение baseline и EDA

В качестве существующего решения был воспроизведён Kaggle notebook
**“Why Humans Pick One AI Answer Over Another” by starkhushi**.

Датасет содержит 57 477 сравнений двух ответов LLM. Целевая переменная
имеет три класса: `A wins`, `B wins`, `Tie`.

## Баланс классов

Распределение классов:

- A wins: 20 064 (34.9%)
- B wins: 19 652 (34.2%)
- Tie: 17 761 (30.9%)

Классы достаточно сбалансированы. Равномерное предсказание
`1/3, 1/3, 1/3` даёт log loss 1.0986, а использование только частот
классов — 1.0972.

## Основные EDA-инсайты

### 1. Длина ответа — сильный сигнал

Чем длиннее A относительно B, тем чаще побеждает A, и наоборот.

В группе, где A значительно длиннее B:

- A wins: 50.7%
- B wins: 23.4%

Зависимость плавная и почти симметричная. При этом нельзя утверждать,
что длина сама является причиной победы: более сильные модели могут
одновременно давать более длинные и более качественные ответы.

### 2. Refusal-фразы связаны с проигрышем

Если только ответ A содержит:

- `"I cannot"` → A wins ≈ 21%
- `"I'm sorry"` → A wins ≈ 17–18%
- `"As an AI"` → A wins ≈ 25%

при обычной вероятности победы A ≈ 35%.

Таким образом, отказ, извинение или неспособность выполнить запрос
являются сильными текстовыми сигналами.

### 3. Форматирование связано с предпочтением

Если A заметно лучше структурирован с помощью Markdown:

- A wins ≈ 47%
- B wins ≈ 27%

Если лучше структурирован B:

- B wins ≈ 46%
- A wins ≈ 27%

Эффект получен на выборках примерно по 11 тысяч примеров с каждой стороны.

### 4. Похожие по длине ответы чаще получают Tie

Если длины ответов похожи:

- Tie: 36.9%

Если длины заметно отличаются:

- Tie: 29.4%

Следовательно, относительная длина полезна не только для выбора A/B,
но и для определения ничьей.

### 5. Идентичность модели использовать нельзя

`model_a` и `model_b` хорошо предсказывают результат на train,
но отсутствуют в test. Поэтому эти признаки исключаются.
Все используемые признаки должны быть вычислимы и на train, и на test.

## Исходные handcrafted features

По результатам EDA были сформированы 9 признаков:

1. relative length A/B
2. log length A
3. log length B
4. Markdown count A
5. Markdown count B
6. refusal count A
7. refusal count B
8. similar-length indicator
9. prompt length

## Baseline-результаты

| Метод | Log loss |
|---|---:|
| Uniform probabilities | 1.0986 |
| Class priors | 1.0972 |
| Logistic Regression + 9 features | 1.0627 |
| TF-IDF + Logistic Regression + features | 1.0605 |
| Blend | **1.0457** |

Для TF-IDF были воспроизведены fold scores:

- fold 0: 1.0580
- fold 1: 1.0570
- fold 2: 1.0665

Главный вывод: даже простые структурные признаки содержат заметный
сигнал о человеческих предпочтениях. Наиболее сильными оказались
относительная длина ответа и refusal-признаки.

Следующий этап — расширить набор handcrafted features примерно до
20–30 признаков и сравнить Logistic Regression и CatBoost на одном
и том же validation protocol.

# Baseline Reproduction and EDA

As an existing solution, we reproduced the Kaggle notebook
**“Why Humans Pick One AI Answer Over Another” by starkhushi**.

The dataset contains 57,477 pairwise LLM response comparisons.
The target has three classes: `A wins`, `B wins`, and `Tie`.

## Class distribution

- A wins: 20,064 (34.9%)
- B wins: 19,652 (34.2%)
- Tie: 17,761 (30.9%)

The classes are relatively balanced. Uniform probabilities
`1/3, 1/3, 1/3` produce a log loss of 1.0986, while using class priors
only slightly improves it to 1.0972.

## Main EDA insights

### 1. Response length is a strong signal

As response A becomes longer relative to B, the probability of A winning
increases, and vice versa.

In the most extreme group where A is much longer:

- A wins: 50.7%
- B wins: 23.4%

The relationship is smooth and approximately symmetric. However, this
does not prove that length itself causes preference, since stronger models
may produce responses that are both longer and better.

### 2. Refusal language is associated with losing

When only response A contains:

- `"I cannot"` → A wins ≈ 21%
- `"I'm sorry"` → A wins ≈ 17–18%
- `"As an AI"` → A wins ≈ 25%

compared with the overall A win rate of approximately 35%.

Refusals, apologies, and inability statements therefore provide strong
lexical signals.

### 3. Formatting correlates with preference

When A is visibly more structured using Markdown:

- A wins ≈ 47%
- B wins ≈ 27%

When B is more structured:

- B wins ≈ 46%
- A wins ≈ 27%

Each comparison is based on roughly 11,000 examples.

### 4. Similar response lengths are associated with ties

For responses with similar lengths:

- Tie rate: 36.9%

For responses with different lengths:

- Tie rate: 29.4%

Thus, relative length provides information not only for distinguishing
A from B, but also for predicting ties.

### 5. Model identity cannot be used

`model_a` and `model_b` are strongly predictive in the training data,
but are unavailable in the test data. They are therefore excluded.
All final features must be computable for both train and test samples.

## Original handcrafted features

The EDA was converted into nine numerical features:

1. relative length A/B
2. log length A
3. log length B
4. Markdown count A
5. Markdown count B
6. refusal count A
7. refusal count B
8. similar-length indicator
9. prompt length

## Baseline results

| Method | Log loss |
|---|---:|
| Uniform probabilities | 1.0986 |
| Class priors | 1.0972 |
| Logistic Regression + 9 features | 1.0627 |
| TF-IDF + Logistic Regression + features | 1.0605 |
| Blend | **1.0457** |

Reproduced TF-IDF fold scores:

- fold 0: 1.0580
- fold 1: 1.0570
- fold 2: 1.0665

The main conclusion is that simple structural properties already contain
a meaningful signal about human preference. Relative response length and
refusal-related features are the strongest simple signals observed.

The next step is to extend the handcrafted feature set to approximately
20–30 features and compare Logistic Regression and CatBoost under the same
validation protocol.