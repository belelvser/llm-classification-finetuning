# Handcrafted Features + CatBoost — компактный отчёт

## 1. Экспериментальная схема

После воспроизведения исходного baseline с 9 handcrafted-признаками был построен расширенный набор признаков и проведён их последовательный отбор.

Для основных сравнений использовался один и тот же `5-fold StratifiedKFold` с `random_state=42`.  
Метрика: **multiclass log loss**, где меньше — лучше.

### Исходный baseline

| Model | Log loss |
|---|---:|
| Logistic Regression + 9 features | 1.062693 |
| CatBoost + 9 features | 1.041892 |

Замена Logistic Regression на CatBoost при неизменных 9 признаках улучшила log loss на **0.020801**.

## 2. Расширение и отбор признаков

Первоначальный расширенный набор содержал **33 признака**: исходные EDA-признаки и структурные характеристики ответов.

Group ablation показал, что наиболее полезными группами являются:

| Удалённая группа | Ухудшение log loss |
|---|---:|
| Response length | +0.008524 |
| Refusals | +0.007563 |
| Repetition | +0.005295 |
| Prompt length | +0.001997 |
| Punctuation | +0.001542 |
| Paragraph statistics | +0.000760 |

Markdown- и sentence-level признаки дополнительного прироста не дали и были удалены.

Внутри length-группы несколько признаков оказались избыточными. Шесть length-признаков были сокращены до трёх:
`len_ratio_a_b`, `length_a`, `length_b`.

Paragraph-группа была сокращена с 10 до 8 признаков: удалены
`sd_sent_per_paragraph_a` и `sd_sent_per_paragraph_b`.

## 3. Новый EDA-инсайт: числовой контент

Если числа присутствовали только в ответе A:

- A wins: **48.8%**
- B wins: 25.3%
- Tie: 25.9%

Если числа присутствовали только в ответе B:

- A wins: 26.0%
- B wins: **48.2%**
- Tie: 25.8%

Эффект почти симметричен, поэтому были добавлены два бинарных признака:
`has_numbers_a` и `has_numbers_b`.

Признак `numeric_density_diff` почти не дал прироста, поэтому в финальный набор не вошёл.

## 4. Финальный набор из 20 признаков

### Length — 3

1. `len_ratio_a_b`  
   Относительная разница длин ответов:
   `log(1 + len(A)) - log(1 + len(B))`.  
   Положительное значение означает, что A длиннее относительно B, отрицательное — B длиннее.

2. `length_a`  
   Количество символов в ответе A.

3. `length_b`  
   Количество символов в ответе B.

### Refusals — 2

4. `refusals_a`  
   Количество refusal/apology-маркеров в A, например:
   `I cannot`, `I can't`, `I'm sorry`, `As an AI`, `I am unable`, `I apologize`.

5. `refusals_b`  
   То же для ответа B.

### Repetition — 2

6. `repetition_density_a`  
   Плотность повторяемости словаря в A:
   `1 - unique_words / all_words`.  
   Чем выше значение, тем чаще используются уже встречавшиеся слова.

7. `repetition_density_b`  
   То же для B.

### Prompt — 1

8. `prompt_length`  
   Длина prompt в символах, масштабированная как `len(prompt) / 1000`.

### Punctuation — 2

9. `punctuation_density_a`  
   Доля знаков пунктуации среди всех символов ответа A:
   `punctuation_count / text_length`.

10. `punctuation_density_b`  
    То же для B.

### Paragraph structure — 8

11. `num_paragraphs_a`  
    Число непустых абзацев в A.

12. `num_paragraphs_b`  
    Число непустых абзацев в B.

13. `mean_paragraph_length_a`  
    Средняя длина абзаца A в словах.

14. `mean_paragraph_length_b`  
    Средняя длина абзаца B в словах.

15. `sd_paragraph_length_a`  
    Стандартное отклонение длины абзацев A в словах. Показывает, насколько размеры абзацев неоднородны.

16. `sd_paragraph_length_b`  
    То же для B.

17. `avg_sent_per_paragraph_a`  
    Среднее число предложений в одном абзаце A.

18. `avg_sent_per_paragraph_b`  
    Среднее число предложений в одном абзаце B.

### Numeric content — 2

19. `has_numbers_a`  
    Бинарный индикатор: `1`, если в A встречается хотя бы одно числовое выражение, иначе `0`.

20. `has_numbers_b`  
    То же для B.

## 5. Итоговое сравнение

| Model | Selection CV |
|---|---:|
| Logistic Regression + 9 | 1.062693 |
| CatBoost + 9 | 1.041892 |
| Logistic Regression + 20 | 1.056715 |
| **CatBoost + 20** | **1.028325** |

Относительно исходного `Logistic Regression + 9` финальный CatBoost улучшил log loss на **0.034368**.

## 6. Robustness check

После фиксации финального набора был проведён дополнительный `5-fold StratifiedKFold` с `random_state=314159`.

| Model | Selection CV | Final CV |
|---|---:|---:|
| Logistic Regression + 9 | 1.062693 | 1.062759 |
| CatBoost + 9 | 1.041892 | 1.042413 |
| Logistic Regression + 20 | 1.056715 | 1.056391 |
| **CatBoost + 20** | **1.028325** | **1.028788** |

Порядок моделей сохранился, а изменения метрик были небольшими.

## 7. Основные выводы

- CatBoost значительно лучше Logistic Regression использует handcrafted-признаки.
- Расширение набора признаков улучшает обе модели, но особенно CatBoost.
- Наиболее сильные группы: response length, refusals и repetition.
- Часть length-признаков была избыточной и была удалена без потери качества.
- Paragraph-признаки полезны коллективно: после удаления одной слабой пары дальнейшее сокращение ухудшало результат.
- Наличие числового контента показало сильный симметричный EDA-эффект и небольшой дополнительный прирост модели.
- Финальный набор содержит 20 признаков и показывает устойчивый результат на другом CV-разбиении.
