# %% [markdown]
# # LMSYS — замороженный бэкбон как экстрактор признаков + обучаемая голова
#
# Один самодостаточный ноутбук. Ничего, кроме него, загружать не нужно.
#
# ## Идея подхода
#
# Есть три способа применить LLM к этой задаче:
#
# | Подход | Что обучается | Стоимость |
# |---|---|---|
# | zero-shot | ничего, модель просто спрашивают | только инференс |
# | **замороженный бэкбон + голова** | **только голова, ~1 млн параметров** | **инференс один раз + секунды CPU** |
# | LoRA-файнтюн | адаптеры внутри бэкбона | часы GPU на каждый эксперимент |
#
# Мы посередине. Веса LLM **не меняются вообще**. Мы прогоняем через неё текст,
# забираем скрытое состояние `h` — вектор, в который модель сжала весь вход, — и
# учим поверх него маленькую сеть на 3 класса.
#
# ## Почему это разрезано на две части
#
# Бэкбон заморожен, значит `h` для конкретной строки **не меняется между
# экспериментами**. Считать его заново на каждый эксперимент — выбрасывать часы GPU.
#
# | Этап | Где | Стоимость | Как часто |
# |---|---|---|---|
# | **A. Экстракция** `h` | GPU | часы | один раз |
# | **B. Обучение головы** | CPU | секунды | сотни раз |
#
# Между ними — файлы на диске в `cache/`. Часть B читает только их и `folds.csv`,
# бэкбон в ней не грузится вообще. Поэтому:
#
# > **После того как экстракция прошла, не запускай ноутбук целиком сверху.**
# > Выполни ячейки до раздела 6 (они дешёвые), затем сразу раздел 9 и дальше.
# > Ячейка экстракции сама увидит готовый кэш и пропустит работу, но лишние
# > минуты на загрузку модели ты потратишь.

# %% [markdown]
# ## Как запустить на Kaggle
#
# 1. **New Notebook** → *File* → *Import Notebook* → загрузить этот файл.
# 2. *Settings* → **Accelerator: GPU T4 x2**.
# 3. *Settings* → **Internet: On** (нужен для скачивания бэкбона с HuggingFace;
#    если интернет выключен — подключи модель через *Add Input* → *Models*, ноутбук
#    сам найдёт её в `/kaggle/input/`).
# 4. *Add Input* → *Competitions* → `llm-classification-finetuning`.
# 5. Run All.
#
# Ноутбук сам определит: Kaggle это или локальная машина, сколько видеокарт,
# поддерживают ли они bfloat16, помещается ли модель в память без квантизации.
#
# ### Лимиты Kaggle, о которые можно убиться
#
# - Сессия с GPU живёт **до ~9 часов**, недельная квота **~30 часов**.
# - `/kaggle/working` сохраняется между версиями ноутбука, но ограничен ~20 ГБ.
#
# Экстракция устроена **с чекпоинтами**: она пишет маску готовых строк на диск и
# при повторном запуске продолжает с места обрыва. Если полный прогон не влезает
# в одну сессию — запусти ноутбук ещё раз, он доработает остаток.
#
# ### Выбор бэкбона под бюджет
#
# Стоимость экстракции линейна по числу параметров и по `max_length`. Оценки для
# полного датасета (57 477 строк × 2 порядка = 114 954 прохода, `max_length=1024`,
# две T4 в data parallel):
#
# | Бэкбон | H | Кэш | Время на 2×T4 | Влезает в сессию? |
# |---|---|---|---|---|
# | Qwen2.5-0.5B | 896 | 1.2 ГиБ | ~1.5 ч | да |
# | **Qwen2.5-1.5B (дефолт)** | **1536** | **2.0 ГиБ** | **~4 ч** | **да** |
# | Qwen2.5-7B | 3584 | 4.6 ГиБ | ~18 ч | нет, 3 запуска |
# | gemma-2-9b-it | 3584 | 4.6 ГиБ | ~23 ч | нет, 3 запуска |
#
# Время посчитано из FLOPs (`2·N·T` на проход) при 20% MFU и пиковых 65 TFLOPS
# fp16 на карту. Это прикидка ±2×; ноутбук всё равно измерит реальную скорость
# смоук-тестом и покажет свою проекцию до того, как запустится надолго.
#
# Дефолт — 1.5B: помещается в одну сессию и даёт представления заметно лучше,
# чем 0.5B. Чтобы взять другую модель, поменяй `model_hint` и `model_fallback`
# в конфиге. Кэш от разных моделей не конфликтует: имя модели входит в имя файла.

# %%
from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Kaggle is UTF-8 already; a local Windows console is not, and the Cyrillic in
# the progress output would come out as mojibake there.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("python     ", sys.version.split()[0])
print("numpy      ", np.__version__)
print("pandas     ", pd.__version__)
print("torch      ", torch.__version__)

# %% [markdown]
# ## 1. Конфиг
#
# Всё настраиваемое живёт в одном dataclass. Правило простое: если по ходу работы
# захотелось поменять число — оно должно быть здесь, а не вкраплено в код где-то
# в середине ноутбука. Иначе через неделю невозможно восстановить, при каких
# настройках получилась та или иная цифра.
#
# Дефолты выбраны **простейшие из возможных**. Это заглушки, чтобы пайплайн
# прошёл end-to-end и выдал осмысленные числа, а не оптимум. Всё, что стоит
# подбирать, помечено `# TODO(ivan):`.

# %%
@dataclass
class Config:
    seed: int = 42

    # ---- данные -------------------------------------------------------------
    n_folds: int = 5
    limit_rows: int | None = None  # None = весь датасет; число = первые N строк folds.csv

    # ---- сборка текста ------------------------------------------------------
    prompt_version: str = "v1"
    max_chars_prompt: int = 1200      # суммарный бюджет на все показанные ходы промпта
    max_chars_response: int = 2400    # тот же бюджет для A и B, симметрия обязательна
    max_turns: int = 4
    max_length: int = 1024            # жёсткий потолок в токенах

    # ---- бэкбон -------------------------------------------------------------
    # model_hint ищется среди подключённых на Kaggle моделей; если не найдено,
    # берётся model_fallback (скачивается с HuggingFace, нужен интернет).
    model_hint: str = "qwen2.5-1.5b"
    model_fallback: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path_override: str | None = None
    load_in_4bit: bool | None = None    # None = решить автоматически по размеру модели
    attn_implementation: str | None = None  # None = eager для gemma-2, иначе sdpa
    batch_size: int | None = None       # None = подобрать по объёму VRAM
    chunk_rows: int = 256               # единица чекпоинта при экстракции

    # ---- что сохраняем при экстракции --------------------------------------
    layer_fractions: tuple[float, ...] = (1.0, 0.75, 0.5)
    pools: tuple[str, ...] = ("last", "mean")
    orders: tuple[str, ...] = ("ab", "ba")

    # ---- голова -------------------------------------------------------------
    # TODO(ivan): выбор слоя и пуллинга — здесь. Доступны l100 / l75 / l50 и last / mean.
    feature_layers: tuple[str, ...] = ("l100",)
    feature_pool: str = "last"
    # TODO(ivan): усреднение против конкатенации двух порядков: mean | concat | ab
    order_combine: str = "mean"
    use_hand_features: bool = True
    # TODO(ivan): размер hidden, число слоёв головы, dropout
    hidden: int = 256
    n_head_layers: int = 1
    dropout: float = 0.1
    # TODO(ivan): label smoothing
    label_smoothing: float = 0.0

    # ---- обучение головы ----------------------------------------------------
    # TODO(ivan): подбор lr
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    patience: int = 5
    head_batch_size: int = 512
    warmup_frac: float = 0.1

    def layer_names(self) -> dict[str, float]:
        """Human-readable name for each saved layer, e.g. {'l100': 1.0}."""
        return {f"l{int(round(f * 100))}": f for f in self.layer_fractions}


CFG = Config()

# Воспроизводимость: один seed на все источники случайности.
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
print(json.dumps(asdict(CFG), indent=2, ensure_ascii=False, default=str))

# %% [markdown]
# ## 2. Среда: где мы и что у нас есть
#
# Ноутбук должен без правок работать и на Kaggle, и локально. Разница только в
# путях, поэтому определяем их один раз и дальше пользуемся переменными.

# %%
IS_KAGGLE = Path("/kaggle/input").exists()

def _resolve_data_dir() -> Path:
    """Find the directory holding train.csv, in order of preference."""
    if os.environ.get("LMSYS_DATA_DIR"):
        return Path(os.environ["LMSYS_DATA_DIR"])
    candidates = [
        Path("/kaggle/input/llm-classification-finetuning"),
        Path("data/raw"),
        Path("../data/raw"),
        Path("../../data/raw"),
    ]
    for c in candidates:
        if (c / "train.csv").exists():
            return c
    return candidates[1]


DATA_DIR = _resolve_data_dir()
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else Path(".")
CACHE_DIR = WORK_DIR / "cache"
RESULTS_DIR = WORK_DIR / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_GPU = torch.cuda.device_count()
DEVICES = [f"cuda:{i}" for i in range(N_GPU)] if N_GPU else ["cpu"]


def pick_dtype(device_index: int = 0) -> torch.dtype:
    """bfloat16 needs compute capability >= 8; Kaggle's T4 is 7.5, so fp16 there.

    Using bf16 on a T4 does not error loudly - it silently falls back to a slow
    emulated path or crashes deep inside a kernel, so the check is worth making
    explicit.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device_index)
    return torch.bfloat16 if major >= 8 else torch.float16


DTYPE = pick_dtype()

print(f"среда      : {'Kaggle' if IS_KAGGLE else 'локально'}")
print(f"данные     : {DATA_DIR}  (train.csv {'найден' if (DATA_DIR / 'train.csv').exists() else 'НЕ НАЙДЕН'})")
print(f"кэш        : {CACHE_DIR.resolve()}")
print(f"результаты : {RESULTS_DIR.resolve()}")
print(f"GPU        : {N_GPU}")
for i in range(N_GPU):
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / 2**30
    cc = torch.cuda.get_device_capability(i)
    print(f"  cuda:{i}  {name}  {total:.1f} GiB  compute capability {cc[0]}.{cc[1]}")
print(f"dtype      : {DTYPE}  ({'bf16 доступен' if DTYPE == torch.bfloat16 else 'bf16 недоступен, работаем в fp16'})")

# %% [markdown]
# ## 3. Данные
#
# `train.csv` содержит 57 477 «битв». В каждой — промпт пользователя и два ответа
# разных моделей; люди голосовали, какой лучше, либо ставили ничью.
#
# Три текстовые колонки — это **JSON-строки со списками**, по элементу на ход
# диалога. Внутри могут быть `null` (модель ничего не вернула) — заменяем на
# пустую строку.
#
# Колонки `model_a` / `model_b` **не используем как признак**: в тесте их нет.
# Три колонки-таргета схлопываем в одну: `label` ∈ {0, 1, 2} = {a, b, ничья}.

# %%
TEXT_COLUMNS = ("prompt", "response_a", "response_b")
TARGET_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")


def parse_turns(raw: str) -> list[str]:
    """Decode one JSON cell into a list of turn strings, `null` becoming ''."""
    turns = json.loads(raw)
    return ["" if t is None else str(t) for t in turns]


def load_train(data_dir: Path) -> pd.DataFrame:
    """Read train.csv, decode the JSON columns and build the integer label."""
    path = data_dir / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"train.csv не найден в {data_dir}. На Kaggle: Add Input -> Competitions "
            f"-> llm-classification-finetuning. Локально: положи файл в data/raw/ "
            f"или задай переменную окружения LMSYS_DATA_DIR."
        )
    df = pd.read_csv(path)
    for column in TEXT_COLUMNS:
        df[f"{column}_turns"] = df[column].map(parse_turns)

    targets = df[list(TARGET_COLUMNS)].to_numpy()
    if not (targets.sum(axis=1) == 1).all():
        raise ValueError("ожидался ровно один флаг победителя в каждой строке")
    df["label"] = targets.argmax(axis=1)
    return df.drop(columns=list(TEXT_COLUMNS))


train = load_train(DATA_DIR)
print(f"строк: {len(train)}")
print(train["label"].value_counts(normalize=True).sort_index().rename({0: "a", 1: "b", 2: "tie"}))

# %% [markdown]
# ## 4. `folds.csv` — разбиение на фолды
#
# Чтобы честно измерить качество, часть данных нельзя показывать при обучении.
# K-fold: режем на 5 частей, обучаемся 5 раз, каждый раз одна часть отложена.
# Собрав предсказания со всех прогонов, получаем **OOF** — предсказание для
# каждой строки от модели, которая эту строку не видела.
#
# Разбиение лежит в **отдельном файле** и делается один раз на весь проект.
# Если zero-shot, эта задача и LoRA нарежут фолды по-разному, их log loss'ы
# станут несравнимы, а блендинг OOF-файлов потечёт: строка из твоей валидации
# окажется в чужом обучении.
#
# Файл: `id,fold`, где `fold` ∈ {0..4}. Он же задаёт **канонический порядок
# строк** — кэш эмбеддингов пишется строго в этом порядке, поэтому строка `i` в
# `.npy` всегда соответствует строке `i` в таблице.
#
# Ячейка ниже **не перезаписывает** существующий файл. Если он есть — читает.

# %%
FOLDS_PATH_CANDIDATES = [
    DATA_DIR / "folds.csv",
    Path("data/raw/folds.csv"),
    WORK_DIR / "folds.csv",
]


def load_or_create_folds(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, Path]:
    """Read folds.csv if it exists anywhere sensible, otherwise create it once.

    Creating is deliberately a one-time, never-overwriting action: the row order
    of the embedding cache is tied to this file, so re-cutting the folds after an
    extraction would silently misalign every feature vector with its label.
    """
    for path in FOLDS_PATH_CANDIDATES:
        if path.exists():
            folds = pd.read_csv(path)
            print(f"folds.csv прочитан из {path}")
            return folds, path

    from sklearn.model_selection import StratifiedKFold

    target = WORK_DIR / "folds.csv"
    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.empty(len(df), dtype=np.int16)
    for k, (_, val_idx) in enumerate(splitter.split(df, df["label"])):
        fold[val_idx] = k
    folds = pd.DataFrame({"id": df["id"].to_numpy(), "fold": fold})
    folds.to_csv(target, index=False)
    print(f"folds.csv СОЗДАН: {target}  (StratifiedKFold, {cfg.n_folds} фолдов, seed={cfg.seed})")
    print("Это разбиение теперь фиксировано. Раздай файл остальным задачам проекта.")
    return folds, target


folds, FOLDS_PATH = load_or_create_folds(train, CFG)

# Канонический порядок строк = порядок folds.csv. Всё дальше живёт в нём.
data = folds.merge(train, on="id", how="left", validate="one_to_one")
missing = data["label"].isna().sum()
if missing:
    raise ValueError(f"{missing} id из folds.csv отсутствуют в train.csv")
data["label"] = data["label"].astype(int)

if CFG.limit_rows is not None:
    data = data.iloc[: CFG.limit_rows].reset_index(drop=True)
    print(f"ВНИМАНИЕ: limit_rows={CFG.limit_rows}, работаем на подвыборке")

N_ROWS = len(data)
LABELS = data["label"].to_numpy()
FOLD_ID = data["fold"].to_numpy()
print(f"строк в работе: {N_ROWS}")
print(data.groupby("fold")["label"].value_counts(normalize=True).unstack().round(4))

# %% [markdown]
# ## 5. Сборка текста
#
# Модель ничего не генерирует — её задача только **закодировать вход**. Поэтому
# инструкции («оцени», «ответь JSON») не нужны, достаточно разделителей.
#
# Ответы показываем под нейтральными метками `RESPONSE 1` / `RESPONSE 2`, а не
# A / B: метка не должна сама по себе намекать, из какой колонки пришёл текст.
#
# Усечение — по каждому полю отдельно, с **одинаковым бюджетом для A и B**
# (иначе мы бы сами внесли асимметрию в задачу, где вся суть в сравнении).
# Режем середину, а не хвост: отказы, выводы и подписи стоят в конце ответа и
# несут много сигнала о предпочтении.

# %%
TRUNCATION_MARKER = "\n[...]\n"
EMPTY_MARKER = "[empty]"


def truncate_middle(text: str, limit: int) -> str:
    """Shorten to `limit` characters keeping both ends, dropping the middle."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return text[:head] + TRUNCATION_MARKER + text[-tail:]


def select_turns(n_turns: int, max_turns: int) -> list[int]:
    """Pick which turn indices to show, keeping the first and the last ones."""
    if n_turns <= max_turns:
        return list(range(n_turns))
    head = max_turns // 2
    tail = max_turns - head
    return list(range(head)) + list(range(n_turns - tail, n_turns))


# TODO(ivan): формат разделителей и нужен ли chat template. Варианты, которые
# стоит сравнить: голые теги (сейчас), разметка в стиле чата модели через
# tokenizer.apply_chat_template, XML-теги, префикс с описанием задачи.
def build_text(prompt_turns, first_turns, second_turns, cfg: Config) -> str:
    """Lay one battle out as a single flat string for the encoder."""
    indices = select_turns(len(prompt_turns), cfg.max_turns)
    n = max(1, len(indices))
    prompt_limit = max(200, cfg.max_chars_prompt // n)
    response_limit = max(400, cfg.max_chars_response // n)

    blocks: list[str] = []
    for position, i in enumerate(indices):
        if position and i != indices[position - 1] + 1:
            blocks.append("<<SKIPPED TURNS>>")
        blocks.append("<<PROMPT>>\n" + (truncate_middle(prompt_turns[i], prompt_limit) or EMPTY_MARKER))
        blocks.append("<<RESPONSE 1>>\n" + (truncate_middle(first_turns[i], response_limit) or EMPTY_MARKER))
        blocks.append("<<RESPONSE 2>>\n" + (truncate_middle(second_turns[i], response_limit) or EMPTY_MARKER))
    return "\n\n".join(blocks)


def build_texts(df: pd.DataFrame, order: str, cfg: Config) -> list[str]:
    """Render every row in one response order: 'ab' or the swapped 'ba'."""
    if order not in ("ab", "ba"):
        raise ValueError(f"неизвестный порядок: {order}")
    a, b = ("response_a_turns", "response_b_turns")
    first, second = (a, b) if order == "ab" else (b, a)
    return [
        build_text(p, f, s, cfg)
        for p, f, s in zip(df["prompt_turns"], df[first], df[second])
    ]


_demo = build_text(
    data["prompt_turns"].iloc[0], data["response_a_turns"].iloc[0], data["response_b_turns"].iloc[0], CFG
)
print(f"пример текста ({len(_demo)} символов):\n")
print(_demo[:700] + ("\n..." if len(_demo) > 700 else ""))

# %% [markdown]
# ## 6. Ручные фичи
#
# Небольшая матрица простых признаков, посчитанных прямо из текста без всякой
# нейросети: длины, число блоков кода, списков, заголовков, ходов диалога.
#
# Зачем они нужны, даже если мы верим в LLM: они дают **опорную точку**. В конце
# мы обучим голову только на них, без `h`. Если голова на скрытых состояниях не
# бьёт эту опору — значит `h` не несёт полезного сигнала, и это меняет всю
# интерпретацию результата.

# %%
CODE_RE = re.compile(r"```")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)", re.MULTILINE)
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)


def _side_stats(turns_list: list[list[str]]) -> dict[str, np.ndarray]:
    """Cheap per-response statistics, one value per row."""
    joined = ["\n".join(t) for t in turns_list]
    return {
        "chars": np.array([len(t) for t in joined], dtype=np.float64),
        "words": np.array([t.count(" ") + 1 for t in joined], dtype=np.float64),
        "code": np.array([len(CODE_RE.findall(t)) // 2 for t in joined], dtype=np.float64),
        "list": np.array([len(LIST_RE.findall(t)) for t in joined], dtype=np.float64),
        "header": np.array([len(HEADER_RE.findall(t)) for t in joined], dtype=np.float64),
        "empty": np.array([float(len(t.strip()) == 0) for t in joined], dtype=np.float64),
    }


# TODO(ivan): расширение набора ручных фич — читаемость, доля пунктуации,
# язык, повтор промпта в ответе, наличие извинений/отказов, и т.д.
def hand_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the handcrafted feature matrix and the matching column names."""
    a = _side_stats(list(df["response_a_turns"]))
    b = _side_stats(list(df["response_b_turns"]))

    columns: dict[str, np.ndarray] = {}
    for key in a:
        columns[f"a_{key}"] = a[key]
        columns[f"b_{key}"] = b[key]
        columns[f"d_{key}"] = a[key] - b[key]
        # Log-ratio keeps the "twice as long" relation on the same scale whichever
        # side is longer, and +1 keeps empty responses finite.
        columns[f"r_{key}"] = np.log1p(a[key]) - np.log1p(b[key])

    columns["n_turns"] = np.array([len(t) for t in df["prompt_turns"]], dtype=np.float64)
    columns["prompt_chars"] = np.array(
        [sum(len(x) for x in t) for t in df["prompt_turns"]], dtype=np.float64
    )

    names = list(columns)
    matrix = np.stack([columns[k] for k in names], axis=1).astype(np.float32)
    return matrix, names


HAND, HAND_NAMES = hand_features(data)
HAND_PATH = CACHE_DIR / f"hand_{CFG.prompt_version}.npy"
np.save(HAND_PATH, HAND)
print(f"ручных фич: {HAND.shape[1]}  ->  {HAND_PATH.name}")
print(HAND_NAMES)

# %% [markdown]
# ## 7. Бэкбон
#
# Четыре решения, каждое из которых легко испортить:
#
# **`AutoModel`, а не `AutoModelForCausalLM`.** Вторая тащит `lm_head` — матрицу
# `[vocab, hidden]`. У Gemma-2 словарь 256 тысяч токенов, это лишние ~1.8 ГБ
# весов, которые нам не нужны: мы ничего не генерируем.
#
# **`padding_side="left"`.** Тогда последний реальный токен всегда стоит на
# позиции `-1`, и пуллинг по последнему токену — это просто `h[:, -1, :]`.
# С правым паддингом пришлось бы вычислять индекс по маске для каждой строки.
#
# **`.eval()`.** Без него dropout внутри бэкбона сделает `h` невоспроизводимым:
# тот же вход дал бы разные векторы.
#
# **`attn_implementation`.** У Gemma-2 в внимании есть soft-capping, который
# несовместим с Flash Attention, поэтому ей нужен `eager`. Остальным моделям
# быстрее `sdpa`.
#
# Ещё про T4: compute capability 7.5 — bfloat16 не поддерживается, только fp16.

# %%
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

print("transformers", transformers.__version__)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


# transformers renamed `torch_dtype` to `dtype` in 4.56. Passing the wrong one is
# not a loud error - the argument lands in **kwargs and the model silently loads
# in float32, which then OOMs. So pick the right keyword by version.
DTYPE_KW = "dtype" if _version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
print(f"ключевое слово для dtype в этой версии: {DTYPE_KW}")


def resolve_model_path(cfg: Config) -> str:
    """Locate the backbone: explicit override, attached Kaggle model, or the hub."""
    if cfg.model_path_override:
        return cfg.model_path_override

    search_roots = [Path("/kaggle/input"), Path("D:/PMLDL/models"), Path("models")]
    hint = cfg.model_hint.lower().replace("_", "-")
    for root in search_roots:
        if not root.exists():
            continue
        for config_file in sorted(root.glob("**/config.json")):
            folder = str(config_file.parent).lower().replace("_", "-")
            if hint in folder:
                print(f"бэкбон найден локально: {config_file.parent}")
                return str(config_file.parent)

    print(f"локально не найдено, качаем с HuggingFace: {cfg.model_fallback}")
    if IS_KAGGLE:
        print("  если сейчас будет ошибка сети - включи Settings -> Internet: On,")
        print("  либо подключи модель через Add Input -> Models и укажи её в model_hint")
    return cfg.model_fallback


MODEL_PATH = resolve_model_path(CFG)
MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_PATH)
MODEL_TAG = re.sub(r"[^a-z0-9]+", "-", Path(str(MODEL_PATH)).name.lower()).strip("-")

HIDDEN_SIZE = MODEL_CONFIG.hidden_size
N_LAYERS = MODEL_CONFIG.num_hidden_layers
MAX_CTX = getattr(MODEL_CONFIG, "max_position_embeddings", None)
N_PARAMS = getattr(MODEL_CONFIG, "num_parameters", None)

print(f"модель        : {MODEL_PATH}")
print(f"тег для кэша  : {MODEL_TAG}")
print(f"hidden_size H : {HIDDEN_SIZE}")
print(f"слоёв         : {N_LAYERS}")
print(f"макс. контекст: {MAX_CTX}")

# %%
# Индексы слоёв, которые сохраняем. hidden_states — кортеж из N_LAYERS + 1
# тензоров: нулевой это эмбеддинги входа, последний — выход после финальной
# нормализации модели.
LAYER_INDEX = {
    name: max(1, min(N_LAYERS, int(round(frac * N_LAYERS))))
    for name, frac in CFG.layer_names().items()
}
print("сохраняем слои:", LAYER_INDEX, f"(из {N_LAYERS + 1} доступных)")

# Промежуточные слои НЕ прошли финальный RMSNorm, их масштаб отличается от
# последнего слоя на порядки. Мы их не нормируем здесь - нормализация стоит
# первым слоем головы (LayerNorm), там ей и место.

VARIANTS = [(l, p) for l in LAYER_INDEX for p in CFG.pools]
CACHE_BYTES = N_ROWS * len(CFG.orders) * len(VARIANTS) * HIDDEN_SIZE * 2
print(f"вариантов на строку: {len(VARIANTS)} x {len(CFG.orders)} порядка = {len(VARIANTS) * len(CFG.orders)}")
print(f"объём кэша: {N_ROWS} x {len(CFG.orders)} x {len(VARIANTS)} x {HIDDEN_SIZE} x 2 байта = {CACHE_BYTES / 2**30:.2f} ГиБ")
if IS_KAGGLE and CACHE_BYTES > 18 * 2**30:
    print("!! это больше, чем /kaggle/working обычно выдерживает (~20 ГБ)")

# %%
def pick_attn_implementation(model_path: str, cfg: Config) -> str:
    """Gemma-2 soft-caps its attention logits, which flash/sdpa kernels reject."""
    if cfg.attn_implementation:
        return cfg.attn_implementation
    name = str(model_path).lower().replace("_", "-")
    return "eager" if "gemma-2" in name else "sdpa"


ATTN = pick_attn_implementation(MODEL_PATH, CFG)


def decide_quantization(cfg: Config, total_memory: int | None = None) -> bool:
    """4-bit only when the fp16 weights would not fit one GPU with room to spare.

    Every replica must fit on its own card: we run one model per GPU, not one
    model split across both. A 9B backbone in fp16 is ~18 GB and does not fit a
    16 GB T4, so it gets quantized; a 0.5B one is left alone.
    """
    if cfg.load_in_4bit is not None:
        return cfg.load_in_4bit
    if total_memory is None:
        if not torch.cuda.is_available():
            return False
        total_memory = torch.cuda.get_device_properties(0).total_memory
    approx_params = HIDDEN_SIZE * HIDDEN_SIZE * 12 * N_LAYERS + HIDDEN_SIZE * MODEL_CONFIG.vocab_size
    return approx_params * 2 > total_memory * 0.55


USE_4BIT = decide_quantization(CFG)
print(f"attn_implementation: {ATTN}")
print(f"4-bit квантизация  : {USE_4BIT}")


def default_batch_size(
    cfg: Config, total_memory: int | None = None, use_4bit: bool | None = None
) -> int:
    """Size the batch from the activation cost of output_hidden_states=True.

    Asking for every hidden state materialises N_LAYERS+1 tensors of
    [B, T, H] in fp16 at once, which for a 9B model at T=1024 is gigabytes -
    usually a bigger constraint than the weights themselves.
    """
    if cfg.batch_size:
        return cfg.batch_size
    if total_memory is None:
        if not torch.cuda.is_available():
            return 2
        total_memory = torch.cuda.get_device_properties(0).total_memory
    if use_4bit is None:
        use_4bit = USE_4BIT
    weights = total_memory * (0.42 if use_4bit else 0.75)
    # output_hidden_states=True keeps every one of the N_LAYERS+1 activation
    # tensors alive at once, which for a 9B model dwarfs the weights. The factor
    # of 3 leaves room for attention scores and MLP intermediates on top.
    per_row = (N_LAYERS + 1) * cfg.max_length * HIDDEN_SIZE * 2
    room = max(total_memory - weights, total_memory * 0.15)
    return int(max(1, min(32, room // (per_row * 3))))


BATCH_SIZE = default_batch_size(CFG)
print(f"batch_size         : {BATCH_SIZE}")

# %%
def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Left padding puts the last real token at index -1 for every row."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(model_path: str, device: str) -> nn.Module:
    """Load one frozen encoder replica pinned to a single device."""
    kwargs: dict = {DTYPE_KW: DTYPE, "attn_implementation": ATTN}
    if USE_4BIT:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "для 4-bit нужен bitsandbytes (`pip install bitsandbytes`, нужен интернет). "
                "Либо возьми модель поменьше, либо поставь load_in_4bit=False в конфиге, "
                "если веса помещаются в карту целиком."
            ) from error
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )
    if device != "cpu":
        # device_map pins the whole model to this one GPU. That is deliberate -
        # see the note on data vs model parallelism in the next section.
        kwargs["device_map"] = {"": int(device.split(":")[1])}

    model = AutoModel.from_pretrained(model_path, **kwargs)
    model.eval().requires_grad_(False)
    if device == "cpu":
        model.to("cpu")
    return model


TOKENIZER = load_tokenizer(MODEL_PATH)
print(f"токенизатор загружен, padding_side={TOKENIZER.padding_side}, vocab={len(TOKENIZER)}")

# %% [markdown]
# ## 8. Две видеокарты: почему `device_map="auto"` — не то, что нужно
#
# На Kaggle T4×2 есть два принципиально разных способа занять обе карты.
#
# ### Model parallel (`device_map="auto"`)
#
# Accelerate режет модель по слоям: первая половина на `cuda:0`, вторая на
# `cuda:1`. Батч проходит слои по очереди — сначала работает нулевая карта,
# первая ждёт, потом наоборот. Это **naive model parallelism**: в каждый момент
# считает только одна GPU. Ускорения нет, суммарная пропускная способность —
# как у одной карты. Смысл в другом: так помещается модель, которая в одну карту
# не влезает.
#
# ### Data parallel (то, что делаем мы)
#
# Модель влезает в одну карту (в fp16 для маленькой, в 4-bit для 9B), поэтому мы
# грузим **две независимые копии** — по одной на каждую GPU — и раздаём им разные
# строки данных. Обе карты считают одновременно, пропускная способность ×2.
#
# Управляем двумя потоками Python. GIL здесь не мешает: операции PyTorch на CUDA
# отпускают его на время работы ядра, поток только ставит задачи в очередь.
#
# ```
#              строки 0..255      строки 256..511
#                    |                   |
#              [ поток 0 ]         [ поток 1 ]
#                    |                   |
#            копия модели на      копия модели на
#                cuda:0               cuda:1
#                    \                  /
#                     -> общие .npy на диске (непересекающиеся строки)
# ```
#
# Если карта одна — всё то же самое в один поток, код не меняется.

# %%
def pool_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Last real token. Valid only because the tokenizer pads on the left."""
    return hidden[:, -1, :]


def pool_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens only; padding must not dilute the average."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


POOL_FN = {"last": pool_last, "mean": pool_mean}


def cache_path(order: str, layer: str, pool: str) -> Path:
    return CACHE_DIR / f"emb_{MODEL_TAG}_{CFG.prompt_version}_{order}_{layer}_{pool}.npy"


def done_path(order: str) -> Path:
    return CACHE_DIR / f"done_{MODEL_TAG}_{CFG.prompt_version}_{order}.npy"


def meta_path(order: str, layer: str, pool: str) -> Path:
    return cache_path(order, layer, pool).with_suffix(".json")


def open_writer(order: str, layer: str, pool: str) -> np.memmap:
    """Open (creating on first use) the memory-mapped .npy for one variant.

    A memmap is what makes the run resumable and RAM-independent: rows are
    written straight to disk at their absolute index, so any subset can be filled
    in any order across any number of sessions.
    """
    path = cache_path(order, layer, pool)
    if not path.exists():
        return np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(N_ROWS, HIDDEN_SIZE)
        )
    # For an existing file open_memmap reads the shape from the header and
    # ignores what we pass, so a mismatch would corrupt the row alignment
    # silently. Catch it here instead.
    existing = np.lib.format.open_memmap(path, mode="r+")
    if existing.shape != (N_ROWS, HIDDEN_SIZE):
        raise ValueError(
            f"{path.name} имеет форму {existing.shape}, а сейчас нужно "
            f"{(N_ROWS, HIDDEN_SIZE)}. Похоже, изменился limit_rows, folds.csv или модель. "
            f"Удали кэш вручную или верни прежний конфиг - молча перезаписывать не буду."
        )
    return existing


def write_meta(order: str, layer: str, pool: str, n_done: int) -> None:
    meta = {
        "model_path": str(MODEL_PATH),
        "model_tag": MODEL_TAG,
        "prompt_version": CFG.prompt_version,
        "order": order,
        "layer_name": layer,
        "layer_index": LAYER_INDEX[layer],
        "n_layers": N_LAYERS,
        "pool": pool,
        "hidden_size": HIDDEN_SIZE,
        "n_rows": N_ROWS,
        "n_rows_done": int(n_done),
        "max_length": CFG.max_length,
        "max_chars_prompt": CFG.max_chars_prompt,
        "max_chars_response": CFG.max_chars_response,
        "max_turns": CFG.max_turns,
        "dtype": str(DTYPE),
        "load_in_4bit": USE_4BIT,
        "attn_implementation": ATTN,
        "folds_file": str(FOLDS_PATH),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path(order, layer, pool).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_done(order: str) -> np.ndarray:
    """Boolean mask of rows already extracted for this order."""
    path = done_path(order)
    if path.exists():
        mask = np.load(path)
        if mask.shape == (N_ROWS,):
            return mask
        print(f"маска {path.name} не того размера ({mask.shape}), начинаем заново")
    return np.zeros(N_ROWS, dtype=bool)

# %%
_write_lock = threading.Lock()
_batch_lock = threading.Lock()
_batch_state = {"size": BATCH_SIZE}
_MODEL_CACHE: dict[str, nn.Module] = {}
_TOKENIZER_CACHE: dict[int, "AutoTokenizer"] = {}
_TEXT_CACHE: dict[str, list[str]] = {}


def get_tokenizer(worker_id: int) -> "AutoTokenizer":
    """One tokenizer instance per worker thread - never share one.

    Fast tokenizers are backed by Rust structures whose padding and truncation
    settings are mutated on every call. Calling one from two threads raises
    `RuntimeError: Already borrowed` at random, which on a multi-hour extraction
    means losing the run to a race that did not show up in any short test.
    See huggingface/tokenizers#537.
    """
    if worker_id not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[worker_id] = load_tokenizer(MODEL_PATH)
    return _TOKENIZER_CACHE[worker_id]


def get_models() -> dict[str, nn.Module]:
    """One frozen replica per device, loaded once and reused.

    Loading a quantized 9B backbone costs minutes, and the smoke test plus both
    response orders would otherwise pay it three times over.
    """
    for device in DEVICES:
        if device not in _MODEL_CACHE:
            print(f"  загружаю копию бэкбона на {device} ...", flush=True)
            _MODEL_CACHE[device] = load_backbone(MODEL_PATH, device)
    return _MODEL_CACHE


def release_models() -> None:
    """Free the backbone replicas; part B does not need them."""
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shrink_batch() -> int:
    """Halve the shared batch size after an out-of-memory error.

    The estimate that produced the starting batch size is a heuristic, and a run
    that dies at hour three because one chunk of unusually long rows overflowed
    is the expensive kind of failure. Shrinking is global and permanent for the
    session: both workers back off together, and nothing grows back.
    """
    with _batch_lock:
        _batch_state["size"] = max(1, _batch_state["size"] // 2)
        return _batch_state["size"]


def get_texts(order: str) -> list[str]:
    if order not in _TEXT_CACHE:
        _TEXT_CACHE[order] = build_texts(data, order, CFG)
    return _TEXT_CACHE[order]


def _extract_chunk(
    rows: np.ndarray,
    texts: list[str],
    model: nn.Module,
    tokenizer: "AutoTokenizer",
    device: str,
    writers: dict[tuple[str, str], np.memmap],
) -> None:
    """Encode one chunk of rows and write every variant to its memmap."""
    # Sorting by length inside the chunk means each batch pads to roughly the
    # length of its own longest row rather than the longest row in the dataset.
    # Rows are written back by absolute index, so the canonical order is kept.
    order_by_length = np.argsort([len(texts[r]) for r in rows], kind="stable")
    rows = rows[order_by_length]

    start = 0
    while start < len(rows):
        batch_rows = rows[start : start + _batch_state["size"]]
        batch_texts = [texts[r] for r in batch_rows]
        try:
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CFG.max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.inference_mode():
                out = model(**encoded, output_hidden_states=True)

            for layer_name, layer_idx in LAYER_INDEX.items():
                hidden = out.hidden_states[layer_idx]
                for pool_name in CFG.pools:
                    vectors = POOL_FN[pool_name](hidden, encoded["attention_mask"])
                    writers[(layer_name, pool_name)][batch_rows] = (
                        vectors.float().cpu().numpy().astype(np.float16)
                    )
            del out, encoded
        except torch.cuda.OutOfMemoryError:
            if len(batch_rows) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"  OOM на батче {len(batch_rows)}, уменьшаю до {shrink_batch()}", flush=True)
            continue
        start += len(batch_rows)


def run_extraction(order: str, chunk_limit: int | None = None, verbose: bool = True) -> dict:
    """Extract one response order, resuming from whatever is already on disk."""
    texts = get_texts(order)
    done = load_done(order)

    chunks = [
        np.arange(s, min(s + CFG.chunk_rows, N_ROWS))
        for s in range(0, N_ROWS, CFG.chunk_rows)
    ]
    todo = [c for c in chunks if not done[c].all()]
    if chunk_limit is not None:
        todo = todo[:chunk_limit]

    if not todo:
        if verbose:
            print(f"[{order}] всё уже в кэше ({int(done.sum())}/{N_ROWS} строк)")
        return {"order": order, "rows": 0, "seconds": 0.0, "done": int(done.sum())}

    n_workers = len(DEVICES)
    if verbose:
        print(f"[{order}] к обработке {len(todo)} чанков на {n_workers} устройствах: {DEVICES}")

    models = get_models()
    if verbose and torch.cuda.is_available():
        for i in range(N_GPU):
            used = torch.cuda.memory_allocated(i) / 2**30
            print(f"  cuda:{i}: занято {used:.2f} GiB после загрузки копии модели")

    processed = {"rows": 0}
    started = time.time()

    def worker(worker_id: int) -> None:
        device = DEVICES[worker_id]
        if device != "cpu":
            # Pin this thread's default CUDA device so any kernel that relies on
            # the ambient context lands on the same card as the model.
            torch.cuda.set_device(int(device.split(":")[1]))
        model = models[device]
        tokenizer = get_tokenizer(worker_id)
        # Each thread opens its own memmap views. Threads only ever touch rows
        # from their own chunks, so the writes never overlap.
        writers = {(l, p): open_writer(order, l, p) for l, p in VARIANTS}
        for chunk_id in range(worker_id, len(todo), n_workers):
            rows = todo[chunk_id]
            _extract_chunk(rows, texts, model, tokenizer, device, writers)
            with _write_lock:
                done[rows] = True
                processed["rows"] += len(rows)
                np.save(done_path(order), done)
                if verbose:
                    elapsed = time.time() - started
                    rate = processed["rows"] / max(elapsed, 1e-9)
                    remaining = (N_ROWS - int(done.sum())) / max(rate, 1e-9)
                    print(
                        f"  [{order}] {int(done.sum())}/{N_ROWS} строк | "
                        f"{rate:.2f} строк/с | осталось ~{remaining / 3600:.2f} ч",
                        flush=True,
                    )
        for writer in writers.values():
            writer.flush()

    if n_workers == 1:
        worker(0)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(worker, range(n_workers)))

    elapsed = time.time() - started
    for layer_name, pool_name in VARIANTS:
        write_meta(order, layer_name, pool_name, int(done.sum()))

    return {
        "order": order,
        "rows": processed["rows"],
        "seconds": elapsed,
        "done": int(done.sum()),
    }

# %% [markdown]
# ## 9. Смоук-тест и проекция
#
# Прежде чем запускать многочасовой прогон, надо измерить реальную скорость на
# маленьком куске и посчитать, во что выльется полный проход. Это единственный
# способ не обнаружить на седьмом часу, что не успеваем.

# %%
SMOKE_CHUNKS = 1  # один чанк на каждое устройство

smoke = run_extraction(CFG.orders[0], chunk_limit=len(DEVICES) * SMOKE_CHUNKS)

if smoke["rows"]:
    rate = smoke["rows"] / smoke["seconds"]
    total_passes = N_ROWS * len(CFG.orders)
    projected_hours = total_passes / rate / 3600
    print()
    print(f"обработано  : {smoke['rows']} строк за {smoke['seconds']:.1f} с")
    print(f"скорость    : {rate:.2f} строк/с на {len(DEVICES)} устройствах")
    print(f"полный объём: {N_ROWS} строк x {len(CFG.orders)} порядка = {total_passes} проходов")
    print(f"ПРОЕКЦИЯ    : ~{projected_hours:.2f} часов")
    print(f"объём кэша  : {CACHE_BYTES / 2**30:.2f} ГиБ")
    if IS_KAGGLE and projected_hours > 8.0:
        print()
        print("!! Не влезает в одну сессию Kaggle (~9 ч).")
        print("   Варианты: уменьшить max_length, взять orders=('ab',), задать limit_rows,")
        print("   выбрать модель поменьше, или просто запустить ноутбук ещё раз -")
        print("   экстракция продолжится с места обрыва.")
else:
    print("кэш уже полон, мерить нечего")

# %% [markdown]
# ## 10. Полный прогон экстракции
#
# Ячейка ниже — единственное дорогое место в ноутбуке. Она защищена флагом:
# посмотри на проекцию выше и осознанно разреши прогон.
#
# Прерывание безопасно. Маска готовых строк пишется после каждого чанка, при
# следующем запуске работа продолжится с того же места.

# %%
RUN_FULL_EXTRACTION = True  # поставь False, если хочешь остановиться после смоук-теста

if RUN_FULL_EXTRACTION:
    for order in CFG.orders:
        stats = run_extraction(order)
        print(f"[{order}] готово: {stats['done']}/{N_ROWS} строк, {stats['seconds'] / 60:.1f} мин")
else:
    print("полный прогон выключен флагом RUN_FULL_EXTRACTION")

# Освобождаем GPU: дальше бэкбон не нужен вообще.
release_models()
if torch.cuda.is_available():
    for i in range(N_GPU):
        print(f"cuda:{i}: занято {torch.cuda.memory_allocated(i) / 2**30:.2f} GiB после освобождения")

# %% [markdown]
# ---
# # Часть B — обучение головы
#
# Отсюда и до конца бэкбон не нужен. Всё читается из `cache/` и работает на CPU
# за секунды. Это те ячейки, которые ты будешь гонять сотни раз, меняя конфиг.

# %%
READY = np.ones(N_ROWS, dtype=bool)
for order in (CFG.orders if CFG.order_combine != "ab" else ("ab",)):
    READY &= load_done(order)

n_ready = int(READY.sum())
print(f"строк с готовыми признаками: {n_ready}/{N_ROWS} ({100 * n_ready / N_ROWS:.1f}%)")
if n_ready == 0:
    raise RuntimeError("кэш пуст - сначала прогони экстракцию")

# %% [markdown]
# ## 11. Сборка входа головы
#
# Вход собирается из блоков, набор задаётся конфигом: один или несколько
# сохранённых векторов `h`, опционально ручные фичи, опционально второй порядок.
# `in_dim` считается автоматически — хардкодить его нельзя, иначе смена конфига
# молча сломает голову.
#
# Про два порядка. Каждую строку мы прогнали дважды: `(a, b)` и `(b, a)`. Это
# **TTA** — test-time augmentation. Усреднение двух векторов гасит позиционный
# сдвиг модели (склонность хвалить того, кто показан первым). `concat` сохраняет
# больше информации, но удваивает `in_dim`.

# %%
def assemble_features(cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Concatenate the configured cache blocks into one float32 matrix."""
    blocks: list[np.ndarray] = []
    names: list[str] = []

    for layer in cfg.feature_layers:
        ab = np.asarray(np.load(cache_path("ab", layer, cfg.feature_pool), mmap_mode="r"), dtype=np.float32)
        if cfg.order_combine == "ab":
            blocks.append(ab)
            names.append(f"{layer}/{cfg.feature_pool}/ab")
            continue
        ba = np.asarray(np.load(cache_path("ba", layer, cfg.feature_pool), mmap_mode="r"), dtype=np.float32)
        if cfg.order_combine == "mean":
            blocks.append((ab + ba) * 0.5)
            names.append(f"{layer}/{cfg.feature_pool}/mean(ab,ba)")
        elif cfg.order_combine == "concat":
            blocks.append(np.concatenate([ab, ba], axis=1))
            names.append(f"{layer}/{cfg.feature_pool}/concat(ab,ba)")
        else:
            raise ValueError(f"неизвестный order_combine: {cfg.order_combine}")

    if cfg.use_hand_features:
        blocks.append(HAND)
        names.append(f"hand({HAND.shape[1]})")

    matrix = np.concatenate(blocks, axis=1)
    return matrix, names


X_FULL, BLOCK_NAMES = assemble_features(CFG)
print(f"вход головы: {X_FULL.shape}  in_dim={X_FULL.shape[1]}")
print("блоки:", BLOCK_NAMES)
print(f"память: {X_FULL.nbytes / 2**30:.2f} ГиБ")

# %% [markdown]
# ## 12. Голова
#
# `LayerNorm` на входе **обязателен**. Промежуточные слои бэкбона не проходили
# финальную нормализацию, их масштабы отличаются на порядки; без нормализации
# градиенты разлетаются и обучение разваливается на первых шагах.

# %%
class PreferenceHead(nn.Module):
    """LayerNorm -> [Linear -> GELU -> Dropout] x n -> Linear(3)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, n_layers: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        width = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_probe = PreferenceHead(X_FULL.shape[1], CFG.hidden, CFG.dropout, CFG.n_head_layers)
print(_probe)
print(f"обучаемых параметров: {sum(p.numel() for p in _probe.parameters()):,}")
del _probe

# %% [markdown]
# ## 13. Обучение по фолдам
#
# Главное правило: **стандартизация считается только на train-части фолда**.
# Если посчитать среднее и дисперсию по всему датасету, в них попадёт информация
# из отложенной части — валидация станет оптимистичной. Это самая частая утечка
# в табличных пайплайнах, и она почти не видна: метрика просто чуть лучше, чем
# на самом деле.

# %%
def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std on the training part only, then apply to both."""
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Train one head with early stopping; return val probabilities and history."""
    set_seed(cfg.seed)
    x_train, x_val = standardize(x_train, x_val)

    xt = torch.from_numpy(np.ascontiguousarray(x_train))
    yt = torch.from_numpy(y_train).long()
    xv = torch.from_numpy(np.ascontiguousarray(x_val))
    yv = torch.from_numpy(y_val).long()

    model = PreferenceHead(xt.shape[1], cfg.hidden, cfg.dropout, cfg.n_head_layers)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(xt) / cfg.head_batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps)
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    best_loss, best_probs, best_epoch, history = float("inf"), None, -1, []

    for epoch in range(cfg.epochs):
        model.train()
        permutation = torch.randperm(len(xt), generator=generator)
        running = 0.0
        for start in range(0, len(xt), cfg.head_batch_size):
            index = permutation[start : start + cfg.head_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[index]), yt[index])
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(index)

        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_loss = nn.functional.cross_entropy(logits, yv).item()
            probs = torch.softmax(logits, dim=1).numpy()

        history.append({"epoch": epoch, "train_loss": running / len(xt), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_probs, best_epoch = val_loss, probs, epoch
        elif epoch - best_epoch >= cfg.patience:
            break

    return best_probs, history


def run_cv(x: np.ndarray, cfg: Config, tag: str = "") -> tuple[np.ndarray, list]:
    """Train one head per fold, returning OOF probabilities for ready rows."""
    oof = np.full((N_ROWS, 3), np.nan, dtype=np.float64)
    histories = []
    started = time.time()

    for fold in range(cfg.n_folds):
        val_mask = READY & (FOLD_ID == fold)
        train_mask = READY & (FOLD_ID != fold)
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        probs, history = train_one_fold(
            x[train_mask], LABELS[train_mask], x[val_mask], LABELS[val_mask], cfg
        )
        oof[val_mask] = probs
        histories.append(history)
        print(
            f"  {tag}fold {fold}: train {int(train_mask.sum())} / val {int(val_mask.sum())}"
            f" | лучший val log loss {min(h['val_loss'] for h in history):.5f}"
            f" на эпохе {min(range(len(history)), key=lambda i: history[i]['val_loss'])}"
        )

    print(f"  {tag}всего {time.time() - started:.1f} с")
    return oof, histories

# %% [markdown]
# ## 14. Метрики
#
# Метрика соревнования — multi-class log loss. Вероятности клипуются в
# `[1e-15, 1-1e-15]` и перенормируются: один-единственный уверенный промах с
# `p = 0` даёт бесконечный лосс и уничтожает всю таблицу.

# %%
EPS = 1e-15


def clip_renorm(probs: np.ndarray) -> np.ndarray:
    """Clip into the safe range, then renormalise so each row sums to one."""
    clipped = np.clip(probs, EPS, 1 - EPS)
    return clipped / clipped.sum(axis=1, keepdims=True)


def log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    safe = clip_renorm(probs)
    return float(-np.log(safe[np.arange(len(y)), y]).mean())


def prior_baseline(y: np.ndarray) -> float:
    """Log loss of always predicting the training class frequencies."""
    priors = np.bincount(y, minlength=3) / len(y)
    return log_loss(y, np.tile(priors, (len(y), 1)))


def non_tie_scores(y: np.ndarray, probs: np.ndarray) -> tuple[float, float, int]:
    """Accuracy and AUC on battles humans did not call a tie.

    A ranking metric cannot be fixed or broken by calibration, so a healthy AUC
    next to a poor log loss means the signal is there but badly scaled.
    """
    from sklearn.metrics import roc_auc_score

    decisive = y != 2
    if decisive.sum() == 0:
        return float("nan"), float("nan"), 0
    y_bin = (y[decisive] == 1).astype(int)
    margin = probs[decisive, 1] - probs[decisive, 0]
    accuracy = float(((margin > 0).astype(int) == y_bin).mean())
    auc = float(roc_auc_score(y_bin, margin)) if len(np.unique(y_bin)) > 1 else float("nan")
    return accuracy, auc, int(decisive.sum())


def find_zeroshot_oof() -> Path | None:
    """Look for a neighbouring task's OOF file to compare against."""
    for path in [
        RESULTS_DIR / "oof_zeroshot.csv",
        Path("results/predictions/oof_zeroshot.csv"),
        Path("results/oof_zeroshot.csv"),
    ]:
        if path.exists():
            return path
    return None

# %% [markdown]
# ## 15. Прогоны
#
# Три конфигурации, три строки в таблице:
#
# 1. **основная** — как задано в конфиге;
# 2. **без TTA** — только порядок `ab`, показывает вклад усреднения порядков;
# 3. **только ручные фичи** — без `h` вообще.
#
# Третья строка обязательна. Если голова на скрытых состояниях не бьёт голову на
# одних длинах и счётчиках markdown, значит LLM не дала ничего, и вывод по всей
# задаче другой.

# %%
from dataclasses import replace

runs: dict[str, np.ndarray] = {}

print("основной прогон")
runs["main"], MAIN_HISTORIES = run_cv(X_FULL, CFG, tag="")

print("\nбез усреднения порядков (только ab)")
cfg_no_tta = replace(CFG, order_combine="ab")
x_no_tta, _ = assemble_features(cfg_no_tta)
runs["no_tta"], _ = run_cv(x_no_tta, cfg_no_tta, tag="")
del x_no_tta

print("\nтолько ручные фичи, без h")
cfg_hand = replace(CFG, feature_layers=(), use_hand_features=True)
x_hand, _ = assemble_features(cfg_hand)
runs["hand_only"], _ = run_cv(x_hand, cfg_hand, tag="")
del x_hand
gc.collect()

# %%
y_ready = LABELS[READY]
rows = []

rows.append({"что": "log loss OOF (голова на h + ручные)", "значение": log_loss(y_ready, runs["main"][READY])})
rows.append({"что": "log loss константы на априорных частотах", "значение": prior_baseline(y_ready)})
rows.append({"что": "log loss равномерного предсказания", "значение": float(-np.log(1 / 3))})

zeroshot_path = find_zeroshot_oof()
if zeroshot_path:
    zs = pd.read_csv(zeroshot_path)
    merged = data[["id"]].merge(zs, on="id", how="left")
    zs_probs = merged[list(TARGET_COLUMNS)].to_numpy()
    zs_mask = READY & ~np.isnan(zs_probs).any(axis=1)
    if zs_mask.sum():
        rows.append({
            "что": f"log loss zero-shot (n={int(zs_mask.sum())})",
            "значение": log_loss(LABELS[zs_mask], zs_probs[zs_mask]),
        })
else:
    rows.append({"что": "log loss zero-shot", "значение": float("nan")})

accuracy, auc, n_decisive = non_tie_scores(y_ready, runs["main"][READY])
rows.append({"что": f"accuracy без ничьих (n={n_decisive})", "значение": accuracy})
rows.append({"что": "ROC-AUC без ничьих", "значение": auc})
rows.append({"что": "log loss без усреднения порядков", "значение": log_loss(y_ready, runs["no_tta"][READY])})
rows.append({"что": "log loss только на ручных фичах", "значение": log_loss(y_ready, runs["hand_only"][READY])})

METRICS = pd.DataFrame(rows)
METRICS["значение"] = METRICS["значение"].round(5)
print(f"строк в оценке: {int(READY.sum())}\n")
print(METRICS.to_string(index=False))

# %% [markdown]
# ## 16. Артефакты
#
# `results/oof_head.csv` — формат общий для проекта, он же формат сабмита Kaggle.
# Все строки из `folds.csv`, тот же порядок, суммы равны единице. Строкам без
# готовых признаков проставляются априорные частоты, чтобы формат не поехал;
# метрики при этом считаются только по готовым.

# %%
priors = np.bincount(LABELS[READY], minlength=3) / int(READY.sum())
submission_probs = np.tile(priors, (N_ROWS, 1))
submission_probs[READY] = clip_renorm(runs["main"][READY])

oof = pd.DataFrame(submission_probs, columns=list(TARGET_COLUMNS))
oof.insert(0, "id", data["id"].to_numpy())
oof_path = RESULTS_DIR / "oof_head.csv"
oof.to_csv(oof_path, index=False)

assert list(oof.columns) == ["id"] + list(TARGET_COLUMNS)
assert len(oof) == N_ROWS
assert np.allclose(oof[list(TARGET_COLUMNS)].to_numpy().sum(axis=1), 1.0)
print(f"{oof_path}  ({len(oof)} строк)")
print(oof.head())

# %%
def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small frame as a markdown table without needing `tabulate`."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, divider] + body)


TODOS = [
    "формат разделителей и нужен ли chat template",
    "выбор слоя (l100 / l75 / l50) и пуллинга (last / mean)",
    "усреднение против конкатенации двух порядков",
    "размер hidden, число слоёв головы, dropout",
    "label smoothing",
    "подбор lr",
    "расширение набора ручных фич",
]

lines = [
    "# Голова на замороженном бэкбоне — отчёт",
    "",
    f"Сгенерировано: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    "",
    "## Конфиг",
    "",
    "```json",
    json.dumps(asdict(CFG), indent=2, ensure_ascii=False, default=str),
    "```",
    "",
    "## Среда",
    "",
    f"- бэкбон: `{MODEL_PATH}` (H={HIDDEN_SIZE}, слоёв {N_LAYERS})",
    f"- dtype: {DTYPE}, 4-bit: {USE_4BIT}, attn: {ATTN}",
    f"- устройств: {len(DEVICES)} ({', '.join(DEVICES)})",
    f"- строк с готовыми признаками: {int(READY.sum())} / {N_ROWS}",
    f"- вход головы: in_dim={X_FULL.shape[1]}, блоки {BLOCK_NAMES}",
    f"- сплит: `{FOLDS_PATH}`",
    "",
    "## Метрики",
    "",
    to_markdown_table(METRICS),
    "",
    "## Кривые обучения (основной прогон)",
    "",
    "| фолд | эпох | лучший val log loss | эпоха лучшего |",
    "|---|---|---|---|",
]
for k, history in enumerate(MAIN_HISTORIES):
    best = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    lines.append(f"| {k} | {len(history)} | {history[best]['val_loss']:.5f} | {best} |")

lines += ["", "## Оставленные TODO(ivan)", ""]
lines += [f"- {t}" for t in TODOS]

report_path = RESULTS_DIR / "report_head.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"{report_path}")
print("\n".join(lines[:40]))

# %% [markdown]
# ## Идеи по улучшению — НЕ внесены в код
#
# Это места, где можно выиграть качество или время. Оставляю списком, как
# договорились: исследовательская часть твоя.
#
# **Качество**
#
# 1. **Слой.** Последний слой у декодера заточен под предсказание следующего
#    токена и часто хуже для классификации, чем слой на 60–80% глубины. Кэш уже
#    содержит `l75` и `l50` — сравнение стоит одного прогона части B.
# 2. **Конкатенация слоёв.** `l100 + l75` вместе часто лучше любого по
#    отдельности; `in_dim` удвоится, голова справится.
# 3. **`concat` вместо `mean` для двух порядков.** Разность `h_ab − h_ba` — это
#    прямой сигнал о том, насколько модель чувствительна к позиции; как отдельный
#    блок она может нести больше, чем усреднение.
# 4. **Симметризация.** Обучить на обоих порядках как на разных примерах с
#    переставленными метками (a↔b), затем усреднить предсказания. Удваивает
#    обучающую выборку и жёстко зашивает симметрию задачи.
# 5. **Голова помощнее.** Два скрытых слоя, residual, `hidden=1024`. Дёшево:
#    часть B работает секунды.
# 6. **Больше `max_length`.** 1024 токена режут длинные ответы, а именно в них
#    много сигнала о многословии.
# 7. **Модель побольше.** Gemma-2-9B против Qwen-0.5B — это разница в качестве
#    представлений, а не в размере головы.
# 8. **Калибровка.** Температура на OOF-логитах, подобранная внутри фолда.
#
# **Скорость экстракции**
#
# 9. **Форвард-хуки вместо `output_hidden_states=True`.** Сейчас модель
#    материализует все `N_LAYERS + 1` тензоров `[B, T, H]`; хуки на трёх нужных
#    слоях снимают этот пик и позволяют поднять batch_size в разы.
# 10. **Сортировка по длине глобально, а не внутри чанка.** Сейчас паддинг
#     выравнивается внутри 256 строк; глобальная сортировка сократила бы его ещё.
# 11. **Динамический batch по числу токенов**, а не по числу строк.
# 12. **Пропустить порядок `ba`** для части строк: TTA даёт меньше, чем стоит.
#
# **Инфраструктура**
#
# 13. **`StratifiedGroupKFold` по промпту.** 3 118 промптов повторяются, в них
#     8 861 строка (15,4%). Сейчас копии одного промпта могут разъехаться по
#     фолдам, и валидация чуть оптимистична. Перенарезка обесценивает кэш, так
#     что это решение надо принять до полного прогона.
# 14. **Хранить кэш в float16 на отдельном датасете Kaggle**, чтобы части A и B
#     жили в разных ноутбуках и B не зависела от квоты GPU.
