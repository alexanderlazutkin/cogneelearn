#!/bin/bash
# ~/AI/run-llamaserver-all.sh - адаптированный скрипт для Ubuntu
# chmod +x run-llamaserver-all.sh
# sudo radeontop  + GUI CoreCtl

set -e  # Остановка при ошибке

# Переход в папку с llama.cpp (укажите свой путь)
# cd ~/llama.cpp || { echo "Ошибка: папка не найдена"; exit 1; }

echo "============================================================"
echo "Запуск llama-server "
echo "Порт: 1234"
echo "Модели будут удерживаться в памяти"
echo "Логи запросов появятся ниже"
echo "============================================================"
echo ""

# Создаём временный файл с пресетами (абсолютные пути)
PRESETS_FILE="presets_claude_all.ini"
cat > "$PRESETS_FILE" << EOF
[qwen3.6-27b-mtp]
model = /media/lan/LLM/Models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q6_K_XL.gguf
mmproj = /media/lan/LLM/Models/unsloth/Qwen3.6-27B-MTP-GGUF/mmproj-F32.gguf
n-gpu-layers = 99
n-gpu-layers-draft = 99
# parallel = 2: 2 слота обработки. Синхронизировано с COGNEE_DATA_PER_BATCH=2 в .env.
# Каждый слот держит отдельный prompt-cache (~233 MiB для 27B при ctx 200k) — следите за VRAM.
# Если памяти мало — верните parallel = 1 и поставьте COGNEE_DATA_PER_BATCH=1 в .env.
parallel = 2
ctx-size = 200000
cache-type-k = q8_0
cache-type-v = q8_0
flash-attn = on
temp = 0.6
top-p = 0.95
top-k = 20
presence-penalty = 1.1
min-p = 0
spec-type = draft-mtp
spec-draft-n-max = 2
load-on-startup = false

[qwen3.6-35b-a3b-mtp]
model = /media/lan/LLM/Models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
mmproj = /media/lan/LLM/Models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/mmproj-F32.gguf
n-gpu-layers = 99
n-gpu-layers-draft = 99
parallel = 1
ctx-size = 200000
cache-type-k = q8_0
cache-type-v = q8_0
flash-attn = on
temp = 0.6
top-p = 0.95
top-k = 20
presence-penalty = 1.1
min-p = 0
spec-type = draft-mtp
spec-draft-n-max = 2
load-on-startup = true

[qwen3-embedding-8b-q8]
model = /media/lan/LLM/Models/Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf
n-gpu-layers = 99
parallel = 1
ctx-size = 32768
cache-type-k = q8_0
cache-type-v = q8_0
embedding = true
pooling = last
n-predict = 0
load-on-startup = true
EOF

echo "[INFO] Файл пресетов создан: $PRESETS_FILE"
echo ""

# Запуск сервера
# При необходимости включите глобальную нормализацию эмбеддингов, добавив --normalize-embeddings
llama-server \
    --models-preset "$PRESETS_FILE" \
    --host 127.0.0.1 --port 1234 \
    -ngl 99 \
    --timeout 36000

# Если сервер завершился с ошибкой
echo ""
echo "Сервер остановлен. Нажмите Enter для выхода..."
read