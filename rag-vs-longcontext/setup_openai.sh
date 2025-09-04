#!/bin/bash
# 用法: ./setup_openai.sh sk-你的key https://你的base/v1 gpt-4o-mini

API_KEY=$1
BASE_URL=$2
MODEL=${3:-gpt-4o-mini}   # 如果没传第三个参数，默认 gpt-4o-mini

if [ -z "$API_KEY" ]; then
  echo "❌ 请传入 API Key，例如:"
  echo "   ./setup_openai.sh sk-xxxx https://api.openai.com/v1"
  exit 1
fi

# 追加到 ~/.bashrc
{
  echo ""
  echo "# >>> OpenAI API config >>>"
  echo "export OPENAI_API_KEY=\"$API_KEY\""
  if [ -n "$BASE_URL" ]; then
    echo "export OPENAI_BASE_URL=\"$BASE_URL\""
  fi
  echo "export OPENAI_MODEL=\"$MODEL\""
  echo "# <<< OpenAI API config <<<"
} >> ~/.bashrc

# 让配置立即生效
source ~/.bashrc

echo "✅ 已写入 ~/.bashrc 并生效"
echo "   OPENAI_API_KEY=${API_KEY:0:8}..."
echo "   OPENAI_BASE_URL=${BASE_URL:-默认 https://api.openai.com/v1}"
echo "   OPENAI_MODEL=$MODEL"
