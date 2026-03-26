@echo off
chcp 65001 >nul
title BioTransLM 可视化训练与对话系统

echo ========================================
echo   BioTransLM 可视化训练与对话系统
echo ========================================
echo.
echo 正在启动服务器...
echo.
echo 提示：浏览器会自动打开，如果没有打开请访问：
echo http://127.0.0.1:7860
echo.
echo 按 Ctrl+C 可停止服务器
echo ========================================
echo.

cd /d "%~dp0"
python ui/app.py --port 7860

pause
