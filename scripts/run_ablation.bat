@echo off
REM 消融实验脚本 (Windows)

echo ==========================================
echo 运行Transformer消融实验
echo ==========================================

cd src

REM 运行消融实验
python ablation_study.py ^
    --experiments all ^
    --save_dir ../results/ablation

echo 消融实验完成！
echo 结果保存在: results\ablation

cd ..
pause
