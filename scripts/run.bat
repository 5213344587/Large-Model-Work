@echo off
REM Transformer训练脚本 (Windows)
REM 使用方法: scripts\run.bat [task] [experiment_name]

REM 设置默认参数
set TASK=%1
set EXP_NAME=%2
set SEED=42

if "%TASK%"=="" set TASK=copy
if "%EXP_NAME%"=="" set EXP_NAME=default

echo ==========================================
echo 训练Transformer模型
echo 任务: %TASK%
echo 实验名称: %EXP_NAME%
echo 随机种子: %SEED%
echo ==========================================

REM 进入src目录
cd src

REM 运行训练
python train.py ^
    --task %TASK% ^
    --num_samples 10000 ^
    --seq_len 10 ^
    --vocab_size 50 ^
    --d_model 256 ^
    --n_heads 8 ^
    --num_encoder_layers 3 ^
    --num_decoder_layers 3 ^
    --d_ff 1024 ^
    --dropout 0.1 ^
    --batch_size 64 ^
    --num_epochs 50 ^
    --lr 0.0001 ^
    --warmup_steps 4000 ^
    --max_grad_norm 1.0 ^
    --label_smoothing 0.1 ^
    --seed %SEED% ^
    --save_dir ../checkpoints/%EXP_NAME%

echo 训练完成！
echo 模型保存在: checkpoints\%EXP_NAME%
echo 训练曲线保存在: checkpoints\%EXP_NAME%\training_curves.png

cd ..
pause
