call conda activate TF-GPU

call set PYTHONPATH=D:\MachineLearning\Tensorflow\models\;D:\MachineLearning\Tensorflow\models\research;D:\MachineLearning\Tensorflow\models\research\slim

call echo PYTHONPATH

call set PATH=%PATH%;%PYTHONPATH%

call echo %PATH%

call set TFENVSET=TRUE

rem Hide my display GPU, only show the 2 quadro RTX 8000's [WHY WON"T THEY LINK?!]
call set CUDA_VISIBLE_DEVICES=0,1