
call Setup_Python_Paths.bat

call Config_Variables.bat


call cd %TF_OD_Path%

call tensorboard --logdir=../../../../Data/TensorflowOutput

call cd %TF_OD_PATH_RETURN%

